# -*- coding: utf-8 -*-
"""并行部署模型测试：参考规模校验 + Python/JS 一致性。"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from roofline.arch import normalize, param_counts
from roofline.crawler import CACHE_DIR, MODEL_LIST
from roofline.parallel import compute_parallel, validate

TEMPLATE = Path(__file__).resolve().parent.parent / "roofline" / "template.html"
NODE = shutil.which("node")


def load_arch(name: str) -> dict:
    entry = next(e for e in MODEL_LIST if e["name"] == name)
    cfg = json.loads(
        (CACHE_DIR / f"{entry['ns']}__{entry['repo']}.json").read_text(encoding="utf-8")
    )
    return normalize(cfg, entry["name"], entry["family"], entry["note"], entry["repo"])


class ParallelReferenceTest(unittest.TestCase):
    def test_tp8_all_reduce_matches_reference(self):
        # Llama-70B, prefill M=8192, TP=8: 2L 次 all-reduce ≈ 20 GiB/卡
        arch = load_arch("Llama-3.1-70B")
        r = compute_parallel(arch, "prefill", 1, 8192, {"tp": 8, "mode": "replica"})
        self.assertTrue(r["valid"], r["message"])
        self.assertEqual(r["widths"]["world_size"], 8)
        self.assertAlmostEqual(r["per_gpu"]["total"]["comm"] / 2**30, 20.0, places=1)
        self.assertTrue(r["summary"]["fit"])

    def test_dsv3_attn_ep_comm_matches_reference(self):
        # DeepSeek-V3, TP8+DP8+EP8, prefill M=8192:
        # DP gather 6.7 GiB + EP 8.3 GiB + 少量 dense TP ≈ 15 GiB/卡
        arch = load_arch("DeepSeek-V3")
        r = compute_parallel(
            arch, "prefill", 1, 8192,
            {"tp": 8, "dp": 8, "ep": 8, "mode": "attn", "mem": 80e9},
        )
        self.assertTrue(r["valid"], r["message"])
        self.assertEqual(r["widths"]["attn_tp"], 1)
        self.assertEqual(r["widths"]["world_size"], 8)
        comm_gib = r["per_gpu"]["total"]["comm"] / 2**30
        self.assertGreater(comm_gib, 14.0)
        self.assertLess(comm_gib, 17.0)
        self.assertFalse(r["summary"]["fit"])  # BF16 权重 > 80GB，需更多卡

    def test_validation_errors(self):
        arch = load_arch("Qwen2.5-7B")
        ok, msg, _, _ = validate({"tp": 64, "mode": "replica"}, arch, "decode")
        self.assertFalse(ok)
        self.assertIn("TP", msg)
        ok, msg, _, _ = validate({"tp": 8, "pp": 3, "mode": "replica"}, arch, "decode")
        self.assertFalse(ok)
        self.assertIn("PP", msg)

    def test_world_size_and_kv_replication(self):
        arch = load_arch("Qwen2.5-7B")  # KVH=4
        r = compute_parallel(arch, "decode", 1, 2048, {"tp": 8, "mode": "replica"})
        self.assertTrue(r["valid"], r["message"])
        self.assertEqual(r["widths"]["kv_tp"], 4)
        self.assertEqual(r["summary"]["world_size"], 8)


class ParallelJsParityTest(unittest.TestCase):
    @unittest.skipUnless(NODE, "需要 Node.js")
    def test_python_js_parallel_parity(self):
        src = TEMPLATE.read_text(encoding="utf-8")
        start = src.index("const WB = 2.0")
        end = src.index("// 缓存:")
        engine = src[start:end]
        pstart = src.index("function parCfg(){")
        pend = src.index("// ---------- 渲染: 总览矩阵 ----------")
        parallel_js = src[pstart:pend]

        cases = [
            ("Llama-3.1-70B", "prefill", 1, 8192, {"tp": 8, "mode": "replica"}),
            ("DeepSeek-V3", "prefill", 1, 8192, {"tp": 8, "dp": 8, "ep": 8, "mode": "attn"}),
            ("DeepSeek-V3", "decode", 128, 2048, {"tp": 8, "dp": 8, "ep": 8, "mode": "attn"}),
            ("Qwen2.5-7B", "decode", 32, 8192, {"tp": 8, "cp": 1, "dcp": 2, "mode": "replica"}),
            ("Qwen2.5-7B", "prefill", 1, 8192, {"tp": 8, "pp": 2, "mode": "replica"}),
            ("Qwen3.5-35B-A3B", "decode", 8, 4096, {"tp": 4, "dp": 4, "ep": 4, "mode": "attn"}),
        ]
        payload = []
        for name, phase, B, S, cfg in cases:
            arch = load_arch(name)
            payload.append({
                "A": arch, "phase": phase, "B": B, "S": S, "cfg": cfg,
                "weightBytes": param_counts(arch)["total"] * 2.0,
            })
        script = (
            engine
            + "\nconst DATA = {stages:['embed','qkv','attn','oproj','mlp','router','lmhead','elemwise'].map(id=>({id}))};\n"
            + "const fmtBytes = v => String(v);\n"
            + parallel_js
            + "\nconst CASES = " + json.dumps(payload, ensure_ascii=False) + ";\n"
            + "const out = CASES.map(c => computeParallelJS(c.A,c.phase,c.B,c.S,c.cfg,c.weightBytes));\n"
            + "process.stdout.write(JSON.stringify(out));\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parallel_parity.js"
            path.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                [NODE, str(path)], capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
        if proc.returncode != 0:
            self.fail(f"Node 执行失败:\n{proc.stderr[:3000]}")
        js_out = json.loads(proc.stdout)
        self.assertEqual(len(js_out), len(cases))

        for i, (name, phase, B, S, cfg) in enumerate(cases):
            arch = load_arch(name)
            py = compute_parallel(arch, phase, B, S, cfg)
            js = js_out[i]
            with self.subTest(model=name, phase=phase, cfg=cfg):
                self.assertEqual(py["valid"], js["valid"])
                if not py["valid"]:
                    continue
                for sid, pv in py["per_gpu"].items():
                    jv = js["total"] if sid == "total" else js["perGpu"][sid]
                    for key in ("flops", "dram", "comm", "intensity", "intensity_total"):
                        self.assertTrue(
                            math.isclose(pv[key], jv[key], rel_tol=1e-9, abs_tol=1.0),
                            f"{sid}.{key}: py={pv[key]} js={jv[key]}",
                        )
                self.assertTrue(
                    math.isclose(
                        py["summary"]["comm_per_token"], js["summary"]["commPerToken"],
                        rel_tol=1e-9, abs_tol=1.0,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        py["summary"]["weight_per_card"], js["summary"]["weightPerCard"],
                        rel_tol=1e-9, abs_tol=1.0,
                    )
                )
                self.assertEqual(py["summary"]["world_size"], js["summary"]["worldSize"])


if __name__ == "__main__":
    unittest.main()
