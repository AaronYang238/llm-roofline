# -*- coding: utf-8 -*-
"""Python 公式引擎与报告内 JS 公式引擎的数值一致性测试。

报告声称 Python `stages.py` 与 `template.html` 中的 `computeStagesJS` 完全一致，
这里用 Node 实际执行模板里的 JS 函数并对全部模型/多组场景做交叉校验。
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from roofline.arch import normalize
from roofline.compute import STAGE_ORDER
from roofline.crawler import CACHE_DIR, MODEL_LIST
from roofline.stages import compute_stages

TEMPLATE = Path(__file__).resolve().parent.parent / "roofline" / "template.html"
NODE = shutil.which("node")


def _extract_js_engine() -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    start = src.index("const WB = 2.0")
    end = src.index("// 缓存:")
    return src[start:end]


class JsParityTest(unittest.TestCase):
    @unittest.skipUnless(NODE, "需要 Node.js 才能校验 JS/Python 公式一致性")
    def test_all_models_and_scenarios_match(self):
        archs = []
        for entry in MODEL_LIST:
            cfg = json.loads(
                (CACHE_DIR / f"{entry['ns']}__{entry['repo']}.json").read_text(encoding="utf-8")
            )
            archs.append(
                normalize(cfg, entry["name"], entry["family"], entry["note"], entry["repo"])
            )

        cases = [
            ("prefill", 1, 128),
            ("prefill", 1, 2048),
            ("prefill", 1, 32768),
            ("decode", 1, 2048),
            ("decode", 32, 8192),
        ]
        js = _extract_js_engine()
        script = (
            js
            + "\nconst MODELS = " + json.dumps(archs, ensure_ascii=False) + ";\n"
            + "const CASES = " + json.dumps(cases) + ";\n"
            + "const out = MODELS.map(A => CASES.map(([p,b,s]) => computeStagesJS(A,p,b,s)));\n"
            + "process.stdout.write(JSON.stringify(out));\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parity.js"
            path.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                [NODE, str(path)], capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
        if proc.returncode != 0:
            self.fail(f"Node 执行失败:\n{proc.stderr[:2000]}")

        js_out = json.loads(proc.stdout)
        self.assertEqual(len(js_out), len(archs))

        for mi, arch in enumerate(archs):
            for ci, (phase, B, S) in enumerate(cases):
                py = compute_stages(arch, phase, B, S)
                js_st = js_out[mi][ci]
                for sid in STAGE_ORDER + ["total"]:
                    with self.subTest(model=arch["name"], phase=phase, B=B, S=S, stage=sid):
                        in_py, in_js = sid in py, sid in js_st
                        if not in_py and not in_js:
                            continue
                        self.assertEqual(in_py, in_js, f"阶段存在性不一致: {sid}")
                        self.assertTrue(
                            math.isclose(py[sid]["flops"], js_st[sid]["flops"], rel_tol=1e-9, abs_tol=1.0),
                            f"FLOPs 不一致: py={py[sid]['flops']} js={js_st[sid]['flops']}",
                        )
                        self.assertTrue(
                            math.isclose(py[sid]["bytes"], js_st[sid]["bytes"], rel_tol=1e-9, abs_tol=1.0),
                            f"Bytes 不一致: py={py[sid]['bytes']} js={js_st[sid]['bytes']}",
                        )


if __name__ == "__main__":
    unittest.main()
