# -*- coding: utf-8 -*-
"""架构归一化与关键公式的回归测试。"""
from __future__ import annotations

import json
import unittest

from roofline.arch import data_warnings, normalize
from roofline.compute import build_stats
from roofline.crawler import CACHE_DIR, MODEL_LIST
from roofline.stages import compute_stages


def load_arch(name: str) -> tuple[dict, dict]:
    entry = next(e for e in MODEL_LIST if e["name"] == name)
    cfg = json.loads(
        (CACHE_DIR / f"{entry['ns']}__{entry['repo']}.json").read_text(encoding="utf-8")
    )
    arch = normalize(cfg, entry["name"], entry["family"], entry["note"], entry["repo"])
    return arch, build_stats(arch)


class ArchNormalizationTest(unittest.TestCase):
    def test_moe_topk_does_not_use_sampling_top_k(self):
        # Kimi-K3 config 同时有 num_experts_per_token=16 与采样 top_k=50;
        # 旧实现会误取 top_k, 把激活专家数放大 3 倍。
        arch, _ = load_arch("Kimi-K3")
        self.assertEqual(arch["moe"]["K"], 16)
        self.assertFalse(arch["moe"]["k_inferred"])

    def test_expert_hidden_dim_preferred(self):
        # Hy-Memory-30B-A3B: moe_intermediate_size=1536, expert_hidden_dim=768;
        # 后者才能得到约 30B 总参数。
        arch, stats = load_arch("Hy-Memory-30B-A3B")
        self.assertEqual(arch["moe"]["Ie"], 768)
        self.assertLess(stats["total_params"] / 1e9, 31.0)
        self.assertGreater(stats["total_params"] / 1e9, 29.0)

    def test_shared_expert_inferred_from_size(self):
        # Qwen3.5/Holo3 只给 shared_expert_intermediate_size, 没有 num_shared_experts
        arch, _ = load_arch("Qwen3.5-35B-A3B")
        self.assertEqual(arch["moe"]["shared_n"], 1)
        self.assertEqual(arch["moe"]["Is"], 512)

    def test_lm_head_counted_in_active_params(self):
        # untied LM head 在 decode 每一步都会读取, 必须进入 active 参数
        arch, stats = load_arch("Qwen2.5-7B")
        self.assertFalse(arch["tie"])
        self.assertEqual(stats["active_params"], stats["total_params"])

    def test_gemma_tie_default_from_note(self):
        # Gemma 2 config 未声明 tie_word_embeddings, 模型清单标注 tied
        arch, stats = load_arch("Gemma-2-9B")
        self.assertTrue(arch["tie"])
        self.assertAlmostEqual(stats["total_params"] / 1e9, 9.24, places=2)

    def test_partial_mla_is_warned(self):
        arch, stats = load_arch("DeepSeek-V4-Pro")
        warnings = data_warnings(arch, stats)
        self.assertTrue(any("kv_lora_rank" in w for w in warnings))

    def test_attention_counts_qk_and_av(self):
        arch, _ = load_arch("Qwen2.5-7B")
        h, L, H, d = arch["h"], arch["L"], arch["H"], arch["d"]

        prefill = compute_stages(arch, "prefill", B=1, S=128)
        expected_prefill = 2 * 1 * H * (2 * d) * 128 * 128 * 0.5 * L
        self.assertAlmostEqual(prefill["attn"]["flops"], expected_prefill)

        decode = compute_stages(arch, "decode", B=1, S=2048)
        expected_decode = 2 * 1 * H * (2 * d) * 2048 * L
        self.assertAlmostEqual(decode["attn"]["flops"], expected_decode)


if __name__ == "__main__":
    unittest.main()
