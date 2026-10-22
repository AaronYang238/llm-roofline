# -*- coding: utf-8 -*-
"""构建报告数据 (强度立方 cube + 模型统计 stats)。"""
from __future__ import annotations

from .arch import normalize, param_counts
from .stages import compute_stages

# ---------- GPU 设备 (BF16/FP16 稠密张量核算力, HBM 带宽) ----------
# MI308X (gfx942, 80 CU × 1850 MHz): BF16 303.1 / FP8 606.2 TFLOP/s, HBM 5325 GB/s
# 脊点 ≈ 56.9 FLOP/B —— 带宽与显存容量满血(MI300X 同级)、算力仅 MI300X 的 23%
DEVICES = [
    {"id": "v100", "name": "V100-SXM2", "peak": 125e12, "bw": 900e9, "mem": 32e9},
    {"id": "a100", "name": "A100-SXM4", "peak": 312e12, "bw": 2039e9, "mem": 80e9},
    {"id": "h100", "name": "H100-SXM", "peak": 989e12, "bw": 3350e9, "mem": 80e9},
    {"id": "h20", "name": "H20", "peak": 148e12, "bw": 4000e9, "mem": 96e9},
    {"id": "mi308x", "name": "MI308X", "peak": 303.1e12, "bw": 5325e9, "mem": 192e9},
    {"id": "b200", "name": "B200", "peak": 2250e12, "bw": 8000e9, "mem": 192e9},
]

# ---------- 场景网格 ----------
PREFILL_S = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DECODE_S = [128, 512, 2048, 8192, 32768]

DEFAULT_S = 2048
DEFAULT_B = 1

# ---------- 阶段元数据 ----------
STAGES = [
    {
        "id": "embed",
        "name": "嵌入查表",
        "desc": "Embedding 行查表: 纯访存, FLOPs≈0",
    },
    {
        "id": "qkv",
        "name": "QKV 投影",
        "desc": "注意力输入投影 (Q/K/V, MLA 含压缩/解压): prefill 为 GEMM, decode 为 GEMV",
    },
    {
        "id": "attn",
        "name": "注意力核心",
        "desc": "QK^T + AV + softmax: prefill 按 FlashAttention 计 Q/K/V 读 + O 写; decode 每步全量读 KV 缓存",
    },
    {
        "id": "oproj",
        "name": "O 投影",
        "desc": "注意力输出投影 (attn out projection)",
    },
    {
        "id": "mlp",
        "name": "MLP/FFN",
        "desc": "dense: gate/up/down SwiGLU; MoE: 每 token K 个路由专家 (+共享专家)",
    },
    {
        "id": "router",
        "name": "MoE 路由",
        "desc": "MoE 门控网络 (h×E) + top-k 选择 (仅 MoE 模型)",
    },
    {
        "id": "lmhead",
        "name": "LM Head",
        "desc": "词表投影 (h×V): prefill 只算每序列最后一个 token; decode 每步 B 个 token",
    },
    {
        "id": "elemwise",
        "name": "元素级/归一化",
        "desc": "RMSNorm×2 + 残差相加×2 + RoPE + SiLU·mul (估算: 读+写各一次)",
    },
]

STAGE_ORDER = [
    "embed",
    "qkv",
    "attn",
    "oproj",
    "mlp",
    "router",
    "lmhead",
    "elemwise",
]


def build_cube(A: dict) -> dict:
    """对场景网格中的每个点计算各阶段强度。"""
    cube = {"prefill": {}, "decode": {}}

    for S in PREFILL_S:
        st = compute_stages(A, "prefill", B=1, S=S)
        cube["prefill"][str(S)] = {
            sid: (st[sid]["intensity"] if sid in st else None)
            for sid in STAGE_ORDER + ["total"]
        }

    for S in DECODE_S:
        cube["decode"][str(S)] = {}
        for B in DECODE_B:
            st = compute_stages(A, "decode", B=B, S=S)
            cube["decode"][str(S)][str(B)] = {
                sid: (st[sid]["intensity"] if sid in st else None)
                for sid in STAGE_ORDER + ["total"]
            }

    return cube


def build_stats(A: dict) -> dict:
    """模型级统计: 参数量、每 token FLOPs / 访存。"""
    pc = param_counts(A)
    st = compute_stages(A, "decode", B=DEFAULT_B, S=DEFAULT_S)

    flops_per_token = st["total"]["flops"] / DEFAULT_B
    bytes_per_token = st["total"]["bytes"] / DEFAULT_B

    return {
        "total_params": pc["total"],
        "active_params": pc["active"],
        "weight_bytes": pc["total"] * 2.0,
        "flops_per_token": flops_per_token,
        "bytes_per_token": bytes_per_token,
    }
