# -*- coding: utf-8 -*-
"""并行部署下的每卡 Roofline 估算（TP / DP / EP / CP / PP / DCP）。

模型是**一阶估算**，目标是把“单卡理想 Roofline”推进到“给定并行策略下每卡
FLOPs / DRAM / 通信”的可比较视图；不模拟 kernel、负载不均和通信重叠。

约定:
  * ``B`` 是全局 batch；普通 DP 的每个副本处理 ``B/dp`` 个请求。
  * TP 切权重/计算；GQA 的 KV 头最多切到 ``KVH``，MLA 的 latent KV 不随 TP 切分。
  * EP 与 MoE-TP 共同组成 TP（``ep * moe_tp = tp``），因此每卡 MoE FLOPs/权重仍按 TP 缩放；
    EP 主要改变 all-to-all 通信与每卡专家数。
  * DP-Attention 的 DP 从 TP 中除出来，``attn_tp = tp / dp / cp``；KV 按请求切，权重在 DP 间复制。
  * CP 仅 prefill 生效，DCP 仅 decode 生效；PP 按层切分。

参考: ``C:\\workspace\\sglang\\sglang-parallelism-strategies.zh.md`` §3/§5/§6/§7/§8/§14。
"""
from __future__ import annotations

import math

from .arch import param_counts
from .compute import STAGE_ORDER
from .stages import AB, KVB, WB, compute_stages

DEFAULT_CFG = {
    "tp": 1, "dp": 1, "ep": 1, "cp": 1, "pp": 1, "dcp": 1,
    "mode": "replica", "bw": 3350e9, "mem": 80e9,
}


def _int(cfg: dict, key: str) -> int:
    try:
        return int(cfg.get(key, 1))
    except (TypeError, ValueError):
        return 1


def derive(cfg: dict, A: dict, phase: str) -> dict:
    """派生并行宽度、卡数与 KV 切分方式（不修改 cfg）。"""
    tp = max(1, _int(cfg, "tp"))
    dp = max(1, _int(cfg, "dp"))
    ep = max(1, _int(cfg, "ep"))
    pp = max(1, _int(cfg, "pp"))
    cp = max(1, _int(cfg, "cp")) if phase == "prefill" else 1
    dcp = max(1, _int(cfg, "dcp")) if phase == "decode" else 1
    mode = "attn" if cfg.get("mode") == "attn" else "replica"

    attn_dp = dp if mode == "attn" else 1
    attn_tp = max(1, tp // max(1, attn_dp) // max(1, cp)) if mode == "attn" else max(1, tp)
    world = tp * pp if mode == "attn" else tp * dp * pp
    moe_tp = max(1, tp // ep) if ep else tp
    kv_tp = 1 if A.get("mla") else max(1, min(attn_tp if mode == "attn" else tp, A.get("KVH", tp)))

    return {
        "tp": tp, "dp": dp, "ep": ep, "cp": cp, "pp": pp, "dcp": dcp,
        "mode": mode, "attn_dp": attn_dp, "attn_tp": attn_tp,
        "moe_tp": moe_tp, "kv_tp": kv_tp, "world_size": world,
        "layers_per_stage": A["L"] / pp,
    }


def validate(cfg: dict, A: dict, phase: str) -> tuple[bool, str, list[str], dict]:
    """校验并行配置；返回 (ok, error, warnings, widths)。"""
    d = derive(cfg, A, phase)
    errors: list[str] = []
    warnings: list[str] = []

    if d["tp"] > A["H"]:
        errors.append(f"TP={d['tp']} 超过注意力头数 H={A['H']}")
    if A["L"] % d["pp"] != 0:
        errors.append(f"PP={d['pp']} 必须整除层数 L={A['L']}")
    if d["ep"] > d["tp"] or d["tp"] % d["ep"] != 0:
        errors.append(f"EP={d['ep']} 必须满足 1 ≤ EP ≤ TP 且整除 TP={d['tp']}")
    if phase == "prefill" and d["cp"] > 1 and d["tp"] % d["cp"] != 0:
        errors.append(f"CP={d['cp']} 必须整除 TP={d['tp']}")
    if phase == "decode" and d["dcp"] > 1 and d["tp"] % d["dcp"] != 0:
        errors.append(f"DCP={d['dcp']} 必须整除 TP={d['tp']}")
    if d["mode"] == "attn" and d["tp"] % (d["attn_dp"] * d["cp"]) != 0:
        errors.append(
            f"DP-Attention 要求 TP % (DP×CP) == 0 "
            f"(TP={d['tp']}, DP={d['attn_dp']}, CP={d['cp']})"
        )
    if d["pp"] > 1:
        warnings.append("PP 按层均分；embedding 只在首 stage、LM Head 只在末 stage，当前按均值近似")
    if d["mode"] == "attn" and d["ep"] > 1 and A.get("moe"):
        warnings.append("MoE EP 的 all-to-all 与 MLP 的 TP all-reduce 可能走不同路径，通信量按两者相加的保守上界估算")
    if d["tp"] > d["kv_tp"] and not A.get("mla"):
        warnings.append(f"TP={d['tp']} > KV 头数 {A['KVH']}，超出的卡复制 KV cache，KV 不再随 TP 缩小")
    if d["ep"] > 1 and not A.get("moe"):
        warnings.append("模型不是 MoE，EP 不生效")
    if d["mode"] == "replica" and d["dp"] > 1:
        warnings.append("普通 DP 是独立副本：每卡 Roofline 不变，DP 只增加集群吞吐；通信不计副本间流量")
    if d["mode"] == "attn" and d["dp"] > 1:
        warnings.append("DP-Attention 下注意力权重在 DP 间复制，每卡权重按总权重 / TP 的一阶下界估算")
    if d["tp"] > 1:
        warnings.append("TP all-reduce 按逻辑字节计；ring 实际收发约为逻辑量的 2(TP−1)/TP")

    if errors:
        return False, "; ".join(errors), warnings, d
    return True, "", warnings, d


def _kv_width(A: dict) -> float:
    mla = A.get("mla")
    if mla:
        return mla["kv_lora"] + mla["rope_d"]
    return 2 * A["KVH"] * A["d"]


def _kv_per_card(A: dict, d: dict, B: int, S: int, phase: str) -> int:
    """当前场景下每卡 KV cache / 线性注意力状态的近似字节数。"""
    if phase != "decode":
        return 0
    req = B / d["dp"] if d["mode"] == "attn" else B
    full = A["hybrid"]["L_full"] if A.get("hybrid") else A["L"]
    if A.get("mla"):
        elems = A["mla"]["kv_lora"] + A["mla"]["rope_d"]
        per_token = elems * KVB * full
    else:
        per_token = A["KVH"] * A["d"] * KVB * full
    kv = req * S * per_token / max(1, d["kv_tp"]) / max(1, d["dcp"])
    if A.get("hybrid"):
        h = A["hybrid"]
        state = req * h["state_elems"] * KVB * 2 * (A["L"] - h["L_full"])
        kv += state
    return int(kv)


def compute_parallel(A: dict, phase: str, B: int, S: int, cfg: dict) -> dict:
    """返回给定并行策略下每卡的阶段 FLOPs / DRAM / 通信与部署摘要。"""
    ok, msg, warns, d = validate(cfg, A, phase)
    if not ok:
        return {
            "valid": False, "message": msg, "warnings": warns,
            "phase": phase, "B": B, "S": S, "widths": d,
            "per_gpu": {}, "total": {},
        }

    g = compute_stages(A, phase, B, S)
    tp, dp, ep, pp = d["tp"], d["dp"], d["ep"], d["pp"]
    cp, dcp, mode = d["cp"], d["dcp"], d["mode"]
    attn_tp = d["attn_tp"]
    L = A["L"]
    Lp = L / pp
    M = B * S if phase == "prefill" else B
    # 普通 DP 的 B 是「每副本 batch」；DP-Attention 的 B 是全局 batch，注意力按 dp 切
    M_dp = M / dp if mode == "attn" else M
    M_mlp = M

    base_div = tp * pp
    qkv_div = pp * (attn_tp if mode == "attn" else tp)
    kv_div = d["kv_tp"] if not A.get("mla") else 1
    attn_div = pp * (dp if mode == "attn" else 1) * kv_div

    # ---------- 通信字节 (每卡) ----------
    moe = A.get("moe")
    comm = {sid: 0.0 for sid in STAGE_ORDER}
    # TP all-reduce: 每个注意力/MLP 块每层各 1 次，共 2 次/层
    if tp > 1:
        ar = Lp * M_mlp * A["h"] * AB
        if mode == "replica" or attn_tp > 1:
            comm["oproj"] += ar
        if moe and ep > 1:
            # MoE 段走 EP all-to-all；只有前 first_dense 个 dense 层仍有 MLP all-reduce
            comm["mlp"] += (moe["first_dense"] / pp) * M_mlp * A["h"] * AB
        else:
            comm["mlp"] += ar
    # DP-Attention gather (SUM_LEN)
    if mode == "attn" and dp > 1:
        comm["mlp"] += Lp * M * A["h"] * AB
    # MoE EP all-to-all: dispatch fp8 (1 B) + combine bf16 (2 B)
    if moe and ep > 1 and phase in ("prefill", "decode"):
        L_moe = L - moe["first_dense"]
        dispatch = (L_moe / pp) * M_dp * moe["K"] * A["h"] * 1.0 * (1.0 - 1.0 / ep)
        comm["mlp"] += 3.0 * dispatch
    # PP p2p: 每个 stage 边界一次
    if pp > 1:
        comm["mlp"] += 2.0 * (pp - 1) * M * A["h"] * AB
    # Prefill-CP: 每层交换 KV
    if phase == "prefill" and cp > 1:
        full_layers = A["hybrid"]["L_full"] if A.get("hybrid") else L
        comm["attn"] += full_layers * (S * _kv_width(A) * AB / cp)
    # DCP: 每层归并部分注意力结果
    if phase == "decode" and dcp > 1:
        full_layers = A["hybrid"]["L_full"] if A.get("hybrid") else L
        comm["attn"] += full_layers * M_dp * A["H"] * AB

    # ---------- 每卡每段 ----------
    pg: dict[str, dict] = {}
    for sid in STAGE_ORDER:
        if sid not in g:
            continue
        div = qkv_div if sid in ("qkv", "oproj") else (attn_div if sid == "attn" else base_div)
        flops = g[sid]["flops"] / base_div
        dram = g[sid]["bytes"] / max(1e-9, div)
        c = comm.get(sid, 0.0)
        pg[sid] = {
            "flops": flops,
            "dram": dram,
            "comm": c,
            "intensity": flops / dram if dram else 0.0,
            "intensity_total": flops / (dram + c) if (dram + c) else 0.0,
        }

    tot_f = sum(v["flops"] for v in pg.values())
    tot_d = sum(v["dram"] for v in pg.values())
    tot_c = sum(v["comm"] for v in pg.values())
    pg["total"] = {
        "flops": tot_f, "dram": tot_d, "comm": tot_c,
        "intensity": tot_f / tot_d if tot_d else 0.0,
        "intensity_total": tot_f / (tot_d + tot_c) if (tot_d + tot_c) else 0.0,
    }

    pc = param_counts(A)
    weight_per_card = pc["total"] * WB / max(1, tp * pp)
    kv_per_card = _kv_per_card(A, d, B, S, phase)
    mem = float(cfg.get("mem", 80e9))
    fit = weight_per_card <= mem * 0.9
    total_weight = pc["total"] * WB
    min_cards = max(1, math.ceil(total_weight / (mem * 0.9)))
    if not fit:
        warns = warns + [
            f"每卡权重 {weight_per_card/1e9:.1f} GB 超过 {mem/1e9:.0f} GB×90%，当前切分放不下；"
            f"按总权重至少需要 {min_cards} 张卡"
        ]

    per_token = (tot_d + tot_c) / (B if phase == "decode" else max(1, B * S))
    return {
        "valid": True, "message": "", "warnings": warns,
        "phase": phase, "B": B, "S": S, "widths": d,
        "per_gpu": pg, "total": pg["total"],
        "summary": {
            "world_size": d["world_size"],
            "weight_per_card": weight_per_card,
            "kv_per_card": kv_per_card,
            "comm_per_token": tot_c / (B if phase == "decode" else max(1, B * S)),
            "bytes_per_token": per_token,
            "fit": fit,
            "min_cards": min_cards,
            "mem": mem,
            "tp": tp, "dp": dp, "ep": ep, "cp": cp, "pp": pp, "dcp": dcp,
            "mode": mode, "attn_tp": attn_tp, "kv_tp": d["kv_tp"],
        },
    }
