# -*- coding: utf-8 -*-
"""各推理阶段 (FLOPs / 访存 Bytes) 的解析计算模型。

符号:
  h=hidden_size, L=层数, H=注意力头数, KVH=KV 头数, d=head_dim,
  V=词表, I=FFN 中间维度, B=批量, S=序列(上下文)长度

统一假设:
  * 权重 BF16 (WB=2 B/参数), 激活 BF16 (AB=2 B/元素), KV 缓存 BF16 (KVB=2 B/元素)
  * 权重: 每个 GEMM/GEMV pass 读一次 (整批共享)
  * 激活: 每阶段输入读一次、输出写一次 (阶段之间不重复计数)
  * Prefill 注意力: FlashAttention 风格, 只计 Q/K/V 读 + O 写; causal 掩码 FLOPs ×0.5
  * Decode 注意力: 每步全量读 KV 缓存
  * MoE: decode 每步激活 n_active = min(E, B×K) 个专家, 其权重各读一次
  * LM head: prefill 只对每序列最后一个 token 计算; decode 每步 B 个 token
"""
from __future__ import annotations

WB = 2.0   # bytes / weight param (BF16)
AB = 2.0   # bytes / activation element
KVB = 2.0  # bytes / KV cache element


def _stage(flops: float, bytes_: float) -> dict:
    return {"flops": flops, "bytes": bytes_, "intensity": (flops / bytes_) if bytes_ else float("inf")}


def compute_stages(A: dict, phase: str, B: int = 1, S: int = 2048) -> dict:
    """返回 {stage_id: {"flops", "bytes", "intensity"}} (含 'total')。

    phase='prefill': B=批, S=提示长度 -> 一次完整前向
    phase='decode':  B=批, S=KV 缓存长度 -> 一步 decode
    """
    h, L, H, KVH, d, V = A["h"], A["L"], A["H"], A["KVH"], A["d"], A["V"]
    moe = A["moe"]
    mla = A["mla"]
    hybrid = A.get("hybrid")  # 混合注意力: 仅 L_full 层累积 KV 缓存
    dsa = A.get("dsa")        # DSA 稀疏注意力: decode 只读 topk 条 KV

    tokens = B * S if phase == "prefill" else B

    # 层数划分: 混合模型只有 L_full 层做全注意力 (读 KV 缓存)
    L_full = hybrid["L_full"] if hybrid else L
    L_lin = L - L_full

    if mla:
        m = mla
        qkv_params = (
            h * m["q_lora"]
            + m["q_lora"] * H * (m["nope_d"] + m["rope_d"])
            + h * (m["kv_lora"] + m["rope_d"])
            + m["kv_lora"] * H * (m["nope_d"] + m["v_d"])
        )
        q_out = H * (m["nope_d"] + m["rope_d"])
        o_params = H * m["v_d"] * h
        o_in = H * m["v_d"]
        qk_pair = (m["nope_d"] + m["rope_d"]) + m["v_d"]   # 每对 (q,k) 每头 FLOPs 数
        kv_cache_elems = m["kv_lora"] + m["rope_d"]
    else:
        qkv_params = h * (H + 2 * KVH) * d
        q_out = (H + 2 * KVH) * d
        o_params = h * H * d
        o_in = H * d
        qk_pair = 2 * d
        kv_cache_elems = KVH * d

    st: dict[str, dict] = {}

    # ---------- 1. 嵌入查表 (纯访存) ----------
    st["embed"] = _stage(0.0, tokens * h * (WB + AB))

    # ---------- 2. QKV 投影 ----------
    qkv_flops = 2 * qkv_params * tokens * L
    qkv_bytes = qkv_params * WB * L + tokens * L * (h + q_out) * AB
    st["qkv"] = _stage(qkv_flops, qkv_bytes)

    # ---------- 3. 注意力核心 ----------
    # 线性注意力层 (KDA/GDN): 固定状态, 每 token 2·se FLOPs, decode 每步读写状态
    if phase == "prefill":
        # 完整注意力层: causal ×0.5
        attn_flops = B * H * qk_pair * S * S * 0.5 * L_full
        if mla:
            attn_bytes = (B * S * H * (mla["nope_d"] + mla["rope_d"]) * AB
                          + B * S * H * mla["nope_d"] * AB
                          + B * S * H * mla["v_d"] * AB
                          + B * S * H * mla["v_d"] * AB) * L_full
        else:
            attn_bytes = B * S * AB * (2 * H + 2 * KVH) * d * L_full
        if L_lin:
            se = hybrid["state_elems"]
            attn_flops += 2 * se * B * S * L_lin   # 顺序处理 S 个 token
            attn_bytes += B * se * KVB * 2 * L_lin  # 状态读+写一次
    else:
        # decode: 每步对 min(topk, S) 个 KV 做一次 QK^T + AV
        S_eff = min(dsa["topk"], S) if dsa else S
        attn_flops = B * H * qk_pair * S_eff * L_full
        if mla:
            attn_bytes = (B * kv_cache_elems * KVB + B * H * (mla["nope_d"] + mla["rope_d"]) * AB
                          + B * H * mla["v_d"] * AB) * L_full
        else:
            attn_bytes = (2 * B * KVH * d * S_eff * KVB + 2 * B * H * d * AB) * L_full
        if L_lin:
            se = hybrid["state_elems"]
            attn_flops += 2 * se * B * L_lin
            attn_bytes += B * se * KVB * 2 * L_lin
    st["attn"] = _stage(attn_flops, attn_bytes)

    # ---------- 4. 注意力输出投影 (O proj) ----------
    o_flops = 2 * o_params * tokens * L
    o_bytes = o_params * WB * L + tokens * L * (o_in + h) * AB
    st["oproj"] = _stage(o_flops, o_bytes)

    # ---------- 5. MLP / MoE 专家 ----------
    if moe:
        E, K, Ie = moe["E"], moe["K"], moe["Ie"]
        shared_n, Is = moe["shared_n"], moe["Is"]
        lg = {"dense": moe["first_dense"], "moe": L - moe["first_dense"]}

        # routed experts (per layer): E 个专家, 每 token 激活 K 个
        if phase == "prefill":
            # 假设 batch 足够大, 所有 E 个专家的权重各被读一次
            routed_flops = 6 * K * h * Ie * tokens * lg["moe"]
            routed_wbytes = E * 3 * h * Ie * WB * lg["moe"]
        else:
            n_active = min(E, B * K)
            routed_flops = 6 * K * h * Ie * B * lg["moe"]
            routed_wbytes = n_active * 3 * h * Ie * WB * lg["moe"]

        # shared experts (DeepSeek 风格): 恒激活, 权重读一次
        shared_flops = 6 * shared_n * h * Is * tokens * lg["moe"]
        shared_wbytes = shared_n * 3 * h * Is * WB * lg["moe"]

        # 激活流量 (近似): 每 token x 读 h + 每 token gate/up 写 2I + silu-out 写 I + down-out 写 h
        act_bytes = tokens * AB * (2 * h + 3 * K * Ie + shared_n * 3 * Is) * lg["moe"]
        mlp_flops = routed_flops + shared_flops
        mlp_bytes = routed_wbytes + shared_wbytes + act_bytes

        # dense-MLP 层 (如 DeepSeek 前 3 层)
        if lg["dense"] > 0 and A["I"]:
            mlp_flops += 6 * h * A["I"] * tokens * lg["dense"]
            mlp_bytes += (
                3 * h * A["I"] * WB * lg["dense"]
                + tokens * AB * (2 * h + 3 * A["I"]) * lg["dense"]
            )
    else:
        mlp_flops = 6 * h * A["I"] * tokens * L
        mlp_bytes = 3 * h * A["I"] * WB * L + tokens * AB * (2 * h + 3 * A["I"]) * L
    st["mlp"] = _stage(mlp_flops, mlp_bytes)

    # ---------- 6. MoE 路由 (router gating) ----------
    if moe:
        E, K, Ie = moe["E"], moe["K"], moe["Ie"]
        lg = {"dense": moe["first_dense"], "moe": L - moe["first_dense"]}
        router_params = h * E
        r_flops = 2 * router_params * tokens * lg["moe"] + 5 * E * tokens * lg["moe"]
        r_bytes = router_params * WB * lg["moe"] + 2 * E * AB * tokens * lg["moe"]
        st["router"] = _stage(r_flops, r_bytes)

    # ---------- 7. LM Head ----------
    head_tokens = B
    lm_flops = 2 * h * V * head_tokens
    lm_bytes = h * V * WB + head_tokens * V * AB * 2 + head_tokens * h * AB
    st["lmhead"] = _stage(lm_flops, lm_bytes)

    # ---------- 8. 元素级 / 归一化 ----------
    if mla:
        m = mla
        rope_raw = H * (m["nope_d"] + m["rope_d"]) + (m["kv_lora"] + m["rope_d"])
    else:
        rope_raw = (H + KVH) * d

    base_flops_per_layer = 10 * h + 2 * h + 4 * rope_raw
    base_bytes_per_layer = AB * (10 * h + 2 * rope_raw)

    if moe:
        lg = {"dense": moe["first_dense"], "moe": L - moe["first_dense"]}
        E, K, Ie = moe["E"], moe["K"], moe["Ie"]
        shared_n, Is = moe["shared_n"], moe["Is"]
        # silu·mul 元素数 per token: dense 层用 I_dense, MoE 层用 K*Ie (+ shared)
        silu_bytes = tokens * AB * (
            lg["dense"] * 3 * A["I"] + lg["moe"] * 3 * (K * Ie + shared_n * Is)
        )
        silu_flops = tokens * (
            lg["dense"] * 5 * A["I"] + lg["moe"] * 5 * (K * Ie + shared_n * Is)
        )
    else:
        silu_bytes = tokens * AB * 3 * A["I"] * L
        silu_flops = tokens * 5 * A["I"] * L

    ew_flops = tokens * L * base_flops_per_layer + silu_flops
    ew_bytes = tokens * L * base_bytes_per_layer + silu_bytes
    st["elemwise"] = _stage(ew_flops, ew_bytes)

    # ---------- total ----------
    total_flops = sum(v["flops"] for v in st.values())
    total_bytes = sum(v["bytes"] for v in st.values())
    st["total"] = _stage(total_flops, total_bytes)
    return st
