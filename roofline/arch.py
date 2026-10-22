# -*- coding: utf-8 -*-
"""将各家族 config.json 归一化为统一的架构描述 (arch record)。

归一化后的字段:
  h, L, H, KVH, d, V, tie, I
MoE 模型附加:
  moe = {"E", "K", "Ie", "shared_n", "Is", "first_dense"}
MLA 模型附加:
  mla = {"q_lora", "kv_lora", "rope_d", "nope_d", "v_d"}
混合注意力模型 (Qwen3.5/3.8, Kimi-K3, GLM-5.3-Flash) 附加:
  hybrid = {"L_full", "state_elems"}
DSA 稀疏注意力模型 (DeepSeek-V4, GLM-5.3) 附加:
  dsa = {"topk"}
"""
from __future__ import annotations

import json
import re


def _first(d, keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def normalize(
    cfg: dict,
    name: str,
    family: str,
    note: str = "",
    repo_used: str | None = None,
) -> dict:
    cfg = _unwrap(cfg)
    h = cfg["hidden_size"]
    L = _first(cfg, ["num_hidden_layers", "n_layers", "num_layers"])
    H = _first(cfg, ["num_attention_heads", "n_heads"])
    KVH = _first(cfg, ["num_key_value_heads", "num_kv_heads"])
    if KVH is None and cfg.get("multi_query_attention"):
        KVH = cfg.get("multi_query_group_num")
    if KVH is None:
        KVH = H
    V = _first(cfg, ["vocab_size", "padded_vocab_size"])

    # tie_word_embeddings 在部分 config (如 Gemma 2) 中缺省;
    # 此时回退到模型清单里的人工标注 ("tied"), 避免重复计算 LM head。
    tie_raw = cfg.get("tie_word_embeddings")
    tie_inferred = tie_raw is None
    tie = bool(tie_raw) if tie_raw is not None else ("tied" in note.lower())

    d = _first(cfg, ["head_dim", "kv_channels"])
    d_inferred = d is None
    if d is None:
        d = (h / H) if H else None

    # ---------- MLA (多级潜注意力) ----------
    mla = None
    if cfg.get("kv_lora_rank"):
        mla = {
            "q_lora": _first(cfg, ["q_lora_rank"], 1536),
            "kv_lora": cfg["kv_lora_rank"],
            "rope_d": _first(cfg, ["qk_rope_head_dim"], 64),
            "nope_d": _first(cfg, ["qk_nope_head_dim"], 128),
            "v_d": _first(cfg, ["v_head_dim"], 128),
        }

    # ---------- MoE ----------
    moe = None
    k_inferred = False
    # dense MLP 维度: 常规字段 -> ChatGLM 系 ffn_hidden_size
    I = _first(cfg, ["intermediate_size", "ffn_hidden_size"])
    E = _first(cfg, ["num_experts", "num_local_experts", "n_routed_experts"])
    if E:
        # 注意: 不能用通用的 top_k —— 很多 config 的 top_k 是采样参数
        # (例如 Kimi-K3: top_k=50, num_experts_per_token=16), 会高估激活专家数。
        K = _first(
            cfg,
            [
                "num_experts_per_tok",
                "num_experts_per_token",
                "experts_per_tok",
                "num_selected_experts",
                "n_experts_per_tok",
            ],
            None,
        )
        if K is None:
            K = 2
            k_inferred = True
        K = int(K)

        # 专家 FFN 维度: 不同家族字段不同。优先使用显式的 expert 维度字段,
        # 否则回退到 moe_intermediate_size。
        Ie = _first(
            cfg,
            [
                "expert_hidden_dim",
                "routed_expert_hidden_size",
                "moe_intermediate_size",
                "intermediate_size",
            ],
        )
        if Ie is None:
            Ie = I if I else h * 4

        # 共享专家: Qwen3.5/Holo3 只给 shared_expert_intermediate_size,
        # 未给 num_shared_experts, 此时按 1 个共享专家计。
        shared_size = _first(
            cfg, ["shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"]
        )
        shared_n = _first(cfg, ["n_shared_experts", "num_shared_experts"], 0)
        if shared_size is not None and not shared_n:
            shared_n = 1
        Is = shared_size if shared_size is not None else Ie

        moe = {
            "E": E,
            "K": K,
            "Ie": Ie,
            "shared_n": shared_n,
            "Is": Is,
            "first_dense": min(int(_first(cfg, ["first_k_dense_replace"], 0) or 0), L),
            "k_inferred": k_inferred,
        }

    # ---------- 混合线性注意力 (GDN / KDA) ----------
    # 部分层是线性注意力 (固定大小循环状态), 只有注意力层累积 KV 缓存
    hybrid = None
    layer_types = cfg.get("layer_types")
    lac = cfg.get("linear_attn_config")
    if isinstance(layer_types, list) and layer_types:
        # Qwen3.5/3.6/3.8, Holo3: layer_types + linear_* 字段 (GDN)
        full = sum(1 for t in layer_types if t != "linear_attention")
        if full < len(layer_types):  # 存在线性注意力层
            nvkh = _first(cfg, ["linear_num_value_heads"], 32)
            kdim = _first(cfg, ["linear_key_head_dim"], 128)
            vdim = _first(cfg, ["linear_value_head_dim"], 128)
            hybrid = {"L_full": min(full, L), "state_elems": nvkh * kdim * vdim}
    elif isinstance(lac, dict) and lac:
        # Kimi-K3 / GLM-5.3-Flash: linear_attn_config 带 kda_layers (KDA)
        full_layers = lac.get("full_attn_layers")
        if full_layers is not None:
            state = lac.get("num_heads", 64) * (lac.get("head_dim", 128) ** 2)
            hybrid = {"L_full": min(len(full_layers), L), "state_elems": state}

    # ---------- DSA 稀疏注意力 ----------
    # indexer 每步只选 topk 个 token 的 KV 参与注意力
    dsa = None
    if cfg.get("index_topk"):
        dsa = {"topk": cfg["index_topk"]}

    # dense MLP 维度: 常规字段 -> ChatGLM 系 ffn_hidden_size
    I = _first(cfg, ["intermediate_size", "ffn_hidden_size"])
    if I is None and moe:
        I = moe["Ie"]

    # 记录归一化过程中的推断, 供报告做数据质量提示
    meta = {
        "tie_inferred": tie_inferred,
        "d_inferred": d_inferred,
        "k_inferred": k_inferred,
        "mla_partial": bool(
            cfg.get("q_lora_rank")
            and not cfg.get("kv_lora_rank")
            and cfg.get("num_key_value_heads", 1) == 1
        ),
    }

    return {
        "name": name,
        "family": family,
        "note": note,
        "repo": repo_used,
        "h": h,
        "L": L,
        "H": H,
        "KVH": KVH,
        "d": d,
        "V": V,
        "tie": tie,
        "I": I,
        "moe": moe,
        "mla": mla,
        "hybrid": hybrid,
        "dsa": dsa,
        "meta": meta,
    }


def _unwrap(cfg: dict) -> dict:
    """多模态 config 把语言模型参数嵌套在 text_config, 统一解包。"""
    tc = cfg.get("text_config")
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except Exception:
            tc = None
    if tc and isinstance(tc, dict):
        merged = dict(cfg)
        merged.update(tc)
        return merged
    return cfg


def layer_groups(A: dict) -> dict:
    """按 MLP 类型分桶: dense-MLP 层 (如 DeepSeek 前 3 层) 与 MoE 层。"""
    L = A["L"]
    if A["moe"]:
        fd = A["moe"]["first_dense"]
        return {"dense": fd, "moe": L - fd}
    return {"dense": L, "moe": 0}


def param_counts(A: dict) -> dict:
    """计算参数量 (与模型卡公布的 B 数对齐, 用于 sanity check)。"""
    h, L, H, KVH, V = A["h"], A["L"], A["H"], A["KVH"], A["V"]
    moe = A["moe"]
    mla = A["mla"]
    lg = layer_groups(A)

    # ---- attention projections ----
    if mla:
        m = mla
        qkv_params = (
            h * m["q_lora"]
            + m["q_lora"] * H * (m["nope_d"] + m["rope_d"])
            + h * (m["kv_lora"] + m["rope_d"])
            + m["kv_lora"] * H * (m["nope_d"] + m["v_d"])
        )
        o_params = H * m["v_d"] * h
    else:
        qkv_params = h * (H + 2 * KVH) * A["d"]
        o_params = h * H * A["d"]

    # ---- MLP: dense layers ----
    dense_mlp_params = 3 * h * A["I"] if A["I"] else 0
    dense_total = lg["dense"] * (qkv_params + o_params + dense_mlp_params)
    dense_active = dense_total

    # ---- MLP: MoE layers ----
    if moe:
        routed_params = moe["E"] * 3 * h * moe["Ie"]
        shared_params = moe["shared_n"] * 3 * h * moe["Is"]
        router_params = h * moe["E"]
        moe_total = lg["moe"] * (
            qkv_params + o_params + routed_params + shared_params + router_params
        )
        moe_active = lg["moe"] * (
            qkv_params
            + o_params
            + moe["K"] * 3 * h * moe["Ie"]
            + shared_params
            + router_params
        )
        total = dense_total + moe_total
        active = dense_active + moe_active
    else:
        total = L * (qkv_params + o_params + dense_mlp_params)
        active = total

    # ---- embedding & LM head ----
    # LM head 在每个 decode step 都会被读取, 因此 active 参数必须包含它;
    # tied embedding 时 head 与 embedding 共享, 不重复计。
    embed_params = h * V
    head_params = h * V
    total += embed_params
    active += embed_params
    if not A["tie"]:
        total += head_params
        active += head_params

    return {
        "total": total,
        "active": active,
        "qkv_params": qkv_params,
        "o_params": o_params,
        "dense_mlp_params": dense_mlp_params,
        "router_params": (h * moe["E"]) if moe else 0,
    }


def data_warnings(A: dict, stats: dict) -> list[str]:
    """对归一化结果做轻量级可信度校验, 在报告中显式提示异常而不是静默展示。"""
    warnings: list[str] = []
    name = A.get("name") or ""
    actual = stats.get("total_params", 0) / 1e9

    # 模型名里的参数量 (如 30B / 0.5B) 与解析值偏差 >35% 时提示
    m = re.search(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*B(?![A-Za-z])", name)
    if m and actual > 0:
        claimed = float(m.group(1))
        if claimed and (actual > claimed * 1.35 or actual < claimed * 0.70):
            warnings.append(
                f"解析参数量 {actual:.1f}B 与模型名标注的 {claimed:g}B 偏差较大"
                "（可能为近似架构或 config 字段缺失）"
            )

    moe = A.get("moe")
    if moe:
        if moe["K"] > moe["E"]:
            warnings.append(f"每 token 专家数 K={moe['K']} 大于专家总数 E={moe['E']}")
        if moe.get("k_inferred"):
            warnings.append("config 未声明每 token 激活专家数，按 K=2 估算")

    meta = A.get("meta") or {}
    if meta.get("mla_partial"):
        warnings.append(
            "检测到 q_lora/o_lora 但缺少 kv_lora_rank，注意力按 GQA 近似建模，MLA 相关数值可能偏差"
        )

    if not A.get("V"):
        warnings.append("缺少 vocab_size，LM Head 与 embedding 未计入")
    if not A.get("L") or not A.get("H"):
        warnings.append("缺少层数或注意力头数，架构参数不完整")

    return warnings
