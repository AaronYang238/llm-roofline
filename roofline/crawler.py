# -*- coding: utf-8 -*-
"""ModelScope 大模型 config.json 爬取器。

数据源: https://modelscope.cn
  - 模型详情: GET {API_ROOT}/models/{namespace}/{name}
  - 仓库文件: GET {API_ROOT}/models/{namespace}/{name}/repo?FilePath=config.json&Revision=master

特性:
  * 失败自动重试 + 本地缓存 (configs_cache/)，二次运行秒级完成
  * 每个条目可配置候选仓库 (alts)，任一命中即成功
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://modelscope.cn/api/v1"
SITE_ROOT = "https://modelscope.cn/models"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

CACHE_DIR = Path(__file__).resolve().parent.parent / "configs_cache"

# 每条: {"ns", "repo", "alts"(可选, 备选仓库), "name", "family", "note"}
MODEL_LIST: list[dict] = [
    # ---------------- Qwen2.5 (dense · GQA) ----------------
    {"ns": "Qwen", "repo": "Qwen2.5-0.5B",        "name": "Qwen2.5-0.5B",  "family": "Qwen2.5", "note": "Dense · GQA · tied"},
    {"ns": "Qwen", "repo": "Qwen2.5-1.5B",        "name": "Qwen2.5-1.5B",  "family": "Qwen2.5", "note": "Dense · GQA · tied"},
    {"ns": "Qwen", "repo": "Qwen2.5-3B",          "name": "Qwen2.5-3B",    "family": "Qwen2.5", "note": "Dense · GQA · tied"},
    {"ns": "Qwen", "repo": "Qwen2.5-7B",          "name": "Qwen2.5-7B",    "family": "Qwen2.5", "note": "Dense · GQA"},
    {"ns": "Qwen", "repo": "Qwen2.5-14B",         "name": "Qwen2.5-14B",   "family": "Qwen2.5", "note": "Dense · GQA"},
    {"ns": "Qwen", "repo": "Qwen2.5-32B",         "name": "Qwen2.5-32B",   "family": "Qwen2.5", "note": "Dense · GQA"},
    {"ns": "Qwen", "repo": "Qwen2.5-72B",         "name": "Qwen2.5-72B",   "family": "Qwen2.5", "note": "Dense · GQA"},
    # ---------------- Qwen3 (dense · GQA) ----------------
    {"ns": "Qwen", "repo": "Qwen3-0.6B",   "name": "Qwen3-0.6B", "family": "Qwen3", "note": "Dense · GQA · QK-norm · tied"},
    {"ns": "Qwen", "repo": "Qwen3-1.7B",   "name": "Qwen3-1.7B", "family": "Qwen3", "note": "Dense · GQA · QK-norm · tied"},
    {"ns": "Qwen", "repo": "Qwen3-4B",     "name": "Qwen3-4B",   "family": "Qwen3", "note": "Dense · GQA · QK-norm · tied"},
    {"ns": "Qwen", "repo": "Qwen3-8B",     "name": "Qwen3-8B",   "family": "Qwen3", "note": "Dense · GQA · QK-norm · tied"},
    {"ns": "Qwen", "repo": "Qwen3-14B",    "name": "Qwen3-14B",  "family": "Qwen3", "note": "Dense · GQA · QK-norm · tied"},
    {"ns": "Qwen", "repo": "Qwen3-32B",    "name": "Qwen3-32B",  "family": "Qwen3", "note": "Dense · GQA · QK-norm"},
    {"ns": "Qwen", "repo": "QwQ-32B",      "name": "QwQ-32B",    "family": "Qwen2.5", "note": "Dense · GQA · 推理模型"},
    # ---------------- Qwen3 MoE ----------------
    {"ns": "Qwen", "repo": "Qwen3-30B-A3B",   "name": "Qwen3-30B-A3B",   "family": "Qwen3 MoE", "note": "MoE · GQA · tied"},
    {"ns": "Qwen", "repo": "Qwen3-235B-A22B", "name": "Qwen3-235B-A22B", "family": "Qwen3 MoE", "note": "MoE · GQA"},
    # ---------------- DeepSeek (MoE · MLA) ----------------
    {"ns": "deepseek-ai", "repo": "DeepSeek-V3", "name": "DeepSeek-V3", "family": "DeepSeek", "note": "MoE · MLA"},
    {"ns": "deepseek-ai", "repo": "DeepSeek-R1", "name": "DeepSeek-R1", "family": "DeepSeek", "note": "MoE · MLA · 推理模型"},
    # ---------------- Llama 3.1 (dense · GQA) ----------------
    {"ns": "LLM-Research", "repo": "Meta-Llama-3.1-8B-Instruct",   "name": "Llama-3.1-8B",   "family": "Llama 3.1", "note": "Dense · GQA"},
    {"ns": "LLM-Research", "repo": "Meta-Llama-3.1-70B-Instruct",  "name": "Llama-3.1-70B",  "family": "Llama 3.1", "note": "Dense · GQA"},
    {"ns": "LLM-Research", "repo": "Meta-Llama-3.1-405B-Instruct", "name": "Llama-3.1-405B", "family": "Llama 3.1", "note": "Dense · GQA"},
    # ---------------- Mistral / Mixtral ----------------
    {"ns": "AI-ModelScope", "repo": "Mistral-7B-Instruct-v0.2",    "name": "Mistral-7B",    "family": "Mistral", "note": "Dense · GQA · 滑动窗口"},
    {"ns": "AI-ModelScope", "repo": "Mixtral-8x7B-Instruct-v0.1",  "name": "Mixtral-8x7B",  "family": "Mixtral", "note": "MoE · GQA"},
    {"ns": "AI-ModelScope", "repo": "Mixtral-8x22B-Instruct-v0.1", "name": "Mixtral-8x22B", "family": "Mixtral", "note": "MoE · GQA"},
    # ---------------- Google Gemma 2 ----------------
    {"ns": "AI-ModelScope", "repo": "gemma-2-9b-it",  "name": "Gemma-2-9B",  "family": "Gemma 2", "note": "Dense · GQA · 滑动窗口 · tied"},
    {"ns": "AI-ModelScope", "repo": "gemma-2-27b-it", "name": "Gemma-2-27B", "family": "Gemma 2", "note": "Dense · GQA · 滑动窗口 · tied"},
    # ---------------- Microsoft Phi-3 ----------------
    {"ns": "microsoft", "repo": "Phi-3-medium-4k-instruct", "name": "Phi-3-medium", "family": "Phi 3", "note": "Dense · GQA"},
    # ---------------- Zhipu GLM ----------------
    {"ns": "ZhipuAI", "repo": "glm-4-9b-chat", "name": "GLM-4-9B", "family": "GLM-4", "note": "Dense · GQA"},
    {"ns": "ZhipuAI", "repo": "GLM-4.5",       "name": "GLM-4.5",  "family": "GLM-4.5", "note": "MoE · MLA"},
    # ---------------- 01.AI Yi ----------------
    {"ns": "01ai", "repo": "Yi-34B", "name": "Yi-34B", "family": "Yi", "note": "Dense · MHA"},
    # ---------------- Baichuan 2 ----------------
    {"ns": "baichuan-inc", "repo": "Baichuan2-7B-Chat", "name": "Baichuan2-7B", "family": "Baichuan 2", "note": "Dense · MHA"},
    # ---------------- InternLM 2.5 ----------------
    {"ns": "Shanghai_AI_Laboratory", "repo": "internlm2_5-7b", "name": "InternLM2.5-7B", "family": "InternLM 2.5", "note": "Dense · GQA"},
    # ---------------- Moonshot Kimi K2 ----------------
    {"ns": "moonshotai", "repo": "Kimi-K2-Instruct", "name": "Kimi-K2", "family": "Kimi K2", "note": "MoE · MLA"},
    # ---------------- Qwen3.5 / 3.8 (hybrid GDN + full, GQA) ----------------
    {"ns": "Qwen", "repo": "Qwen3.5-0.8B",  "name": "Qwen3.5-0.8B",  "family": "Qwen3.5", "note": "Dense · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.5-4B",    "name": "Qwen3.5-4B",    "family": "Qwen3.5", "note": "Dense · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.5-9B",    "name": "Qwen3.5-9B",    "family": "Qwen3.5", "note": "Dense · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.5-35B-A3B", "name": "Qwen3.5-35B-A3B", "family": "Qwen3.5", "note": "MoE · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.6-35B-A3B", "name": "Qwen3.6-35B-A3B", "family": "Qwen3.5", "note": "MoE · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.8-27B",   "name": "Qwen3.8-27B",   "family": "Qwen3.5", "note": "Dense · 混合线性注意力"},
    {"ns": "Qwen", "repo": "Qwen3.8-Flash-Next", "name": "Qwen3.8-Flash-Next", "family": "Qwen3.5", "note": "MoE · 混合线性注意力"},
    # ---------------- DeepSeek V4 (MoE · DSA) ----------------
    {"ns": "deepseek-ai", "repo": "DeepSeek-V4.1-Flash", "name": "DeepSeek-V4.1-Flash", "family": "DeepSeek", "note": "MoE · DSA 稀疏注意力"},
    {"ns": "deepseek-ai", "repo": "DeepSeek-V4-Pro",     "name": "DeepSeek-V4-Pro",     "family": "DeepSeek", "note": "MoE · DSA 稀疏注意力"},
    # ---------------- Zhipu GLM-5 ----------------
    {"ns": "ZhipuAI", "repo": "GLM-5.3",       "name": "GLM-5.3",       "family": "GLM-5", "note": "MoE · MLA · DSA 稀疏注意力"},
    {"ns": "ZhipuAI", "repo": "GLM-5.3-Flash", "name": "GLM-5.3-Flash", "family": "GLM-5", "note": "MoE · MLA · 混合线性注意力"},
    # ---------------- Moonshot Kimi K3 ----------------
    {"ns": "moonshotai", "repo": "Kimi-K3", "name": "Kimi-K3", "family": "Kimi K3", "note": "MoE · MLA · 混合线性注意力"},
    # ---------------- inclusionAI Ling / Ring ----------------
    {"ns": "inclusionAI", "repo": "Ling-3.1-flash", "name": "Ling-3.1-flash", "family": "Ling 3.1", "note": "MoE · MLA"},
    {"ns": "inclusionAI", "repo": "Ring-2.6-1T",    "name": "Ring-2.6-1T",    "family": "Ling 3.1", "note": "MoE · MLA"},
    # ---------------- Tencent Hunyuan ----------------
    {"ns": "Tencent-Hunyuan", "repo": "Hy3",                "name": "Hy3",                "family": "Hunyuan", "note": "MoE"},
    {"ns": "Tencent-Hunyuan", "repo": "Hy-Memory-30B-A3B",  "name": "Hy-Memory-30B-A3B",  "family": "Hunyuan", "note": "MoE"},
    # ---------------- Hcompany Holo3 ----------------
    {"ns": "Hcompany", "repo": "Holo3-35B-A3B", "name": "Holo3-35B-A3B", "family": "Holo3", "note": "MoE · 混合线性注意力"},
    # ---------------- openPangu ----------------
    {"ns": "openpangu", "repo": "openPangu-2.0-Pro", "name": "openPangu-2.0-Pro", "family": "openPangu", "note": "MoE · MLA"},
    # ---------------- upstage Solar ----------------
    {"ns": "upstage", "repo": "Solar-Open2-250B", "name": "Solar-Open2-250B", "family": "Solar", "note": "MoE"},
]


def http_get(url: str, timeout: float = 30.0, retries: int = 3) -> bytes:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 + i)
    assert last_err is not None
    raise last_err


def model_url(ns: str, repo: str) -> str:
    return f"{SITE_ROOT}/{ns}/{repo}"


def crawl(refresh: bool = False, verbose: bool = True) -> list[dict]:
    """爬取 MODEL_LIST 中所有模型的 config.json。

    返回条目列表:
      {"entry": {...}, "ok": bool, "cfg": dict|None, "repo_used": str, "error": str|None}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for entry in MODEL_LIST:
        repos = [entry["repo"]] + list(entry.get("alts") or [])
        cfg = None
        used = None
        err = None
        for repo in repos:
            cache_path = CACHE_DIR / f"{entry['ns']}__{repo}.json"
            if cache_path.exists() and not refresh:
                try:
                    cfg = json.loads(cache_path.read_text(encoding="utf-8"))
                    used = repo
                    break
                except Exception:
                    cache_path.unlink(missing_ok=True)
            try:
                url = (
                    f"{API_ROOT}/models/{entry['ns']}/{repo}"
                    "/repo?FilePath=config.json&Revision=master"
                )
                raw = http_get(url)
                cfg = json.loads(raw)
                cache_path.write_text(
                    json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                used = repo
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)
                continue

        ok = cfg is not None
        if verbose:
            tag = "ok " if ok else "FAIL"
            suffix = f" <- {used}" if (ok and used != entry["repo"]) else ""
            extra = f" ({err[:80]})" if not ok else ""
            print(f"  [{tag}] {entry['ns']}/{entry['repo']}{suffix}{extra}")
        results.append(
            {
                "entry": entry,
                "ok": ok,
                "cfg": cfg,
                "repo_used": used,
                "error": None if ok else (err or "unknown error"),
            }
        )
        time.sleep(0.2)

    return results
