# -*- coding: utf-8 -*-
"""roofline 主入口: 爬取 ModelScope config → 计算阶段强度 → 渲染 HTML 报告。

用法:
    python main.py                 # 使用缓存 (首次运行自动爬取)
    python main.py --refresh       # 忽略缓存, 强制重新爬取
    python main.py --out xxx.html  # 指定输出路径
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from roofline.arch import data_warnings, normalize
from roofline.compute import (
    DECODE_B,
    DECODE_S,
    DEVICES,
    PREFILL_S,
    STAGES,
    build_cube,
    build_stats,
)
from roofline.crawler import crawl, model_url
from roofline.report import render_report

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"

ARCH_KEYS = ["h", "L", "H", "KVH", "d", "V", "I", "tie", "moe", "mla", "hybrid", "dsa"]


def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-").lower()
    return s or "model"


def main():
    # Windows 控制台默认 cp936, 统一成 UTF-8 避免中文输出乱码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="大模型推理算力强度 (FLOP/B) 分析工具")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新爬取")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "roofline_report.html",
                    help="输出 HTML 路径 (默认 output/roofline_report.html)")
    args = ap.parse_args()

    print("== 1/3 爬取 ModelScope 模型 config ==")
    results = crawl(refresh=args.refresh)
    ok_results = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    if not ok_results:
        print("没有成功爬取任何模型，退出。")
        return 1

    print(f"== 2/3 计算各模型阶段强度 (成功 {len(ok_results)}/{len(results)}) ==")
    models = []
    for r in ok_results:
        entry = r["entry"]
        try:
            arch = normalize(r["cfg"], entry["name"], entry["family"], entry["note"], r["repo_used"])
            stats = build_stats(arch)
            cube = build_cube(arch)
            models.append({
                "id": slugify(entry["name"]),
                "name": entry["name"],
                "family": entry["family"],
                "note": entry["note"],
                "url": model_url(entry["ns"], r["repo_used"]),
                "arch": {k: arch[k] for k in ARCH_KEYS},
                "stats": stats,
                "warnings": data_warnings(arch, stats),
                "cube": cube,
            })
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {entry['name']}: {e}")

    # 摘要打印 (供 sanity check)
    for m in models:
        print(f"  {m['name']:<22} total={m['stats']['total_params']/1e9:>8.2f}B "
              f"active={m['stats']['active_params']/1e9:>8.2f}B "
              f"bytes/token={m['stats']['bytes_per_token']/1e9:>6.2f}GB")

    print("== 3/3 渲染 HTML 报告 ==")
    data = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "grid": {"prefill_S": PREFILL_S, "decode_B": DECODE_B, "decode_S": DECODE_S},
        "devices": DEVICES,
        "stages": STAGES,
        "models": models,
    }
    html = render_report(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"报告已生成: {args.out}")

    if failed:
        print(f"\n注意: {len(failed)} 个模型爬取失败:")
        for r in failed:
            print(f"  - {r['entry']['ns']}/{r['entry']['repo']}: {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
