# -*- coding: utf-8 -*-
"""渲染 HTML 报告 (template.html + 数据)。"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


def render_report(data: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    marker = "/*__DATA__*/null"
    if marker not in template:
        raise ValueError("template.html 缺少 /*__DATA__*/null 数据占位符")
    # 转义 </ 防止字符串中出现 </script> 提前闭合脚本标签
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(marker, payload)
