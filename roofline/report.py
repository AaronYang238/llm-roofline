# -*- coding: utf-8 -*-
"""渲染 HTML 报告 (template.html + 数据)。"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent / "template.html"


def render_report(data: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return template.replace("/*__DATA__*/null", payload)
