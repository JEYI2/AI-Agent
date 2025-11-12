# -*- coding: utf-8 -*-
import re, time
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def _slugify(text: str) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^0-9A-Za-z가-힣\-_]+", "", text)
    return text[:80] or "output"

def save_markdown(query: str, route: str, markdown: str) -> str:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    slug = _slugify(query)
    route = _slugify(route or "auto")
    f = PROCESSED_DIR / f"{ts}__{route}__{slug}.md"
    f.write_text(markdown, encoding="utf-8")
    return str(f)
