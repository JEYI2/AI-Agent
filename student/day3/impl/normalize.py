# -*- coding: utf-8 -*-
"""
raw → GovNotice 표준 스키마 정규화 (강사용/답지)
- fetchers.py에서 온 Day1형 raw 결과를 GovNotice 필드로 매핑
- URL 중복 제거
"""
from typing import List, Dict
from datetime import datetime

DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%dT%H:%M:%S%z")


def _as_date_iso(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    # 숫자 8자리(YYYYMMDD) 대응
    if s.isdigit() and len(s) == 8:
        try:
            return datetime.strptime(s, "%Y%m%d").date().isoformat()
        except Exception:
            pass
    return ""


def normalize_all(raw_items: List[Dict]) -> List[Dict]:
    norm: List[Dict] = []
    for r in raw_items or []:
        # Day1 웹결과 스키마: title/url/source/snippet/date
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        source = (r.get("source") or "").strip().lower()
        snippet = (r.get("snippet") or "").strip()
        date_guess = _as_date_iso(r.get("date") or "")

        norm.append({
            "title": title,
            "url": url,
            "source": "nipa" if "nipa" in source else ("bizinfo" if "bizinfo" in source else "web"),
            "agency": "",
            "announce_date": date_guess,   # 알 수 없으면 빈 값
            "close_date": "",              # 랭커에서 없을 경우 패널티
            "budget": "",
            "snippet": snippet,
            "attachments": [],
            "content_type": "notice",
            "score": 0.0,
        })

    # URL 기준 중복 제거
    seen = set()
    deduped = []
    for n in norm:
        u = n["url"]
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(n)
    return deduped
