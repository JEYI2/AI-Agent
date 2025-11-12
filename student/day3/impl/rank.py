# -*- coding: utf-8 -*-
"""
랭킹: 마감(50%) + 키워드(30%) + 출처신뢰(20%) + 규칙 보정
- close_date가 없으면 마감 점수는 0 처리
- 보정: 정부 도메인 가점 / 허브·목록 URL 강등
- 정렬: 마감 임박(오름) → 점수(내림) → 신뢰(내림)
"""
from typing import List, Dict
from datetime import date, datetime
from urllib.parse import urlparse
import re

# ── [내장] 허브/토픽/목록 URL 판정 (fetchers 의존 제거) ─────────────────────────────
_TOPIC_KEYWORDS = (
    "/tag/", "/topic/", "/hub/", "/section/", "/category/", "/tags/",
    "/검색", "/search", "/list", "/lists", "/board/list", "/news/list"
)
def _is_topic_hub(url: str) -> bool:
    u = (url or "").lower()
    return any(k in u for k in _TOPIC_KEYWORDS)

# ── 가중치/신뢰도 설정 ──────────────────────────────────────────────────────────────
WEIGHTS = {"deadline": 0.5, "keyword": 0.3, "trust": 0.2}
TRUST = {"nipa": 1.0, "bizinfo": 0.9, "web": 0.6}

_GOV_BONUS_DOMAINS = (
    "https://www.nipa.kr/home/2-2/","bizinfo.go.kr","k-startup.go.kr","g2b.go.kr",
    "ntis.go.kr","keit.re.kr","keiti.re.kr"
)

# ── 스코어러들 ────────────────────────────────────────────────────────────────────
def _days_until(dstr: str) -> int:
    if not dstr:
        return 9999
    try:
        d = datetime.strptime(dstr, "%Y-%m-%d").date()
        return (d - date.today()).days
    except Exception:
        return 9999

def _deadline_score(close_date: str) -> float:
    days = _days_until(close_date)
    if days <= 0:   # 마감 당일/초과
        return 1.0
    if days >= 30:
        return 0.0
    return max(0.0, 1.0 - (days / 30.0))

def _keyword_score(query: str, title: str, snippet: str) -> float:
    toks = re.findall(r"[가-힣A-Za-z0-9]+", (query or "").lower())
    if not toks:
        return 0.0
    t = (title or "").lower()
    s = (snippet or "").lower()
    hit = 0.0
    for tok in toks:
        if tok in t:
            hit += 2.0
        elif tok in s:
            hit += 1.0
    denom = max(1.0, 2.0 * len(toks))
    return min(1.0, hit / denom)

def _trust_score(source: str) -> float:
    return TRUST.get((source or "").lower(), 0.5)

def score_item(it: Dict, query: str) -> float:
    base = (
        WEIGHTS["deadline"] * _deadline_score(it.get("close_date","")) +
        WEIGHTS["keyword"]  * _keyword_score(query, it.get("title",""), it.get("snippet","")) +
        WEIGHTS["trust"]    * _trust_score(it.get("source",""))
    )

    # 규칙 보정: 정부 도메인 가점 / 허브·목록 강등
    url = it.get("url") or ""
    netloc = urlparse(url).netloc.lower()
    if any(netloc.endswith(d) for d in _GOV_BONUS_DOMAINS):
        base += 0.2
    if _is_topic_hub(url):
        base -= 0.5

    return max(0.0, min(1.0, base))

def rank_items(items: List[Dict], query: str) -> List[Dict]:
    scored = []
    for it in items:
        sc = score_item(it, query)
        it2 = dict(it); it2["score"] = round(sc, 4)
        scored.append(it2)

    def sort_key(x):
        # close_date 없으면 맨 뒤로 (9999일 처리)
        return (_days_until(x.get("close_date","")), -x["score"], -_trust_score(x.get("source","")))
    scored.sort(key=sort_key)
    return scored
