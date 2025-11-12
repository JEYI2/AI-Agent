# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple, Callable
import re, os
from .tavily_client import search_tavily, extract_url, extract_text

PROFILE_DOMAINS = [
    "wikipedia.org", "en.wikipedia.org", "ko.wikipedia.org",
    "finance.google.com", "google.com/finance",
    "invest.deepsearch.com", "m.invest.zum.com",
    "companiesmarketcap.com", "marketscreener.com",
    "alphasquare.co.kr",
]

def looks_like_ticker(q: str) -> bool:
    return bool(re.search(r"\b([A-Z]{1,5}(?:\.[A-Z]{2,4})?|\d{6}(?:\.[A-Z]{2,4})?)\b", q))

def search_company_profile(query: str, api_key: str, topk: int = 6, timeout: int = 20) -> List[Dict[str, Any]]:
    q = f"{query} company profile overview 기업 개요 회사 소개 무엇을 하는 회사"
    # ⬇ 원문 발췌를 렌더에서 쓰고 싶다면 include_raw_content=True를 켜도 좋음
    results = search_tavily(q, api_key, top_k=topk, timeout=timeout, include_raw_content=True)
    def score(r: Dict[str, Any]) -> Tuple[int, float]:
        dom = (r.get("source") or r.get("url") or "").lower()
        prio = 0
        for i, d in enumerate(PROFILE_DOMAINS):
            if d in dom:
                prio = 100 - i
                break
        return (-prio, -float(r.get("score", 0.0)))
    return sorted(results, key=score)

def extract_and_summarize_profile(
    urls: List[str],
    api_key: str,
    summarizer: Callable[[str], str],
    max_chars: int = 6000
) -> str:
    texts: List[str] = []
    for u in urls[:2]:
        try:
            clean = extract_url(u)  # ← URL 정리(인자 1개)
            t = extract_text(clean, api_key)[:max_chars]  # ← 본문 추출
            if len(t) > 500:  # 최소 분량 보장
                texts.append(f"[{clean}]\n{t}")
        except Exception:
            continue
    if not texts:
        return ""
    joined = "\n\n---\n\n".join(texts)
    prompt = (
        "다음 자료를 근거로 '기업 개요'를 한국어 5~7줄로 요약하세요.\n"
        "- 핵심 사업/제품, 수익원, 주요 시장/고객, 차별점, 최근 이슈(있으면)\n"
        "- 과도한 재무 디테일은 피하고, 문장당 20~30자 이내로 간결하게.\n\n"
        f"{joined}\n"
    )
    return summarizer(prompt)
