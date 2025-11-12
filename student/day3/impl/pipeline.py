# -*- coding: utf-8 -*-
"""
Day3 파이프라인
- 기존: fetchers(NIPA/Bizinfo/Web) → normalize → rank
- 변경: PPS OpenAPI(선택) 결과도 함께 병합
  * .env USE_PPS=1 일 때 pps_fetch_bids(query) 실행
"""
from __future__ import annotations
from typing import Dict, Any, List
import os

from .fetchers import fetch_all             # NIPA/Bizinfo/Web (Tavily)
from .normalize import normalize_all
from .rank import rank_items

# 공용 스키마
from student.common.schemas import GovNotices, GovNoticeItem

# ▶ 추가: PPS OpenAPI
from student.day3.impl.pps_api import pps_fetch_bids


def _merge_and_dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """URL+제목 기준 단순 중복 제거"""
    seen, out = set(), []
    for it in items or []:
        key = (it.get("title", "").strip(), it.get("url", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def find_notices(query: str) -> dict:
    """
    1) Tavily 기반 수집(fetch_all)
    2) (옵션) PPS OpenAPI 수집(pps_fetch_bids) 추가 병합
    3) normalize → rank → GovNotices 스키마 반환
    """
    # 1) 기존 소스 수집
    raw_items = fetch_all(query)  # Day1형 스키마 리스트(title/url/snippet/...)
    
    # 2) PPS OpenAPI(선택)
    use_pps = os.getenv("USE_PPS", "1")  # 기본 1(ON)으로 두는 게 데모에 유리
    if use_pps and use_pps != "0":
        try:
            pps_items = pps_fetch_bids(query)   # 이미 GovNotice형에 가깝게 매핑됨
            # 정규화 파이프라인에 태우기 위해 Day1형처럼 최소 필드 구성
            # (normalize_all이 기대하는 최소 스키마를 맞추기 위해 변환)
            converted = []
            for it in pps_items:
                converted.append({
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "source": "pps.data.go.kr",
                    "snippet": it.get("snippet", ""),
                    "date": it.get("announce_date", ""),
                })
            raw_items.extend(converted)
        except Exception:
            pass

    # 3) normalize → rank
    norm = normalize_all(raw_items)         # Day1형 → GovNotice 표준 스키마
    norm = _merge_and_dedup(norm)           # URL+제목 중복 제거
    ranked = rank_items(norm, query)        # 점수 부여/정렬

    model = GovNotices(
        query=query,
        items=[GovNoticeItem(**it) for it in ranked]
    )
    return model.model_dump()
