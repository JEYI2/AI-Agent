# -*- coding: utf-8 -*-
"""
Day1 본체
- 역할: 웹 검색 / 주가 / 기업개요(추출+요약)를 병렬로 수행하고 결과를 정규 스키마로 병합
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.adk.models.lite_llm import LiteLlm
from student.common.schemas import Day1Plan
from student.day1.impl.merge import merge_day1_payload
# 외부 I/O
from student.day1.impl.tavily_client import search_tavily, extract_url
from student.day1.impl.finance_client import get_quotes
from student.day1.impl.web_search import (
    looks_like_ticker,
    search_company_profile,
    extract_and_summarize_profile,
)

DEFAULT_WEB_TOPK = 6
MAX_WORKERS = 4
DEFAULT_TIMEOUT = 20


_SUM: Optional[LiteLlm] = LiteLlm(model="openai/gpt-4o-mini")

def _summarize(text: str) -> str:
    """
    입력 텍스트를 LLM으로 3~5문장 수준으로 요약합니다.
    실패 시 빈 문자열("")을 반환해 상위 로직이 안전하게 진행되도록 합니다.
    """
    if not _SUM:
        return ""
    try:
        # LiteLlm의 invoke 메서드는 보통 텍스트를 직접 받거나,
        # 혹은 특정 형식의 입력을 요구할 수 있습니다. (예: {"prompt": text})
        # 여기서는 간단히 텍스트를 직접 전달하는 것으로 가정합니다.
        # 실제 ADK의 LiteLlm 사용법에 따라 조정이 필요할 수 있습니다.
        response = _SUM.invoke(f"다음 텍스트를 3-5문장으로 요약해줘:\n\n{text}")
        
        # 응답 객체 구조에 따라 실제 텍스트를 추출해야 합니다.
        # 예: response.text, response['content'], 등
        # 여기서는 응답이 바로 문자열이라고 가정합니다.
        if isinstance(response, str):
            return response
        # ADK의 LlmResponse를 반환하는 경우
        elif hasattr(response, 'content'):
            return response.content.parts[0].text
        else:
            return str(response) # 최후의 수단

    except Exception:
        # 로깅을 추가하면 좋지만, 여기서는 실패 시 빈 문자열 반환 원칙을 따릅니다.
        return ""


class Day1Agent:
    def __init__(self, tavily_api_key: Optional[str], web_topk: int = DEFAULT_WEB_TOPK, request_timeout: int = DEFAULT_TIMEOUT):
        """
        필드 저장만 담당합니다.
        - tavily_api_key: Tavily API 키(없으면 웹 호출 실패 가능)
        - web_topk: 기본 검색 결과 수
        - request_timeout: 각 HTTP 호출 타임아웃(초)
        """
        self.tavily_api_key = tavily_api_key
        self.web_topk = web_topk
        self.request_timeout = request_timeout

    def handle(self, query: str, plan: Day1Plan) -> Dict[str, Any]:
        """
        병렬 파이프라인:
          1) results 스켈레톤 만들기
             results = {"type":"web_results","query":query,"analysis":asdict(plan),"items":[],
                        "tickers":[], "errors":[], "company_profile":"", "profile_sources":[]}
          2) ThreadPoolExecutor(max_workers=MAX_WORKERS)에서 작업 제출:
             - plan.do_web: search_tavily(검색어, 키, top_k=self.web_topk, timeout=...)
             - plan.do_stocks: get_quotes(plan.tickers)
             - (기업개요) looks_like_ticker(query) 또는 plan에 tickers가 있을 때:
                 · search_company_profile(query, api_key, topk=2) → URL 상위 1~2개
                 · extract_and_summarize_profile(urls, api_key, summarizer=_summarize)
          3) as_completed로 결과 수집. 실패 시 results["errors"]에 '작업명:에러' 저장.
          4) merge_day1_payload(results) 호출해 최종 표준 스키마 dict 반환.
        """
        results: Dict[str, Any] = {
            "type": "web_results",
            "query": query,
            "analysis": asdict(plan),
            "items": [],
            "tickers": [],
            "errors": [],
            "company_profile": "",
            "profile_sources": []
        }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures: Dict[Any, str] = {}

            # 1. 웹 검색 작업
            if plan.do_web and plan.web_keywords:
                future = executor.submit(
                    search_tavily, 
                    query=plan.web_keywords[0], 
                    api_key=self.tavily_api_key, 
                    top_k=self.web_topk, 
                    timeout=self.request_timeout
                )
                futures[future] = "web"

            # 2. 주가 조회 작업
            if plan.do_stocks and plan.tickers:
                future = executor.submit(get_quotes, plan.tickers)
                futures[future] = "stock"

            # 3. 기업 개요 검색 및 요약 작업
            # 쿼리가 티커처럼 보이거나, 이미 티커가 지정된 경우
            should_run_profile = looks_like_ticker(query) or (plan.tickers and len(plan.tickers) > 0)
            if should_run_profile:
                future = executor.submit(self._get_company_profile, query, plan.tickers)
                futures[future] = "profile"

            for future in as_completed(futures):
                kind = futures[future]
                try:
                    data = future.result()
                    if kind == "web":
                        results["items"] = data
                    elif kind == "stock":
                        results["tickers"] = data
                    elif kind == "profile":
                        if data:
                            profile_text, profile_urls = data
                            results["company_profile"] = profile_text
                            results["profile_sources"] = profile_urls
                except Exception as e:
                    results["errors"].append(f"{kind}_pipeline: {type(e).__name__}: {e}")

        return merge_day1_payload(results)

    def _get_company_profile(self, query: str, tickers: List[str]) -> Optional[Tuple[str, List[str]]]:
        """기업 개요 파이프라인 헬퍼"""
        # 검색어는 티커 또는 사용자 원본 쿼리 사용
        search_query = tickers[0] if tickers else query
        
        urls = search_company_profile(search_query, self.tavily_api_key, topk=2)
        if not urls:
            return None
        
        profile_text, profile_urls = extract_and_summarize_profile(
            urls=urls,
            api_key=self.tavily_api_key,
            summarizer=_summarize
        )
        return profile_text, profile_urls
