# -*- coding: utf-8 -*-
"""
Day1: 웹+주가+기업개요 에이전트
- 역할: 사용자 질의를 받아 Day1 본체 호출 → 결과 렌더 → 파일 저장(envelope) → 응답
- 본 파일은 "UI용 래퍼"로, 실질적인 수집/요약 로직은 impl/agent.py 등에 있음.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import re

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from student.common.schemas import Day1Plan
from student.common.writer import render_day1, render_enveloped
from student.common.fs_utils import save_markdown
from student.day1.impl.agent import Day1Agent



MODEL = LiteLlm(model="openai/gpt-4o")

def _extract_tickers_from_query(query: str) -> List[str]:
    """
    사용자 질의에서 '티커 후보'를 추출합니다.
    예시:
      - "AAPL 주가 알려줘"      → ["AAPL"]
      - "삼성전자 005930 분석"  → ["005930"]
      - "NVDA/TSLA 비교"       → ["NVDA", "TSLA"]
    구현 포인트:
      1) 두 타입 모두 잡아야 함
         - 영문 대문자 1~5자 (미국 티커 일반형) + 선택적 .XX (예: BRK.B 처럼 도메인 일부가 있을 수 있으나, 여기선 단순히 대문자 1~5자를 1차 타깃)
         - 숫자 6자리 (국내 종목코드)
      2) 중복 제거(순서 유지)
      3) 불필요한 특수문자 제거 후 패턴 매칭
    """
    # 영문 티커(1~5자) 및 숫자 티커(6자리) 추출
    cleaned_query = re.sub(r'[^A-Z0-9\s]', '', query.upper())
    
    # 영문 티커 추출 (예: AAPL, NVDA)
    eng_tickers = re.findall(r'\b[A-Z]{1,5}\b', cleaned_query)
    
    # 숫자 티커 추출 (예: 005930)
    num_tickers = re.findall(r'\b\d{6}\b', cleaned_query)
    
    # 순서 유지하며 중복 제거
    all_tickers = eng_tickers + num_tickers
    seen = set()
    unique_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]
    
    return unique_tickers


def _normalize_kr_tickers(tickers: List[str]) -> List[str]:
    """
    한국식 6자리 종목코드에 '.KS'를 붙여 yfinance 호환 심볼로 보정합니다.
    예:
      ['005930', 'AAPL'] → ['005930.KS', 'AAPL']
    구현 포인트:
      1) 각 원소가 6자리 숫자면 뒤에 '.KS'를 붙임
      2) 이미 확장자가 붙은 경우(예: '.KS')는 그대로 둠
    """
    normalized = []
    for ticker in tickers:
        if re.fullmatch(r"\d{6}", ticker):
            normalized.append(f"{ticker}.KS")
        else:
            normalized.append(ticker)
    return normalized


def _handle(query: str) -> Dict[str, Any]:
    """
    Day1 전체 흐름(오케스트레이션):
      1) 키 준비: os.getenv("TAVILY_API_KEY", "")
      2) 티커 추출 → 한국형 보정
      3) Day1Plan 구성
         - do_web=True (웹 검색은 기본 수행)
         - do_stocks=True/False (티커가 존재하면 True)
         - web_keywords: [query] (필요시 키워드 가공 가능)
         - tickers: 보정된 티커 리스트
      4) Day1Agent(tavily_api_key=...) 인스턴스 생성
      5) agent.handle(query, plan) 호출 → payload(dict) 수신
    반환:
      merge된 표준 스키마 dict (impl/merge.py 참고)
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        print("Warning: TAVILY_API_KEY is not set. Web search will be disabled.")
    tickers = _normalize_kr_tickers(_extract_tickers_from_query(query))
    plan = Day1Plan(
        do_web=True,
        do_stocks=bool(tickers),
        web_keywords=[query],
        tickers=tickers
    )
    agent = Day1Agent(tavily_api_key=api_key)
    return agent.handle(query, plan)


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> Optional[LlmResponse]:
    """
    UI 엔트리포인트:
      1) llm_request.contents[-1]에서 사용자 메시지 텍스트(query) 추출
      2) _handle(query) 호출 → payload 획득
      3) 본문 마크다운 렌더: render_day1(query, payload)
      4) 저장: save_markdown(query, route='day1', markdown=본문MD) → 경로
      5) envelope: render_enveloped('day1', query, payload, saved_path)
      6) LlmResponse로 반환
      7) 예외시 간단한 오류 텍스트 반환
    """
    try:
        last_content = llm_request.contents[-1]
        if last_content.role != "user":
            raise ValueError("Last content is not from user.")
        query = last_content.parts[0].text
        
        payload = _handle(query)
        
        body_md = render_day1(query, payload)
        
        saved_path = save_markdown(query=query, route='day1', markdown=body_md)
        
        md = render_enveloped(kind="day1", query=query, payload=payload, saved_path=saved_path)
        
        return LlmResponse(content=types.Content(parts=[types.Part(text=md)], role="model"))
    except Exception as e:
        error_message = f"Day1 에러: {e}"
        return LlmResponse(content=types.Content(parts=[types.Part(text=error_message)], role="model"))



day1_web_agent = Agent(
    name="Day1WebAgent",
    model=MODEL,
    description="웹 검색 + (필요시) 주가 + 기업 개요 요약 제공",
    instruction=(
        "사용자 질의에 대해 웹 검색 결과와 (티커가 있으면) 주가 스냅샷, "
        "기업 개요 요약을 표준 마크다운으로 제공하라. 필요한 경우 출처 링크를 포함하라."
    ),
    tools=[],
    before_model_callback=before_model_callback,
)
