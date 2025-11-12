# -*- coding: utf-8 -*-
"""
PPS 검색 에이전트
- 실제 구현은 impl/pps_tool.py의 pps_search()에 모두 포함
- FunctionTool.from_callable 사용
"""
from __future__ import annotations
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.function_tool import FunctionTool
from student.day3.impl.pps_tool import pps_search 

MODEL = LiteLlm(model=os.getenv("DAY4_INTENT_MODEL","gpt-4o-mini"))

pps_tool = FunctionTool.from_callable(
    func=pps_search,
    name="PpsBidsSearch",
    description="나라장터 입찰공고(최근 N일) 검색. 입력=키워드(예: 헬스케어, AI). 날짜/행수는 .env(PPS_*)로 제어."
)

day3_pps_agent = Agent(
    name="Day3PpsAgent",
    model=MODEL,
    instruction="키워드를 받으면 나라장터(OpenAPI)에서 최근 공고를 표로 만들고 저장하라.",
    tools=[pps_tool],
)