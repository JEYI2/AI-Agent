# -*- coding: utf-8 -*-
"""
루트 오케스트레이터 (학생용 스켈레톤)
- 목표: 서브 에이전트(Day1/Day2/Day3)를 도구로 연결하고, 프롬프트/모델을 설정
- 구현 없음: TODO만 보고 직접 채우세요.
"""

from __future__ import annotations
from typing import Optional
# apps/root_app/agent.py 최상단 근처
import os, sys
APPS_DIR = os.path.dirname(__file__)                 # .../apps/root_app
PROJECT_ROOT = os.path.dirname(os.path.dirname(APPS_DIR))  # .../ (프로젝트 루트)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

# 서브 에이전트(도구) — 이미 각 day의 agent.py에서 정의되어 있다고 가정
# (모듈 경로가 다르면 프로젝트 구조에 맞게 수정)
from student.day1.agent import day1_web_agent
from student.day2.agent import day2_rag_agent
from student.day3.agent import day3_gov_agent

# 프롬프트(설명/규칙)
from root_app.prompt import ORCHESTRATOR_DESC, ORCHESTRATOR_PROMPT
ORCHESTRATOR_NAME = 'student'

# ------------------------------------------------------------------------------
# TODO[ROOT-A-01] 모델 선택:
#  - 경량 LLM을 선택하여 LiteLlm(model="...")로 초기화
#  - 예: "openai/gpt-4o-mini"
# ------------------------------------------------------------------------------
#MODEL = None  # <- LiteLlm(...)
MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
MODEL = LiteLlm(model="openai/gpt-4o")

# ------------------------------------------------------------------------------
# TODO[ROOT-A-02] 루트 에이전트 구성:
#  요구:
#   - name: Pydantic 제약(영문/숫자/언더스코어만) → 예: "KT_AIVLE_Orchestrator"
#   - model: 위 MODEL 사용
#   - description/instruction: prompts.py에서 작성한 상수 사용
#   - tools: Day1/Day2/Day3를 AgentTool로 감싸 순서대로 등록
#   - before/after 콜백은 필요 없음(기본 LLM-Tool 루프)
# ------------------------------------------------------------------------------
root_agent = Agent(
    name="KT_AIVLE_Orchestrator",  # <- 필요 시 수정(하이픈 금지!)
    model=MODEL,                   # <- TODO[ROOT-A-01]
    description=ORCHESTRATOR_DESC, # <- TODO[ROOT-P-01]
    instruction=ORCHESTRATOR_PROMPT,  # <- TODO[ROOT-P-02]
    tools=[
        # TODO[ROOT-A-03] 도구 등록: 아래 3개는 기본 예시
        #  - 필요 시 순서를 바꾸거나 disable할 수도 있음
        AgentTool(agent=day1_web_agent),
        AgentTool(agent=day2_rag_agent),
        AgentTool(agent=day3_gov_agent),
    ],
)
