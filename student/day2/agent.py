# -*- coding: utf-8 -*-
"""
Day2: RAG 도구 에이전트
- 역할: Day2 RAG 본체 호출 → 결과 렌더 → 저장(envelope) → 응답
"""

from __future__ import annotations
from typing import Dict, Any
import os

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from student.day2.impl.rag import Day2Agent
from student.common.writer import render_day2, render_enveloped
from student.common.schemas import Day2Plan
from student.common.fs_utils import save_markdown


# ------------------------------------------------------------------------------
# TODO[DAY2-A-01] 모델 선택
#  - LiteLlm(model="openai/gpt-4o-mini") 등 경량 모델 지정
# ------------------------------------------------------------------------------
MODEL = LiteLlm(model="openai/gpt-4o-mini")


def _handle(query: str) -> Dict[str, Any]:
    """
    1) plan = Day2Plan()  (필요 시 top_k 등 파라미터 명시)
    2) agent = Day2Agent(index_dir=os.getenv("DAY2_INDEX_DIR","indices/day2"))
    3) return agent.handle(query, plan)
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-A-02] 구현 지침
    #  - plan = Day2Plan()
    #  - index_dir = os.getenv("DAY2_INDEX_DIR", "indices/day2")
    #  - agent = Day2Agent(index_dir=index_dir)
    #  - payload = agent.handle(query, plan); return payload
    # ----------------------------------------------------------------------------

    # 1) 실행 계획
    plan = Day2Plan()  # 필요하면 Day2Plan(top_k=5) 등 파라미터 추가

    # 2) 인덱스 경로(환경변수 우선)
    index_dir = os.getenv("DAY2_INDEX_DIR", "indices/day2")

    # 3) 에이전트 생성 & 실행
    agent = Day2Agent(index_dir=index_dir)
    payload: Dict[str, Any] = agent.handle(query, plan)
    return payload



def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> LlmResponse | None:
    """
    1) 사용자 메시지에서 query 텍스트 추출
    2) payload = _handle(query)
    3) body_md = render_day2(query, payload)
    4) saved = save_markdown(query, 'day2', body_md)
    5) md = render_enveloped('day2', query, payload, saved)
    6) LlmResponse로 반환 (예외 발생 시 간단 메시지)
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-A-03] 구현 지침
    #  - last = llm_request.contents[-1]
    #  - query = last.parts[0].text
    #  - payload → 렌더/저장/envelope → 응답
    # ----------------------------------------------------------------------------
    try:
        history = getattr(llm_request, "contents", []) or []
        if not history:
            return None

        last = history[-1]
        if getattr(last, "role", None) != "user":
            # 사용자 메시지가 아니면 콜백에서 빠져나가 모델 기본 흐름을 진행
            return None

        parts = getattr(last, "parts", []) or []
        query = (getattr(parts[0], "text", None) or "").strip() if parts else ""
        if not query:
            return LlmResponse(
                content=types.Content(
                    parts=[types.Part(text="입력 문장을 찾지 못했습니다.")],
                    role="model",
                )
            )

        # 1) 본체 호출
        payload: Dict[str, Any] = _handle(query)

        # 2) 렌더 + 저장
        body_md: str = render_day2(query, payload)
        saved_path: str = save_markdown(query=query, route="day2", markdown=body_md)

        # 3) envelope 마크다운 생성
        md: str = render_enveloped(kind="day2", query=query, payload=payload, saved_path=saved_path)

        # 4) 모델 응답 형태로 반환
        return LlmResponse(
            content=types.Content(parts=[types.Part(text=md)], role="model")
        )

    except Exception as e:
        return LlmResponse(
            content=types.Content(
                parts=[types.Part(text=f"Day2 에러: {e}")],
                role="model",
            )
        )



day2_rag_agent = Agent(
    name="Day2RagAgent",
    model=MODEL,
    description="로컬 인덱스를 활용한 RAG 요약/근거 제공",
    instruction="사용자 질의와 관련된 문서를 인덱스에서 찾아 요약하고 근거를 함께 제시하라.",
    tools=[],
    before_model_callback=before_model_callback,
)
