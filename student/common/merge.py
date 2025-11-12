# -*- coding: utf-8 -*-
"""
Day4 merge helpers
- 목적: Day1(웹) + Day2(RAG) 결과를 정책에 따라 병합
- 사용처: 라우터가 웹/RAG를 병렬 호출하거나, RAG 라우트에서 게이팅 미통과 시 웹 폴백이 필요할 때
- 출력 스키마: merged_day1_day2 (writer에서 그대로 렌더 가능)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

# ----------------------------
# 내부 헬퍼
# ----------------------------
def _pick_web_items(web_payload: Dict[str, Any], topk: int = 5) -> List[Dict[str, Any]]:
    if not web_payload or web_payload.get("type") != "web_results":
        return []
    return (web_payload.get("items") or [])[:topk]

def _combine_tickers(web_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not web_payload or web_payload.get("type") != "web_results":
        return []
    return web_payload.get("tickers") or []

def _should_use_rag(rag_payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    반환: (use_rag, reason)
    - plan.force_rag_only: True면 무조건 사용
    - gating.status == 'enough'이면 사용
    - 그 외에는 사용 안 함
    """
    if not rag_payload or rag_payload.get("type") != "rag_answer":
        return False, "rag:missing_or_invalid"
    plan = rag_payload.get("plan", {}) or {}
    gating = rag_payload.get("gating", {}) or {}
    if plan.get("force_rag_only"):
        return True, "rag:forced_by_user"
    if gating.get("status") == "enough":
        return True, "rag:gating_pass"
    return False, "rag:gating_fail"

def _web_confidence(web_payload: Dict[str, Any]) -> float:
    """
    매우 단순한 웹 신뢰 지표:
    - 아이템 개수 + 티커 유무로 0~1 스케일
    """
    if not web_payload or web_payload.get("type") != "web_results":
        return 0.0
    n_items = len(web_payload.get("items") or [])
    has_tickers = bool(web_payload.get("tickers"))
    # item 5개 이상 + 티커 있으면 1.0, 아니면 개수 기반 완만히 증가
    base = min(1.0, n_items / 5.0)
    if has_tickers:
        base = min(1.0, base + 0.2)
    return max(0.0, min(1.0, base))

def _rag_confidence(rag_payload: Dict[str, Any]) -> float:
    """
    간단 RAG 신뢰 지표:
    - gating.mean_topk (0~1 범위 가정) 사용, 없으면 0
    """
    if not rag_payload or rag_payload.get("type") != "rag_answer":
        return 0.0
    g = rag_payload.get("gating", {}) or {}
    return float(g.get("mean_topk", 0.0))

# ----------------------------
# 병합 전략 결정
# ----------------------------
def decide_strategy(web_payload: Dict[str, Any], rag_payload: Dict[str, Any]) -> str:
    """
    반환: 'web_only' | 'rag_only' | 'web_plus_rag'
    정책:
      - RAG 강제(force) → rag_only (웹은 참고만)
      - RAG 게이팅 통과:
          * 웹 신뢰가 높으면 web_plus_rag
          * 아니면 rag_only
      - RAG 미통과 → web_only
    """
    use_rag, reason = _should_use_rag(rag_payload)
    if not use_rag:
        return "web_only"

    plan = rag_payload.get("plan", {}) or {}
    if plan.get("force_rag_only"):
        return "rag_only"

    # 게이팅 통과 시: 웹 신뢰 보정으로 동시 사용 여부 결정
    w_conf = _web_confidence(web_payload)
    r_conf = _rag_confidence(rag_payload)

    # 웹 근거도 충분(>=0.5)이면 둘 다 보여주자
    if w_conf >= 0.5:
        return "web_plus_rag"

    # 웹 미약하면 RAG가 주도
    if r_conf >= 0.35:
        return "rag_only"

    # 애매하면 둘 다 (보수적)
    return "web_plus_rag"

# ----------------------------
# 메인 병합 함수
# ----------------------------
def merge_day1_day2(web_payload: Dict[str, Any], rag_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    반환 스키마:
    {
      "type": "merged_day1_day2",
      "strategy": "web_only|rag_only|web_plus_rag",
      "web_used": bool,
      "rag_used": bool,
      "web": { "items": [...], "tickers": [...], "errors": [...] },
      "rag": { "gating": {...}, "contexts": [...], "answer": "..." , "notice": "used|gated_out|forced" },
      "confidence": {"web": float, "rag": float},
      "notes": [str, ...]
    }
    """
    notes: List[str] = []

    # 기본 구성
    merged: Dict[str, Any] = {
        "type": "merged_day1_day2",
        "strategy": "web_only",
        "web_used": False,
        "rag_used": False,
        "web": {"items": [], "tickers": [], "errors": []},
        "rag": {"gating": {}, "contexts": [], "answer": "", "notice": ""},
        "confidence": {"web": 0.0, "rag": 0.0},
        "notes": notes,
    }

    # 웹 섹션
    top_items = _pick_web_items(web_payload, topk=5)
    tickers = _combine_tickers(web_payload)
    errors = (web_payload or {}).get("errors", []) if isinstance(web_payload, dict) else []
    merged["web"].update({"items": top_items, "tickers": tickers, "errors": errors})
    merged["confidence"]["web"] = _web_confidence(web_payload)

    # RAG 섹션
    if rag_payload and rag_payload.get("type") == "rag_answer":
        gating = rag_payload.get("gating", {}) or {}
        contexts = (rag_payload.get("contexts") or [])[:5]
        answer = rag_payload.get("answer", "") or ""
        merged["rag"].update({"gating": gating, "contexts": contexts, "answer": ""})
        merged["confidence"]["rag"] = _rag_confidence(rag_payload)

        use_rag, reason = _should_use_rag(rag_payload)
        if use_rag:
            merged["rag"]["answer"] = answer
            merged["rag"]["notice"] = "used" if gating.get("status") == "enough" else "forced"
            notes.append(f"{reason}")
        else:
            merged["rag"]["notice"] = "gated_out"
            notes.append(f"{reason}")
    else:
        notes.append("rag:missing_or_invalid")

    # 전략 결정
    strategy = decide_strategy(web_payload, rag_payload)
    merged["strategy"] = strategy

    # 사용 플래그
    merged["web_used"]  = strategy in ("web_only", "web_plus_rag")  and (len(top_items) > 0 or len(tickers) > 0)
    merged["rag_used"]  = strategy in ("rag_only", "web_plus_rag") and (merged["rag"]["notice"] in ("used","forced"))

    return merged
