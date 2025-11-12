# -*- coding: utf-8 -*-
"""
impl 전용: PPS(나라장터) 검색 + 렌더 + 저장 유틸
- writer.py / fs_utils.py에 의존하지 않도록 독립 저장 로직 포함
- .env의 PPS_* 파라미터를 흡수하고, 없으면 최근 N일(기본 30일)로 자동
- pps_api 함수명/시그니처 차이에 방어적으로 대응
"""
from __future__ import annotations
import os, re
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# pps_api: 환경마다 함수명이 다를 수 있으므로 유연하게 import
try:
    from student.day3.impl.pps_api import pps_fetch_bids as _FETCH  # type: ignore[attr-defined]
except Exception:
    try:
        from student.day3.impl.pps_api import fetch_pps_notices as _FETCH  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _FETCH = None  # type: ignore[assignment]

try:
    from student.day3.impl.pps_api import to_common_schema as _TO_COMMON  # type: ignore[attr-defined]
except Exception:
    _TO_COMMON = None  # type: ignore[assignment]

KST = timezone(timedelta(hours=9))

# 경로/저장 유틸(독립 구현)
def _slugify(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9A-Za-z가-힣\-_\.]+", "", text)
    return text[:120] or "output"

def _find_project_root() -> Path:
    start = Path(__file__).resolve()
    markers = ("uv.lock", "pyproject.toml", "apps", "student", ".git")
    for p in [start, *start.parents]:
        try:
            if any((p / m).exists() for m in markers):
                return p
        except Exception:
            pass
    return Path.cwd().resolve()

def _default_output_dir() -> Path:
    env_dir = os.getenv("OUTPUT_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (_find_project_root() / "data" / "processed").resolve()

def _save_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)

# 파라미터 해석
def _yyyymmddhhmm(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")

@dataclass
class PpsParams:
    keyword: str
    date_from: str
    date_to: str
    rows: int
    page_max: int
    inqry_div: str  # 쿼리 구분(예: '1': 공고)

def resolve_params(user_query: str) -> PpsParams:
    # 날짜 기본: 최근 N일
    last_days = int(os.getenv("PPS_DEFAULT_LAST_DAYS", "30") or "30")
    now = datetime.now(KST)
    default_from = _yyyymmddhhmm(now - timedelta(days=last_days))
    default_to = _yyyymmddhhmm(now)

    date_from = (os.getenv("PPS_DATE_FROM", "") or default_from).replace(" ", "").replace("-", "")
    date_to   = (os.getenv("PPS_DATE_TO", "")   or default_to).replace(" ", "").replace("-", "")
    rows      = int(os.getenv("PPS_ROWS", "100") or "100")
    page_max  = int(os.getenv("PPS_PAGE_MAX", "3") or "3")
    inqry_div = os.getenv("PPS_INQRY_DIV", "1").strip() or "1"

    keyword = (user_query or os.getenv("PPS_DEFAULT_QUERY", "")).strip()
    return PpsParams(keyword=keyword, date_from=date_from, date_to=date_to,
                     rows=rows, page_max=page_max, inqry_div=inqry_div)

# 데이터 렌더 & 저장
def _render_table(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "관련 공고를 찾지 못했습니다."
    lines = [
        "| 공고명 | 발주기관 | 공고번호 | 공고일자 | 마감일자 | 예산 | 링크 | 첨부 |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    def link(u: str) -> str:
        return f"[바로가기]({u})" if u else "-"
    def attach(a: Any) -> str:
        if not a:
            return "-"
        if isinstance(a, list):
            head = []
            for i, att in enumerate(a[:2], 1):
                name = att.get("name") or att.get("title") or f"첨부{i}"
                url = att.get("url") or ""
                head.append(f"[{name}]({url})" if url else name)
            if len(a) > 2:
                head.append(f"...(+{len(a)-2})")
            return ", ".join(head)
        return str(a)

    for it in items[:30]:
        lines.append(
            "| {title} | {agency} | {bid_no} | {ann} | {close} | {budget} | {url} | {att} |".format(
                title=it.get("title", "-"),
                agency=it.get("agency", "-"),
                bid_no=it.get("bid_no", "") or it.get("bidNo", ""),
                ann=it.get("announce_date", "") or it.get("announceDate", ""),
                close=it.get("close_date", "") or it.get("closeDate", ""),
                budget=it.get("budget", "-"),
                url=link(it.get("url", "")),
                att=attach(it.get("attachments")),
            )
        )
    return "\n".join(lines)

def _render_markdown(query: str, items: List[Dict[str, Any]], saved_path: str) -> str:
    header = (
        f"---\noutput_schema: v1\ntype: markdown\nroute: pps\n"
        f"saved: {saved_path}\nquery: \"{query.replace('\"','\\\"')}\"\n---\n\n"
    )
    body = [f"# 나라장터 입찰공고(최근)","",f"- 질의: {query}","",_render_table(items)]
    footer = f"\n\n---\n> 저장 위치: `{saved_path}`\n"
    return header + "\n".join(body) + footer

def save_markdown(query: str, items: List[Dict[str, Any]], route: str="pps") -> str:
    outdir = _default_output_dir()
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}__{route}__{_slugify(query)}.md"
    abspath = (outdir / fname).resolve()
    md = _render_markdown(query, items, saved_path=str(abspath))
    _save_text(abspath, md)
    return str(abspath)

# 검색 실행(외부 노출 함수)
def pps_search(query: str) -> str:
    """
    1) 파라미터 해석(.env 흡수, 최근 N일 기본)
    2) pps_api 호출(함수명/시그니처 다양성 방어)
    3) to_common_schema 있으면 정규화, 없으면 최소 매핑
    4) 마크다운 저장 후 본문 반환
    """
    if _FETCH is None:
        return "⚠️ PPS API 모듈(student/day3/impl/pps_api.py)을 찾을 수 없습니다."

    p = resolve_params(query)

    # 호출(함수 시그니처 차이를 방어적으로 처리)
    try:
        raw = _FETCH(
            keyword=p.keyword,
            date_from=p.date_from,
            date_to=p.date_to,
            rows=p.rows,
            page_max=p.page_max,
            inqry_div=p.inqry_div,
        )
    except TypeError:
        # 일부 구현은 (keyword, page_max)만 받도록 단순화됐을 수 있음
        raw = _FETCH(keyword=p.keyword, page_max=p.page_max)  # type: ignore[misc]

    # 공통 스키마 정규화
    items: List[Dict[str, Any]] = []
    if _TO_COMMON:
        try:
            items = _TO_COMMON(raw)
        except Exception:
            items = []

    if not items:
        # 최소 매핑
        for r in (raw or []):
            items.append({
                "title": r.get("bidNtceNm") or r.get("title") or "-",
                "agency": r.get("dminsttNm") or r.get("agency") or r.get("organ", "-"),
                "bid_no": r.get("bidNtceNo") or r.get("bid_no") or r.get("bidNo", ""),
                "announce_date": r.get("bidNtceDate") or r.get("announce_date") or r.get("announceDate", ""),
                "close_date": r.get("opengDt") or r.get("close_date") or r.get("closeDate", ""),
                "budget": r.get("presmptPrce") or r.get("budget", "-"),
                "url": r.get("bidNtceDtlUrl") or r.get("url", ""),
                "attachments": r.get("atchFileList") or r.get("attachments", []),
            })

    # 저장 + 본문 반환
    save_markdown(p.keyword or query, items, route="pps")
    # 렌더 본문만 반환(ADK FunctionTool은 문자열 반환이 간단)
    return _render_markdown(p.keyword or query, items, saved_path="(see header)")