# -*- coding: utf-8 -*-
from typing import Dict, Any
from textwrap import dedent

# --------- 본문 렌더러들 ---------
def render_day1(query: str, payload: Dict[str, Any]) -> str:
    web = payload.get("web_top", []) or []
    prices = payload.get("prices", []) or []
    profile = (payload.get("company_profile") or "").strip()
    profile_sources = payload.get("profile_sources") or []

    lines = [f"# 웹 리서치 리포트", f"- 질의: {query}", ""]

    # 1) 시세 스냅샷
    if prices:
        lines.append("## 시세 스냅샷")
        for p in prices:
            sym = p.get("symbol", "")
            cur = f" {p.get('currency')}" if p.get("currency") else ""
            if p.get("price") is not None:
                lines.append(f"- **{sym}**: {p['price']}{cur}")
            else:
                lines.append(f"- **{sym}**: (가져오기 실패) — {p.get('error','')}")
        lines.append("")

    # 2) 기업 정보 요약(발췌 + 출처)
    if profile:
        # 500자 정도로 길이 제한(가독)
        short = profile[:500].rstrip()
        if len(profile) > 500:
            short += "…"
        lines.append("## 기업 정보 요약")
        lines.append(short)
        if profile_sources:
            lines.append("")
            lines.append("**출처(기업 정보):**")
            for u in profile_sources[:3]:
                lines.append(f"- {u}")
        lines.append("")

    # 3) 상위 웹 결과(타이틀 + 메타 + 2줄 발췌)
    if web:
        lines.append("## 관련 링크 & 발췌")
        for r in web[:5]:
            title = r.get("title") or r.get("url") or "link"
            src = r.get("source") or ""
            date = r.get("published_date") or r.get("date") or ""
            url = r.get("url", "")
            tail = f" — {src}" + (f" ({date})" if date else "")
            lines.append(f"- [{title}]({url}){tail}")

            # 2줄 발췌: content > snippet > '' 우선순위
            raw = (r.get("content") or r.get("snippet") or "").strip().replace("\n", " ")
            if raw:
                excerpt = raw[:280].rstrip()
                if len(raw) > 280:
                    excerpt += "…"
                lines.append(f"  > {excerpt}")
        lines.append("")

    # 웹 결과가 전혀 없을 때 힌트
    if not (web or profile or prices):
        lines.append("_참고: 결과가 비어있습니다. 쿼리/도메인 제한/키워드 설정을 확인하세요._")
        lines.append("")

    return "\n".join(lines)


def render_day2(query: str, payload: dict) -> str:
    # 기존 요약/머리말 생성부는 유지
    lines = []
    lines.append(f"# Day2 – RAG 요약")
    lines.append("")
    lines.append(f"**질의:** {query}")
    lines.append("")

    # ── 추가: 초안(answer) 표시
    answer = (payload or {}).get("answer") or ""
    if answer:
        lines.append("## 초안 요약")
        lines.append("")
        lines.append(answer.strip())
        lines.append("")

    # ── 추가: 근거 상위 K 표
    contexts = (payload or {}).get("contexts") or []
    if contexts:
        lines.append("## 근거(Top-K)")
        lines.append("")
        lines.append("| rank | score | path | chunk_id | excerpt |")
        lines.append("|---:|---:|---|---:|---|")
        for i, c in enumerate(contexts, 1):
            score = f"{float(c.get('score', 0.0)):.3f}"
            path = str(c.get("path") or c.get("meta", {}).get("path") or "")

            # excerpt 후보(우선순위: text > chunk > content)
            raw = (
                c.get("text")
                or c.get("chunk")
                or c.get("content")
                or ""
            )
            excerpt = (str(raw).replace("\n", " ").strip())[:200]

            # chunk_id 후보(우선순위: id > meta.chunk > chunk_id > chunk_index)
            chunk_id = (
                c.get("id")
                or c.get("meta", {}).get("chunk")
                or c.get("chunk_id")
                or c.get("chunk_index")
                or ""
            )

            lines.append(f"| {i} | {score} | {path} | {chunk_id} | {excerpt} |")
        lines.append("")

    return "\n".join(lines)

def render_day3(query: str, payload: Dict[str, Any]) -> str:
    items = payload.get("items", [])
    lines = [f"# 공고 탐색 결과", f"- 질의: {query}", ""]
    if items:
        lines.append("| 출처 | 제목 | 기관 | 접수 마감 | 예산 | URL | 점수 |")
        lines.append("|---|---|---|---:|---:|---|---:|")
        for it in items[:10]:
            src = it.get('source','-')
            title = it.get('title','-')
            agency = it.get('agency','-')
            close = it.get('close_date','-')
            budget = it.get('budget','-')
            url = it.get('url','-')
            score = it.get('score',0)
            lines.append(f"| {src} | {title} | {agency} | {close or '-'} | {budget or '-'} | {url} | {score:.3f} |")
    else:
        lines.append("관련 공고를 찾지 못했습니다.")
        
    has_atts = any(it.get("attachments") for it in items)
    if has_atts:
        lines.append("\n## 첨부파일 요약")
        for i, it in enumerate(items[:10], 1):
            atts = it.get("attachments") or []
            if not atts: 
                continue
            lines.append(f"- **{i}. {it.get('title','(제목)')}**")
            for a in atts[:5]:
                lines.append(f"  - {a}")
    return "\n".join(lines)

# --------- Envelope(머리말/푸터) ---------
def _compose_envelope(kind: str, query: str, body_md: str, saved_path: str) -> str:
    header = dedent(f"""\
    ---
    output_schema: v1
    type: markdown
    route: {kind}
    saved: {saved_path}
    query: "{query.replace('"','\\\"')}"
    ---

    """)
    footer = dedent(f"""\n\n---\n> 저장 위치: `{saved_path}`\n""")
    return header + body_md.strip() + footer

def render_enveloped(kind: str, query: str, payload: Dict[str, Any], saved_path: str) -> str:
    if kind == "day1":
        body = render_day1(query, payload)
    elif kind == "day2":
        body = render_day2(query, payload)
    elif kind == "day3":
        body = render_day3(query, payload)
    else:
        body = f"### 결과\n\n(알 수 없는 kind: {kind})"
    return _compose_envelope(kind, query, body, saved_path)
