# -*- coding: utf-8 -*-
"""
Day3 스모크 테스트
- 검사항목:
  1) .env 로드/키 확인
  2) import 경로/패키지 구조 확인(student/day3/impl/*)
  3) find_notices(query) 정상 반환/미리보기
  4) 실패 시 단계별(fetch_nipa/bizinfo/web) 분해 점검
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# 유틸: 프로젝트 루트/경로 보정 & .env 로드(Windows/macOS)
ROOT_SENTINELS = ("uv.lock", "pyproject.toml", "apps", "student", ".git")

def _guess_project_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        try:
            if any((p / m).exists() for m in ROOT_SENTINELS):
                return p
        except Exception:
            pass
    return Path.cwd().resolve()

def _ensure_sys_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

def _load_env_if_any(root: Path) -> Path:
    env_path = root / ".env"
    loaded = False
    if env_path.exists():
        # python-dotenv이 있으면 사용(없어도 무시)
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(str(env_path), override=False)
            loaded = True
        except Exception:
            # 매우 단순 파서( key=value 라인만 )
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
            loaded = True
    print(f"[INFO] ROOT: {root}")
    print(f"[INFO] .env : {env_path} | OPENAI_API_KEY: {bool(os.getenv('OPENAI_API_KEY'))}")
    return env_path if loaded else Path("")

# find_notices 반환값 → 리스트로 정규화(안전 처리)
def _as_list(maybe_items: Any) -> List[Dict[str, Any]]:
    # 1) 딕셔너리: {"items":[...]} 우선
    if isinstance(maybe_items, dict):
        if "items" in maybe_items and isinstance(maybe_items["items"], list):
            return maybe_items["items"]
        # 혹시 다른 키가 리스트인 경우
        for v in maybe_items.values():
            if isinstance(v, list):
                return v
        return []
    # 2) 리스트 그대로
    if isinstance(maybe_items, list):
        return maybe_items
    # 3) (items, meta) 형태 튜플
    if isinstance(maybe_items, tuple) and maybe_items:
        first = maybe_items[0]
        if isinstance(first, list):
            return first
    return []

def _print_env_snapshot() -> None:
    keys = [
        "NIPA_LIST_URL",
        "NIPA_MAX_PAGES",
        "NIPA_MIN_YEAR",
        "NIPA_MAX_ITEMS",
        "TAVILY_API_KEY",
    ]
    print("[INFO] 환경 변수 스냅샷:")
    for k in keys:
        v = os.getenv(k)
        if not v:
            print(f"  - {k} = (미설정)")
        elif k == "TAVILY_API_KEY":
            print(f"  - {k} = {v[:4]}...{v[-4:]}")
        else:
            # 긴 URL은 짧게
            val = (v[:50] + "...") if len(v) > 60 else v
            print(f"  - {k} = {val}")

# 미리보기 & 단계별 점검
def _preview_items(items_raw: Any, limit: int = 5) -> None:
    items = _as_list(items_raw)
    print(f"[OK] 표준 스키마 아이템 수={len(items)} (상위 {min(limit, len(items))}개 미리보기)")
    for i, it in enumerate(items[:limit], 1):
        title = it.get("title", "-")
        agency = it.get("agency", "-")
        ann = it.get("announce_date", "-")
        close = it.get("close_date", "-")
        budget = it.get("budget", "-")
        score = float(it.get("score", 0) or 0)
        url = it.get("url", "")
        print(f"  [{i}] {title} | {agency} | 공고일={ann} | 마감={close} | 예산={budget} | 점수={score:.3f}")
        if url:
            print(f"      URL: {url}")

def _diagnostic_hints() -> None:
    print("\n[DIAG HINTS]")
    print(" - GOV_ONLY=1 이면 NIPA/화이트리스트 외 도메인 결과가 모두 필터링될 수 있습니다.")
    print(" - NIPA_MIN_YEAR 가 너무 크게 잡히면 오래된 공고만 걸러져 0건이 될 수 있습니다.")
    print(" - GOV_WEB_WHITELIST 가 너무 좁으면 링크가 전부 제외될 수 있습니다.")
    print(" - Tavily 키/쿼터/지역 제한 이슈 가능성 점검.")
    print(" - 네트워크(프록시/방화벽)로 인한 요청 차단 여부 확인.")

def _run_fallbacks(query: str) -> None:
    """소스별 단독 점검 (검색만 테스트)"""
    try:
        from student.day3.impl.fetchers import fetch_nipa, fetch_bizinfo, fetch_web  # type: ignore
    except Exception as e:
        print(f"[FAIL] fetchers 임포트 실패: {e}")
        return

    # NIPA
    try:
        print("\n[STEP] fetch_nipa() 점검… (site:nipa.kr + 공고/모집/지원)")
        rs = fetch_nipa(query)
        print(f"  - 결과 {len(rs)}건")
        for i, r in enumerate(rs[:3], 1):
            print(f"    [{i}] {r.get('title') or r.get('url')} | {r.get('url','')}")
    except Exception as e:
        print(f"  [WARN] fetch_nipa 실패: {e}")

    # Bizinfo
    try:
        print("\n[STEP] fetch_bizinfo() 점검… (site:bizinfo.go.kr)")
        rs = fetch_bizinfo(query)
        print(f"  - 결과 {len(rs)}건")
        for i, r in enumerate(rs[:3], 1):
            print(f"    [{i}] {r.get('title') or r.get('url')} | {r.get('url','')}")
    except Exception as e:
        print(f"  [WARN] fetch_bizinfo 실패: {e}")

    # Web
    try:
        print("\n[STEP] fetch_web() 점검… (일반 웹 Fallback)")
        rs = fetch_web(query)
        print(f"  - 결과 {len(rs)}건")
        for i, r in enumerate(rs[:3], 1):
            print(f"    [{i}] {r.get('title') or r.get('url')} | {r.get('url','')}")
    except Exception as e:
        print(f"  [WARN] fetch_web 실패: {e}")

# 메인 실행
def main() -> int:
    root = _guess_project_root()
    _ensure_sys_path(root)
    _load_env_if_any(root)

    # 환경 스냅샷
    _print_env_snapshot()

    # lazy import (경로 확정 후)
    try:
        from student.day3.impl.pipeline import find_notices  # type: ignore
    except Exception as e:
        print("[FAIL] 임포트 실패: student.day3.impl 모듈을 불러오지 못했습니다.")
        import traceback; traceback.print_exc()
        print("\n[HINT]\n - 파일/패키지 경로가 맞는지 확인 (student/day3/impl/*)\n"
              " - 루트에서 실행했는지 확인 (python student/day3/smoke_test.py)\n"
              " - __init__.py 존재 여부 확인")
        return 1

    # 질의 설정
    query = "모집 공고" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print(f"\n[STEP] find_notices(query='{query}') 호출…")
    try:
        items_raw = find_notices(query=query)
        _preview_items(items_raw, limit=5)
        items = _as_list(items_raw)
        if not items:
            print("\n[WARN] 표준 스키마 아이템이 0건입니다. 소스별 Fallback 점검을 진행합니다.")
            _run_fallbacks(query)
            _diagnostic_hints()
            return 2
        print("\n[OK] Day3 파이프라인 스모크 테스트 통과")
        return 0
    except Exception as e:
        print("[FAIL] find_notices 실패")
        import traceback; traceback.print_exc()
        _diagnostic_hints()
        return 3

if __name__ == "__main__":
    raise SystemExit(main())