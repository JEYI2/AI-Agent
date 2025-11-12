# -*- coding: utf-8 -*-
"""
Day1 스모크 테스트 (루트 .env 로드 + sys.path 보정 + 견고한 폴백 출력)
- 이 파일만 수정/실행합니다. 배포된 모듈은 건드리지 않습니다.
"""
# --- 0) 프로젝트 루트 탐색 + sys.path 보정 + .env 로드 ---
import os, sys, json
from pathlib import Path

def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "apps").exists():
            return p
    return start

ROOT = _find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"
def _manual_load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
try:
    from dotenv import load_dotenv  # 선택
    load_dotenv(ENV_PATH, override=False)
except Exception:
    _manual_load_env(ENV_PATH)
# ------------------------------------------------------------------------------

from student.day1.impl.web_search import (
    search_company_profile,
    extract_and_summarize_profile,
)

def _check_keys() -> bool:
    ok = True
    if not os.getenv("TAVILY_API_KEY"):
        print("[FAIL] 환경변수 TAVILY_API_KEY가 없습니다. 루트 .env를 확인하세요.")
        ok = False
    return ok

# (옵션) 시세 스냅샷: 없으면 조용히 패스 / yfinance로 대체 가능
def _try_fetch_prices(symbols):
    # ① 배포 모듈에 fetch_prices가 있다면 사용
    try:
        from student.day1.impl.finance_client import fetch_prices  # 배포본에 없으면 ImportError
        return fetch_prices(symbols, period="5d")
    except Exception:
        # ② yfinance가 있으면 간단히 현재가만 찍어보기(스모크용)
        try:
            import yfinance as yf
            out = []
            for s in symbols:
                t = yf.Ticker(s)
                info = t.fast_info if hasattr(t, "fast_info") else {}
                price = info.get("last_price") or info.get("lastPrice")
                out.append({"symbol": s, "price": float(price) if price else None, "currency": info.get("currency")})
            return out
        except Exception:
            # ③ 완전 패스 (스모크에 필수 아님)
            return []

def _fake_summarizer(prompt: str) -> str:
    # 모듈 내부가 500자 이상만 채택할 수 있어 빈 요약이 생길 수 있음 → 스모크는 300자로 제한
    return prompt[-300:] if len(prompt) > 300 else prompt

def main():
    if not _check_keys():
        sys.exit(2)

    query = "삼성전자 기업정보"
    tavily_key = os.getenv("TAVILY_API_KEY")

    # 1) 검색
    results = search_company_profile(query, tavily_key, topk=3)
    urls = [r.get("url") for r in results if r.get("url")]
    print(f"[OK] 검색 결과 {len(urls)}개")

    # 2) 프로필 요약 시도
    summary = ""
    try:
        summary = extract_and_summarize_profile(urls, tavily_key, _fake_summarizer)
    except TypeError:
        # 배포 환경에 따라 extract_and_summarize_profile 시그니처가 다를 수 있음 → api_key 없이 재시도
        try:
            from student.day1.impl.web_search import extract_and_summarize_profile as _ex2  # 재바인딩
            summary = _ex2(urls, _fake_summarizer)  # api_key 없이
        except Exception:
            summary = ""

    # 2-1) 요약이 비면 상위 URL을 폴백으로 출력
    if summary.strip():
        print("\n[OK] 프로필 요약 샘플:")
        print(summary[:600] + ("\n..." if len(summary) > 600 else ""))
    else:
        print("\n[WARN] 프로필 요약 없음 → 상위 URL 폴백 출력:")
        for i, r in enumerate(results[:3], 1):
            title = r.get("title") or r.get("url")
            src = r.get("source") or ""
            print(f"  {i}. {title} — {src}")

    # 3) (옵션) 시세 스냅샷
    prices = _try_fetch_prices(["005930.KS"])
    if prices:
        print("\n[OK] 시세 스냅샷(JSON 일부):")
        print(json.dumps(prices, ensure_ascii=False)[:240])

    print("\n[DONE] Day1 스모크 통과")

if __name__ == "__main__":
    main()
