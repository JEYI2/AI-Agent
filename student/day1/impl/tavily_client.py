# -*- coding: utf-8 -*-
import os, requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

TAVILY_BASE = "https://api.tavily.com"

def _headers(api_key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

def search_tavily(
    query: str,
    api_key: Optional[str],
    top_k: int = 6,
    timeout: int = 20,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_depth: str = "basic",
    include_answer: bool = False,
    include_images: bool = False,
    include_raw_content: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required for web search")

    payload: Dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "max_results": top_k,
        "top_k": top_k,
        "include_answer": include_answer,
        "include_images": include_images,
        "include_raw_content": include_raw_content,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    payload.update({k: v for k, v in kwargs.items() if v is not None})

    r = requests.post(f"{TAVILY_BASE}/search", headers=_headers(api_key), json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("results", []) or []

def extract_url(url: str) -> str:
    """URL을 정리(normalize)해서 반환 (추적 파라미터/fragment 제거)"""
    if not url:
        return ""
    url = url.strip()
    try:
        parts = urlsplit(url)
        qs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
              if not (k.lower().startswith("utm_") or k.lower() in {"fbclid", "gclid"})]
        cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(qs), ""))
        return cleaned
    except Exception:
        return url

# 본문 추출 (Tavily Extract API 사용)
def extract_text(url: str, api_key: Optional[str], timeout: int = 20) -> str:
    """
    주어진 URL에서 본문 텍스트를 추출해 반환.
    - Tavily의 /extract 엔드포인트를 사용 (서비스 정책/응답 스키마 변화 가능성 있어 방어적 처리)
    - 실패하면 빈 문자열 반환
    """
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required for extract")
    try:
        payload = {"url": url}
        r = requests.post(f"{TAVILY_BASE}/extract", headers=_headers(api_key), json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # 다양한 응답 스키마를 방어적으로 지원
        # 1) {"content": "..."}  2) {"result":"..."}  3) {"results":[{"content":"..."}]}
        if isinstance(data, dict):
            if "content" in data and isinstance(data["content"], str):
                return data["content"]
            if "result" in data and isinstance(data["result"], str):
                return data["result"]
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                first = data["results"][0]
                if isinstance(first, dict) and isinstance(first.get("content"), str):
                    return first["content"]
    except Exception:
        pass
    return ""
