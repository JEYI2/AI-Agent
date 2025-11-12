# -*- coding: utf-8 -*-
"""
yfinance 가격 조회
- 목표: 티커 리스트에 대해 현재가/통화를 가져와 표준 형태로 반환
"""

from typing import List, Dict, Any
import re

# (강의 안내) yfinance는 외부 네트워크 환경에서 동작. 인터넷 불가 환경에선 모킹이 필요할 수 있음.


def _normalize_symbol(s: str) -> str:
    """
    6자리 숫자면 한국거래소(.KS) 보정.
    예:
      '005930' → '005930.KS'
      'AAPL'   → 'AAPL' (그대로)
    """

    if re.fullmatch(r"\d{6}", s):
        return f"{s}.KS"
    else:
        return s



def get_quotes(symbols: List[str], timeout: int = 20) -> List[Dict[str, Any]]:
    """
    yfinance로 심볼별 시세를 조회해 리스트로 반환합니다.
    반환 예:
      [{"symbol":"AAPL","price":123.45,"currency":"USD"},
       {"symbol":"005930.KS","price":...,"currency":"KRW"}]
    실패시 해당 심볼은 {"symbol":sym, "error":"..."} 형태로 표기.
    """

    from yfinance import Ticker
    
    out: List[Dict[str, Any]] = []
    for raw in (symbols or []):
        sym = _normalize_symbol(str(raw).strip())
        try:
            t = Ticker(sym)
            fi = getattr(t, "fast_info", None)

            price = None
            currency = None
            if fi is not None:
                # 객체/딕트 양쪽 케이스를 모두 처리
                price = getattr(fi, "last_price", None)
                if price is None:
                    try:
                        price = fi.get("last_price")
                    except Exception:
                        pass

                currency = getattr(fi, "currency", None)
                if currency is None:
                    try:
                        currency = fi.get("currency")
                    except Exception:
                        pass

            if price is None or currency is None:
                out.append({"symbol": sym, "error": "가격/통화 조회 실패"})
                continue

            # 가격 형식 확인
            try:
                price = float(price)
            except Exception:
                out.append({"symbol": sym, "error": f"가격 형식 오류: {price!r}"})
                continue

            out.append({"symbol": sym, "price": price, "currency": str(currency)})
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})

    return out

