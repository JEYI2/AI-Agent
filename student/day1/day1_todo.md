
## Day1

### /student/day1/agent.py
- [ ] DAY1-A-01: LiteLlm 모델 선택 및 MODEL 변수 설정
- [ ] DAY1-A-02: `_extract_tickers_from_query` 구현 (영문 1~5자, 숫자 6자리 → 중복 제거)
- [ ] DAY1-A-03: `_normalize_kr_tickers` 구현 (6자리면 .KS)
- [ ] DAY1-A-04: `_handle` 구현 (API 키 획득 → Day1Plan → Day1Agent.handle)
- [ ] DAY1-A-05: `before_model_callback` 구현 (query 추출 → 렌더/저장/envelope)
- [ ] DAY1-A-06: Agent 메타데이터 점검/수정

### /student/day1/impl/agent.py
- [ ] DAY1-I-01: 요약 LLM 인스턴스(_SUM) 준비
- [ ] DAY1-I-02: `_summarize` 구현 (LLM 호출 → 실패시 "")
- [ ] DAY1-I-03: `__init__` 필드 초기화
- [ ] DAY1-I-04: `handle` 병렬 파이프라인 구현 (웹/주가/기업개요 → merge)

### /student/day1/impl/finance_client.py
- [ ] DAY1-F-01: `_normalize_symbol` 구현
- [ ] DAY1-F-02: `get_quotes` 구현 (yfinance.fast_info → price/currency)

### /student/day1/impl/merge.py
- [ ] DAY1-M-01: `_top_results` 구현
- [ ] DAY1-M-02: `merge_day1_payload` 구현 (표준 스키마 매핑)