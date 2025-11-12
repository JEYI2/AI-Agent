# Day3 실습 TODO 체크리스트

## kt_aivle/sub_agents/day3/agent.py
- [ ] DAY3-A-01: LiteLlm 모델 선택 및 MODEL 변수 설정
- [ ] DAY3-A-02: `_handle` 구현 (Day3Plan/Day3Agent 호출)
- [ ] DAY3-A-03: `before_model_callback` 구현 (query 추출 → 렌더/저장/envelope → 응답)
- [ ] DAY3-A-04: Agent 메타데이터(name/description/instruction) 다듬기

## kt_aivle/sub_agents/day3/impl/agent.py
- [ ] DAY3-I-01: `_set_source_topk` 구현 (plan 값을 fetchers 상수에 동기화)
- [ ] DAY3-I-02: `__init__`에서 환경변수 로딩/보관 (예: TAVILY_API_KEY)
- [ ] DAY3-I-03: `handle` 파이프라인 구현 (fetch → normalize → rank → payload)

## kt_aivle/sub_agents/day3/impl/fetchers.py
- [ ] DAY3-F-01: `fetch_nipa` 구현 (site:nipa.kr + include_domains=["nipa.kr"])
- [ ] DAY3-F-02: `fetch_bizinfo` 구현 (site:bizinfo.go.kr + include_domains=["bizinfo.go.kr"])
- [ ] DAY3-F-03: `fetch_web` 구현 (보조 키워드, 도메인 제한 없음)
- [ ] DAY3-F-04: `fetch_all` 구현 (세 소스 호출 결과 병합)
