# 오케스트레이터 TODO 체크리스트

## student/prompts.py
- [ ] ROOT-P-01: ORCHESTRATOR_DESC 작성
  - 역할/라우팅 기준(정부=Day3, 문서=Day2, 웹/뉴스/주가=Day1), 출력 가이드(한글 요약+표/목록)
- [ ] ROOT-P-02: ORCHESTRATOR_PROMPT 작성
  - 도구 선택 규칙, Day2 신뢰도 낮을 때 Day1 보강 규칙, 마크다운 출력 규칙

## student/agent.py
- [ ] ROOT-A-01: MODEL = LiteLlm(model="...") 설정
- [ ] ROOT-A-02: root_agent 설정 (name/description/instruction/tools)
- [ ] ROOT-A-03: AgentTool 등록( Day1, Day2, Day3 )
