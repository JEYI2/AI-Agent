## Day2

### /student/day2/agent.py
- [ ] DAY2-A-01: LiteLlm 모델 선택 및 MODEL 설정
- [ ] DAY2-A-02: `_handle` 구현 (Day2Plan/Day2Agent 호출 → payload)
- [ ] DAY2-A-03: `before_model_callback` 구현 (렌더/저장/envelope)

### /student/day2/impl/build_index.py
- [ ] DAY2-I-01: `build_index` 구현 (코퍼스→임베딩→FAISS 저장→docs.jsonl)
- [ ] DAY2-I-02: `__main__` 엔트리포인트 구현

### /student/day2/impl/embeddings.py
- [ ] DAY2-E-01: `__init__` 구성 (모델/배치/재시도/클라이언트)
- [ ] DAY2-E-02: `_embed_once` 구현 (OpenAI 호출 + L2정규화)
- [ ] DAY2-E-03: `encode` 구현 (배치/재시도/스택)

### /student/day2/impl/ingest.py
- [ ] DAY2-G-01: `read_text_file` 구현
- [ ] DAY2-G-02: `read_pdf_file` 구현
- [ ] DAY2-G-03: `clean_text` 구현
- [ ] DAY2-G-04: `chunk_text` 구현
- [ ] DAY2-G-05: `load_documents` 구현
- [ ] DAY2-G-06: `build_corpus` 구현
- [ ] DAY2-G-07: `save_docs_jsonl` 구현