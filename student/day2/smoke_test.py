# -*- coding: utf-8 -*-
"""
Day2 RAG 스모크 테스트
"""

import os, sys, json, time, argparse
from pathlib import Path

# ───────── 0) 루트 탐색 + sys.path + .env ─────────
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "apps").exists():
            return p
    return start

ROOT = _find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"
def _manual_load_env(p: Path):
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except Exception:
    _manual_load_env(ENV_PATH)

# ───────── 1) 배포 모듈 임포트 ─────────
def _import_all():
    from student.day2.impl.rag import Day2Agent
    from student.common.schemas import Day2Plan
    from student.day2.impl.store import FaissStore
    from student.day2.impl.embeddings import Embeddings
    from student.day2.impl.build_index import build_index
    return Day2Agent, Day2Plan, FaissStore, Embeddings, build_index

Day2Agent, Day2Plan, FaissStore, Embeddings, build_index = _import_all()

# ───────── 2) 유틸 ─────────
def _idx_paths(index_dir: str):
    d = Path(index_dir)
    return d / "faiss.index", d / "docs.jsonl"

def _file_info(p: Path) -> str:
    try:
        return f"{p} ({p.stat().st_size:,} bytes)"
    except Exception:
        return f"{p} (size: ?)"""

def _read_docs_head(docs_path: Path, n: int = 5):
    lines = docs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    empty_cnt = 0
    for i, ln in enumerate(lines[:n]):
        try:
            obj = json.loads(ln)
            text = (obj.get("text") or "").strip()
            if not text:
                empty_cnt += 1
            out.append({"i": i, "id": obj.get("id"), "path": obj.get("path"), "len": len(text)})
        except Exception:
            out.append({"i": i, "parse_error": True})
    return len(lines), empty_cnt, out

def _estimate_store_size(store) -> str:
    """FaissStore 구현마다 다른 경우를 모두 수용해 사이즈 추정."""
    # 1) __len__
    try:
        return str(len(store))
    except Exception:
        pass
    # 2) size() 메서드
    try:
        s = store.size() if callable(getattr(store, "size", None)) else None
        if s is not None:
            return str(int(s))
    except Exception:
        pass
    # 3) ntotal 속성 직접/내부 index 통해
    try:
        n = getattr(store, "ntotal", None)
        if n is not None:
            return str(int(n))
    except Exception:
        pass
    try:
        idx = getattr(store, "index", None)
        if idx is not None:
            n = getattr(idx, "ntotal", None)
            if n is not None:
                return str(int(n))
    except Exception:
        pass
    # 4) 알 수 없음
    return "?"

# ───────── 3) 인덱스/FAISS/임베딩 진단 ─────────
def _diagnose(index_dir: str, paths: str, model: str, autobuild: bool, batch_size: int):
    idx_path, docs_path = _idx_paths(index_dir)
    ok = True
    if not idx_path.exists():
        print("[WARN] faiss.index 없음 →", idx_path)
        ok = False
    if not docs_path.exists():
        print("[WARN] docs.jsonl 없음  →", docs_path)
        ok = False
    if not ok:
        if not autobuild:
            print("  해결: 인덱스 생성")
            print(f"  uv run python -m student.day2.impl.build_index --paths {paths} --index_dir {index_dir} --model {model} --batch_size {batch_size}")
            return None, None
        print("[INFO] --autobuild 지정 → 인덱스 생성 시작")
        build_index(paths, index_dir, model, batch_size)

    # 파일 정보
    print("[INFO] 인덱스 파일:", _file_info(idx_path))
    print("[INFO] 문서 파일  :", _file_info(docs_path))
    try:
        total, empty_cnt, head = _read_docs_head(docs_path, n=5)
        print(f"[OK] docs.jsonl 라인수={total}, (빈 텍스트 {empty_cnt})")
        for r in head:
            print("   ", r)
    except Exception as e:
        print("[WARN] docs.jsonl 파싱 이슈:", e)

    # FAISS 로드
    try:
        store = FaissStore.load(str(idx_path), str(docs_path))
        print("[OK] FAISS 로드 성공")
    except Exception as e:
        print("[FAIL] FAISS 로드 실패:", e)
        return None, None

    # 임베딩 초기화 + 차원 확인
    try:
        emb = Embeddings(model=model, batch_size=4)
        dim = emb.encode(["__dim_check__"]).shape[1]
        print(f"[OK] 임베딩 초기화: model={model}, dim={dim}")
    except Exception as e:
        print("[FAIL] 임베딩 초기화 실패:", e)
        return store, None

    # store.dim 출력(없으면 넘어감)
    try:
        print(f"[INFO] store.dim = {getattr(store, 'dim')}")
    except Exception:
        pass

    # 사이즈 추정
    size_hint = _estimate_store_size(store)
    print(f"[INFO] 인덱스 사이즈(추정) = {size_hint}")

    # 차원 불일치 검사 (가능할 때만)
    try:
        sdim = getattr(store, "dim")
        if sdim and sdim != dim:
            print(f"[FAIL] 차원 불일치: index_dim={sdim}, embed_dim={dim}")
            print("  해결: 인덱스를 동일 모델로 재생성하거나, 스모크의 --model 값을 맞추세요.")
    except Exception:
        # store.dim이 없으면 스킵 (구현체별 차이)
        pass

    return store, dim

# ───────── 4) 검색 + Agent.handle ─────────
def _run_search_and_agent(query: str, index_dir: str, model: str, top_k: int):
    from student.day2.impl.rag import Day2Agent
    from student.common.schemas import Day2Plan
    from student.day2.impl.embeddings import Embeddings
    from student.day2.impl.store import FaissStore

    # 임베딩/스토어 준비
    emb = Embeddings(model=model, batch_size=4)
    qv = emb.encode([query])[0]
    store = FaissStore.load(str(Path(index_dir)/"faiss.index"), str(Path(index_dir)/"docs.jsonl"))

    # 로우 검색
    try:
        hits = store.search(qv, top_k=top_k)
        print(f"[OK] 로우 검색 hit={len(hits)} (상위 3개 미리보기)")
        for i, h in enumerate(hits[:3], 1):
            score = float(h.get("score", 0.0))
            path = str(h.get("path") or h.get("id") or "")
            text = (h.get("text") or h.get("chunk") or "").replace("\n"," ").strip()[:160]
            print(f"   {i:>2}. {score:.3f} | {path} | {text}")
    except Exception as e:
        print("[FAIL] 로우 검색 실패:", e)

    # Agent.handle
    plan = Day2Plan(index_dir=index_dir, embedding_model=model, top_k=top_k,
                    force_rag_only=True, return_draft_when_enough=True)
    agent = Day2Agent(plan_defaults=plan)
    out = agent.handle(query)
    print(f"[OK] Agent.handle 완료 | gating={out.get('gating')} | ctx={len(out.get('contexts', []))}")
    if out.get("answer"):
        print("\n[OK] 초안 요약(일부):")
        print(out["answer"][:400] + ("..." if len(out["answer"]) > 400 else ""))
    return out

# ───────── 5) 리포트 저장 ─────────
def _save_report(query: str, index_dir: str, model: str, payload: dict):
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{ts}__day2_smoke__{query.replace(' ','-')}.json"
    path.write_text(json.dumps({
        "query": query,
        "index_dir": index_dir,
        "model": model,
        "result": payload,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 진단 리포트 저장: {path}")

# ───────── Entry ─────────
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Day2 RAG 스모크/디버그(meta 의존 제거판)")
    p.add_argument("--index_dir", default="indices/day2")
    p.add_argument("--paths", default="data/raw")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--query", default="헬스케어 규제")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--autobuild", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    print("[INFO] ROOT:", ROOT)
    print("[INFO] .env :", ENV_PATH, "| OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
    print("[INFO] index:", args.index_dir, "| paths:", args.paths, "| model:", args.model)

    store, dim = _diagnose(args.index_dir, args.paths, args.model, args.autobuild, args.batch_size)
    if store is None:
        sys.exit(2)

    out = _run_search_and_agent(args.query, args.index_dir, args.model, args.top_k)
    _save_report(args.query, args.index_dir, args.model, out)
    print("\n[DONE] Day2 스모크 테스트 완료")

if __name__ == "__main__":
    main()