# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict, Any, List
import numpy as np

from student.common.schemas import Day2Plan
from .embeddings import Embeddings
from .store import FaissStore

def _idx_paths(index_dir: str):
    return (
        os.path.join(index_dir, "faiss.index"),
        os.path.join(index_dir, "docs.jsonl"),
    )

def _load_store(plan: Day2Plan, emb: Embeddings) -> FaissStore:
    index_path, docs_path = _idx_paths(plan.index_dir)
    if not (os.path.exists(index_path) and os.path.exists(docs_path)):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다. 먼저 ingest를 실행하세요: {plan.index_dir}")
    store = FaissStore.load(index_path, docs_path)
    # 차원 체크
    test_dim = emb.encode(["__dim_check__"]).shape[1]
    if store.dim != test_dim:
        raise ValueError(f"임베딩 차원이 인덱스와 다릅니다. (index={store.dim}, embedder={test_dim})")
    return store

def _gate(contexts: List[Dict[str, Any]], plan: Day2Plan) -> Dict[str, Any]:
    if not contexts:
        return {"status":"insufficient","top_score":0.0,"mean_topk":0.0}
    top_score = float(contexts[0]["score"])
    mean_topk = float(np.mean([c["score"] for c in contexts[:plan.top_k]]))
    if top_score >= plan.min_score and mean_topk >= plan.min_mean_topk:
        return {"status":"enough","top_score":top_score,"mean_topk":mean_topk}
    return {"status":"insufficient","top_score":top_score,"mean_topk":mean_topk}

def _draft_answer(query: str, contexts: List[Dict[str, Any]], plan: Day2Plan) -> str:
    buf, budget = [], plan.max_context
    for c in contexts:
        t = c["chunk"].strip().replace("\n", " ")
        if len(t) > budget:
            t = t[:budget] + "..."
        buf.append(f"- {t}")
        budget -= len(t)
        if budget <= 0:
            break
    return f"질의: {query}\n\n핵심 근거 요약:\n" + "\n".join(buf) if buf else ""

class Day2Agent:
    def __init__(self, plan_defaults: Day2Plan = Day2Plan()):
        self.plan_defaults = plan_defaults

    def handle(self, query: str, plan: Day2Plan = None) -> Dict[str, Any]:
        plan = plan or self.plan_defaults
        emb = Embeddings(model=plan.embedding_model)

        store = _load_store(plan, emb)
        qv = emb.encode([query])[0]
        contexts = store.search(qv, top_k=plan.top_k)

        gate = _gate(contexts, plan)
        payload: Dict[str, Any] = {
            "type": "rag_answer",
            "query": query,
            "plan": plan.__dict__,
            "contexts": contexts,
            "gating": gate,
            "answer": "",
            "notice": "web_merge_in_day4_only",
        }
        if plan.force_rag_only or (gate["status"] == "enough" and plan.return_draft_when_enough):
            payload["answer"] = _draft_answer(query, contexts, plan)
        return payload
