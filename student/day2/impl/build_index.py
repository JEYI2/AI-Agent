# -*- coding: utf-8 -*-
"""
Day2 인덱싱 엔트리포인트
- 목표: 코퍼스 생성 → 임베딩 → FAISS 저장 + docs.jsonl 저장
"""

import os, argparse, numpy as np
from typing import List

from student.day2.impl.ingest import build_corpus, save_docs_jsonl
from student.day2.impl.embeddings import Embeddings
from student.day2.impl.store import FaissStore  # 제공됨


def build_index(paths: List[str], index_dir: str, model: str | None = None, batch_size: int = 128):
    """
    절차:
      1) corpus = build_corpus(paths)
         - [{"id":..., "text":..., "meta":{...}}, ...]
      2) texts = [item["text"] for item in corpus]
      3) emb = Embeddings(model=model, batch_size=batch_size)
         vecs = emb.encode(texts)  # (N, D) L2 정규화된 np.ndarray
      4) index_path = os.path.join(index_dir, "faiss.index")
         docs_path  = os.path.join(index_dir, "docs.jsonl")
      5) store = FaissStore(dim=vecs.shape[1], index_path=index_path, docs_path=docs_path)
         store.add(vecs, corpus); store.save()
      6) save_docs_jsonl(corpus, docs_path)
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-I-01] 구현 지침
    #  - corpus = build_corpus(paths)
    #  - texts = [...]
    #  - emb = Embeddings(model, batch_size)
    #  - vecs = emb.encode(texts)
    #  - os.makedirs(index_dir, exist_ok=True)
    #  - store = FaissStore(...); store.add(...); store.save()
    #  - save_docs_jsonl(corpus, docs_path)
    # ----------------------------------------------------------------------------
    corpus = build_corpus(paths)
    texts = [item["text"] for item in corpus]
    emb = Embeddings(model=model, batch_size=batch_size)
    vecs = emb.encode(texts)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss.index")
    docs_path = os.path.join(index_dir, "docs.jsonl")
    store = FaissStore(dim=vecs.shape[1], index_path=index_path, docs_path=docs_path)
    store.add(vecs, corpus)
    store.save()
    save_docs_jsonl(corpus, docs_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--index_dir", default="indices/day2")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    # ----------------------------------------------------------------------------
    # TODO[DAY2-I-02] 구현 지침
    #  - os.makedirs(args.index_dir, exist_ok=True)
    #  - build_index(args.paths, args.index_dir, args.model, args.batch_size)
    # ----------------------------------------------------------------------------
    os.makedirs(args.index_dir, exist_ok=True)
    build_index(args.paths, args.index_dir, args.model, args.batch_size)
