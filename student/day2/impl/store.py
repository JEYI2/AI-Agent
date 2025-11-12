# -*- coding: utf-8 -*-
import os, json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

class FaissStore:
    def __init__(self, dim: int, index_path: str, docs_path: str):
        self.dim = dim
        self.index_path = index_path
        self.docs_path = docs_path
        self.index = faiss.IndexFlatIP(dim)  # 코사인=내적 (임베딩 정규화 가정)
        self.docs: List[Dict[str, Any]] = []

    # ---------- Build ----------
    def add(self, embeddings: np.ndarray, items: List[Dict[str, Any]]):
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype("float32"))
        self.docs.extend(items)

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            for it in self.docs:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---------- Load ----------
    @classmethod
    def load(cls, index_path: str, docs_path: str):
        index = faiss.read_index(index_path)
        dim = index.d
        store = cls(dim, index_path, docs_path)
        store.index = index
        store.docs = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                store.docs.append(json.loads(line))
        return store

    # ---------- Search ----------
    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        D, I = self.index.search(query_vec.astype("float32"), top_k)
        out = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1:
                continue
            doc = self.docs[idx]
            out.append({
                "doc_id": doc["id"],
                "chunk": doc["text"],
                "score": float(score),  # 내적값(정규화 가정 → 코사인)
                "meta": doc.get("meta", {})
            })
        return out
