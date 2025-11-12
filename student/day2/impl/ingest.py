# -*- coding: utf-8 -*-
"""
인덱싱 입력 데이터 로딩/정제/청크
"""

import re, json
from typing import List, Dict, Any
from pathlib import Path

def read_text_file(path: str) -> str:
    """
    안전한 텍스트 로드(utf-8, errors='ignore')
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-01] 구현 지침
    #  - with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
    # ----------------------------------------------------------------------------
    # 정답 구현:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    """
    pypdf 로 PDF 모든 페이지 텍스트 추출
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-02] 구현 지침
    #  - from pypdf import PdfReader
    #  - reader = PdfReader(path); texts=[]
    #  - for page in reader.pages: texts.append(page.extract_text() or "")
    #  - return "\n".join(texts)
    # ----------------------------------------------------------------------------
    # 정답 구현:
    from pypdf import PdfReader  # type: ignore
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def clean_text(s: str) -> str:
    """
    과도한 공백/개행/컨트롤 문자 정제
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-03] 구현 지침
    #  - s = s or ""
    #  - s = re.sub(r"\r", "\n", s)
    #  - s = re.sub(r"[ \t]+", " ", s)
    #  - s = re.sub(r"\n{3,}", "\n\n", s)
    #  - return s.strip()
    # ----------------------------------------------------------------------------
    # 정답 구현:
    s = s or ""
    s = re.sub(r"\r", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    슬라이딩 윈도우로 청크 분할.
    - 길이가 chunk_size 이하이면 그대로 1청크
    - 그 외에는 overlap 적용하여 분할
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-04] 구현 지침
    #  - if len(text) <= chunk_size: return [text]
    #  - chunks=[]; start=0
    #  - while start < len(text):
    #       end = min(len(text), start+chunk_size)
    #       chunks.append(text[start:end])
    #       start += (chunk_size - chunk_overlap)
    #  - return chunks
    # ----------------------------------------------------------------------------
    # 정답 구현:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks


def load_documents(paths_or_dir: List[str]) -> List[Dict[str, Any]]:
    """
    입력 경로(디렉토리/파일)에서 txt/md/pdf 수집 → [{"path":..., "text":...}, ...]
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-05] 구현 지침
    #  - files=[]
    #  - for p in paths_or_dir:
    #       pp=Path(p)
    #       if pp.is_dir():
    #          for ext in ("*.txt","*.md","*.pdf"): files.extend([str(x) for x in pp.rglob(ext)])
    #       else: files.append(str(pp))
    #  - docs=[]
    #  - for fp in files:
    #       ext = fp.lower().split(".")[-1]
    #       if ext in ("txt","md"): raw = read_text_file(fp)
    #       elif ext=="pdf": raw = read_pdf_file(fp)
    #       else: continue
    #       txt = clean_text(raw); docs.append({"path":fp,"text":txt})
    #  - return docs
    # ----------------------------------------------------------------------------
    # 정답 구현:
    files: List[str] = []
    for p in paths_or_dir:
        pp = Path(p)
        if pp.is_dir():
            for ext in ("*.txt", "*.md", "*.pdf"):
                files.extend([str(x) for x in pp.rglob(ext)])
        else:
            files.append(str(pp))

    docs: List[Dict[str, Any]] = []
    for fp in files:
        ext = fp.lower().split(".")[-1]
        if ext in ("txt", "md"):
            raw = read_text_file(fp)
        elif ext == "pdf":
            raw = read_pdf_file(fp)
        else:
            continue
        txt = clean_text(raw)
        docs.append({"path": fp, "text": txt})
    return docs


def build_corpus(paths_or_dir: List[str]) -> List[Dict[str, Any]]:
    """
    문서를 청크 단위로 나눠 코퍼스 생성
    반환 예: [{"id":"<path>::chunk_0000","text":"...", "meta":{"path":..., "chunk":0}}, ...]
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-06] 구현 지침
    #  - docs = load_documents(paths_or_dir)
    #  - corpus=[]
    #  - for d in docs:
    #       chunks = chunk_text(d["text"])
    #       for i,ch in enumerate(chunks):
    #           cid = f"{d['path']}::chunk_{i:04d}"
    #           corpus.append({"id":cid,"text":ch,"meta":{"path":d["path"],"chunk":i}})
    #  - return corpus
    # ----------------------------------------------------------------------------
    # 정답 구현:
    docs = load_documents(paths_or_dir)
    corpus: List[Dict[str, Any]] = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, ch in enumerate(chunks):
            cid = f"{d['path']}::chunk_{i:04d}"
            corpus.append({"id": cid, "text": ch, "meta": {"path": d["path"], "chunk": i}})
    return corpus


def save_docs_jsonl(items: List[Dict[str, Any]], out_path: str):
    """
    문서 메타를 JSONL로 저장(ensure_ascii=False)
    """
    # ----------------------------------------------------------------------------
    # TODO[DAY2-G-07] 구현 지침
    #  - with open(out_path,"w",encoding="utf-8") as f:
    #       for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # ----------------------------------------------------------------------------
    # 정답 구현:
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")