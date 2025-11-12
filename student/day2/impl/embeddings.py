# -*- coding: utf-8 -*-
"""
OpenAI 임베딩 래퍼
- 요구사항: 배치 인코딩, 재시도(backoff), L2 정규화
"""

import os, time
from typing import List
import numpy as np
from httpx import ReadTimeout  # 선택: 재시도 구분용
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
# 현재 작업 디렉터리(CWD) 기준으로 .env를 찾아서 강제 로드
load_dotenv(find_dotenv(usecwd=True), override=True)



class Embeddings:
    def __init__(self, model: str | None = None, batch_size: int = 128, max_retries: int = 4):
        """
        - self.model 기본값: "text-embedding-3-small" 권장
        - self.batch_size, self.max_retries 저장
        - OpenAI 클라이언트 생성 (키는 환경변수 OPENAI_API_KEY)
        """
        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-01] 구현 지침
        #  - self.model = model or "text-embedding-3-small"
        #  - self.batch_size = batch_size; self.max_retries = max_retries
        #  - key = os.getenv("OPENAI_API_KEY"); self.client = OpenAI(api_key=key)
        # ----------------------------------------------------------------------------
        self.model = model or "text-embedding-3-small"
        self.batch_size = batch_size
        self.max_retries = max_retries
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def _embed_once(self, text: str) -> np.ndarray:
        """
        단일 텍스트 임베딩 호출 → np.ndarray(float32) + L2 정규화
        - 예외 발생 시 상위 encode에서 재시도하도록 예외를 그대로 올려보냄
        """
        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-02] 구현 지침
        #  - resp = self.client.embeddings.create(model=self.model, input=text)
        #  - vec = np.array(resp.data[0].embedding, dtype="float32")
        #  - norm = np.linalg.norm(vec) + 1e-12; vec = vec / norm
        #  - return vec
        # ----------------------------------------------------------------------------
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(vec) + 1e-12
        vec = vec / norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        배치 인코딩 + 재시도(backoff). 최종 shape = (N, D)
        - 비어 있으면 (0, D) 반환. D는 1536 등 모델 차원 (미정이면 1536 가정 가능)
        """
        # ----------------------------------------------------------------------------
        # TODO[DAY2-E-03] 구현 지침
        #  - if not texts: return np.zeros((0, 1536), dtype="float32")
        #  - out = []
        #  - for start in range(0, len(texts), self.batch_size):
        #       batch = texts[start:start+self.batch_size]
        #       for each in batch:
        #          for attempt in range(self.max_retries):
        #             try:
        #                out.append(self._embed_once(each)); break
        #             except Exception as e:
        #                time.sleep(0.5 * (2 ** attempt))
        #                if attempt == self.max_retries - 1: raise
        #  - return np.vstack(out)
        # ----------------------------------------------------------------------------
        if not texts:
            return np.zeros((0, 1536), dtype="float32")
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for each in batch:
                for attempt in range(self.max_retries):
                    try:
                        out.append(self._embed_once(each))
                        break
                    except Exception as e:
                        time.sleep(0.5 * (2**attempt))
                        if attempt == self.max_retries - 1:
                            raise
        return np.vstack(out)
