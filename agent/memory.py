"""
轻量向量记忆 —— 纯 Python 实现
零外部依赖：添加记忆、语义检索、短期窗口
"""
import math
import json
import os
from typing import List, Dict

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)


class AgentMemory:
    """Agent 记忆系统：短期滑动窗口 + 长期向量记忆（纯 Python）"""

    def __init__(self, embedding_dim: int = 256, short_term_size: int = 10):
        self.short_term_size = short_term_size
        self.short_term: List[Dict] = []
        self.long_term_vectors: List[List[float]] = []
        self.long_term_texts: List[str] = []
        self.embedding_dim = embedding_dim
        self._load()

    # ---- 短期记忆（滑动窗口）----

    def add_message(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.short_term_size * 2:
            self.short_term = self.short_term[-self.short_term_size * 2:]

    def get_recent(self, n: int = None) -> List[Dict]:
        n = n or self.short_term_size
        return self.short_term[-n:]

    # ---- 长期记忆（向量检索）----

    def _simple_embed(self, text: str) -> List[float]:
        """字符级 n-gram 哈希嵌入"""
        n = 3
        vec = [0.0] * self.embedding_dim
        for i in range(len(text) - n + 1):
            h = hash(text[i:i+n]) % self.embedding_dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def save_memory(self, text: str, metadata: Dict = None):
        vec = self._simple_embed(text)
        self.long_term_vectors.append(vec)
        self.long_term_texts.append(text)
        self._save()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.long_term_texts:
            return []
        query_vec = self._simple_embed(query)
        scores = [(i, self._cosine_sim(query_vec, self.long_term_vectors[i]))
                  for i in range(len(self.long_term_texts))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.long_term_texts[i], "score": round(s, 3)}
            for i, s in scores[:top_k] if s > 0.05
        ]

    def _save(self):
        path = os.path.join(MEMORY_DIR, "long_term.json")
        data = {"texts": self.long_term_texts, "vectors": self.long_term_vectors}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        path = os.path.join(MEMORY_DIR, "long_term.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.long_term_texts = data.get("texts", [])
            self.long_term_vectors = data.get("vectors", [])

    def get_summary(self) -> str:
        return f"短期对话: {len(self.short_term)} 条 | 长期记忆: {len(self.long_term_texts)} 条"


# 全局单例
memory = AgentMemory()
