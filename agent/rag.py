"""
RAG 知识库 —— 文档摄入、分块、向量存储、语义检索

支持 PDF / Markdown / TXT
当前用 n-gram 哈希嵌入（可一键替换为 OpenAI/DeepSeek Embedding API）
"""
import json
import math
import os
import re
from typing import List, Dict, Optional

# 存储路径
KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
CHUNKS_FILE = os.path.join(KB_DIR, "chunks.json")
os.makedirs(KB_DIR, exist_ok=True)


class DocumentStore:
    """文档知识库：分块 + 向量索引 + 检索"""

    def __init__(self, embedding_dim: int = 256, chunk_size: int = 500, chunk_overlap: int = 80):
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Dict] = []  # [{id, text, source, embedding}]
        self._load()

    # ========== 文档解析 ==========

    def parse_pdf(self, filepath: str) -> str:
        """解析 PDF 文本"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
        except ImportError:
            return f"[错误] 需要 pypdf 库解析 PDF。运行: uv pip install pypdf"
        except Exception as e:
            return f"[PDF 解析错误] {e}"

    def parse_text(self, text: str) -> str:
        """直接接收文本"""
        return text

    def parse_markdown(self, text: str) -> str:
        """去除 Markdown 标记，保留纯文本"""
        # 去掉代码块
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        # 去掉行内代码
        text = re.sub(r'`[^`]+`', '', text)
        # 去掉标题标记但保留文字
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # 去掉链接，保留文字
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 去掉图片
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 去掉加粗斜体标记
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        return text

    # ========== 文本分块 ==========

    def chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """将文本切成有重叠的块"""
        chunks = []
        text = text.strip()
        if not text:
            return chunks

        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尽量在句子边界切分
            if end < len(text):
                # 找最近的句号/换行作为切分点
                boundary = max(
                    text.rfind("。", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) > 20:  # 跳过太短的块
                chunks.append({
                    "id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "embedding": self._embed(chunk_text),
                })
                chunk_id += 1

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    # ========== 向量嵌入 ==========

    def _embed(self, text: str) -> List[float]:
        """n-gram 哈希嵌入（占位——替换为 Embedding API 即升级）"""
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

    # ========== 文档摄入 ==========

    def ingest_file(self, filepath: str) -> int:
        """摄入一个文件，返回新增块数"""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = self.parse_pdf(filepath)
        elif ext in [".md", ".markdown"]:
            with open(filepath, "r", encoding="utf-8") as f:
                text = self.parse_markdown(f.read())
        elif ext in [".txt", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"]:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            return 0

        return self.ingest_text(text, source=filename)

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """摄入纯文本"""
        new_chunks = self.chunk_text(text, source=source)
        self.chunks.extend(new_chunks)
        self._save()
        return len(new_chunks)

    # ========== 检索 ==========

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义检索最相关的文档块"""
        if not self.chunks:
            return []

        query_vec = self._embed(query)
        scores = [(i, self._cosine_sim(query_vec, self.chunks[i]["embedding"]))
                  for i in range(len(self.chunks))]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scores[:top_k]:
            if score > 0.05:
                results.append({
                    "text": self.chunks[i]["text"],
                    "source": self.chunks[i]["source"],
                    "score": round(score, 3),
                })
        return results

    def search_formatted(self, query: str, top_k: int = 5) -> str:
        """检索并格式化为上下文文本"""
        results = self.search(query, top_k)
        if not results:
            return ""

        lines = ["## 📚 知识库相关内容\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**片段 {i}** (来源: {r['source']}, 相关度: {r['score']})")
            lines.append(r["text"][:800])
            lines.append("")
        return "\n".join(lines)

    # ========== 管理 ==========

    def list_sources(self) -> List[Dict]:
        """列出所有文档来源及块数"""
        from collections import Counter
        counts = Counter(c["source"] for c in self.chunks)
        return [{"source": s, "chunks": n} for s, n in counts.most_common()]

    def delete_source(self, source: str) -> int:
        """删除指定来源的所有块"""
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["source"] != source]
        self._save()
        return before - len(self.chunks)

    def clear(self):
        self.chunks = []
        self._save()

    def stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "total_sources": len(set(c["source"] for c in self.chunks)),
            "sources": self.list_sources(),
        }

    # ========== 持久化 ==========

    def _save(self):
        data = {
            "chunk_size": self.chunk_size,
            "embedding_dim": self.embedding_dim,
            "chunks": self.chunks,
        }
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(CHUNKS_FILE):
            try:
                with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = data.get("chunks", [])
                self.chunk_size = data.get("chunk_size", self.chunk_size)
                self.embedding_dim = data.get("embedding_dim", self.embedding_dim)
            except Exception:
                self.chunks = []


# 全局单例
kb = DocumentStore()
