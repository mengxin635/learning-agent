"""
学习型记忆系统 —— 纯 Python 实现（零外部依赖）

架构：
  WorkingMemory   — 短期对话窗口
  KnowledgeGraph  — 知识图谱：主题掌握度 + 艾宾浩斯间隔复习
  UserProfile     — 用户模型：偏好、薄弱点、学习路径
  SemanticIndex   — TF-IDF 语义检索引擎

设计原则：面试可讲、实用性优先、零外部 API 依赖
"""
import re
import json
import math
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)


# ============================================================
# TF-IDF 语义向量引擎（零依赖，效果远超 n-gram hash）
# ============================================================

class SemanticIndex:
    """TF-IDF 向量化 + 余弦相似度检索"""

    def __init__(self):
        self.documents: List[str] = []
        self.vectors: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self._dirty = True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词：按 2-gram 切分 + 单字 + 英文单词"""
        tokens = []
        # 英文单词
        tokens.extend(re.findall(r'[a-zA-Z]+', text.lower()))
        # 中文字符
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        for chunk in chinese:
            # 2-gram 切分（捕获词级模式）
            for i in range(len(chunk) - 1):
                tokens.append(chunk[i:i+2])
            # 单字（捕获精确匹配）
            tokens.extend(list(chunk))
        return tokens

    def add(self, text: str) -> int:
        """添加文档，返回索引"""
        self.documents.append(text)
        self._dirty = True
        return len(self.documents) - 1

    def _build_index(self):
        """构建 TF-IDF 向量"""
        if not self._dirty:
            return

        # Step 1: 词频统计
        doc_tokens = [self._tokenize(d) for d in self.documents]
        doc_tf = []
        for tokens in doc_tokens:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # 归一化
            total = len(tokens) or 1
            doc_tf.append({t: c/total for t, c in tf.items()})

        # Step 2: IDF
        N = len(self.documents)
        all_terms = set()
        for tf in doc_tf:
            all_terms.update(tf.keys())
        self.idf = {}
        for term in all_terms:
            df = sum(1 for tf in doc_tf if term in tf)
            self.idf[term] = math.log((N + 1) / (df + 1)) + 1

        # Step 3: TF-IDF 向量
        self.vectors = []
        for tf in doc_tf:
            vec = {t: tf[t] * self.idf[t] for t in tf}
            # L2 归一化
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1
            self.vectors.append({t: v/norm for t, v in vec.items()})

        self._dirty = False

    def _vectorize_query(self, text: str) -> Dict[str, float]:
        """向量化查询文本"""
        tokens = self._tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        vec = {}
        for t, c in tf.items():
            if t in self.idf:
                vec[t] = (c/total) * self.idf[t]
        # L2
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1
        return {t: v/norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相似的文档"""
        if not self.documents:
            return []
        self._build_index()
        q_vec = self._vectorize_query(query)

        # 余弦相似度 = 点积（向量已 L2 归一化）
        scores = []
        for i, d_vec in enumerate(self.vectors):
            dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in set(q_vec) | set(d_vec))
            scores.append((i, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"text": self.documents[i], "score": round(s, 3)}
            for i, s in scores[:top_k] if s > 0.05
        ]


# ============================================================
# 知识图谱：主题掌握度 + 艾宾浩斯间隔重复
# ============================================================

# 艾宾浩斯遗忘曲线复习间隔（天）
EBBINGHAUS_INTERVALS = [1, 2, 4, 7, 15, 30, 60, 120]

@dataclass
class KnowledgeNode:
    """知识图谱中的一个知识点"""
    topic: str                              # 主题名，如 "LoRA 微调"
    mastery: float = 0.0                    # 掌握度 0.0~1.0
    evidence_count: int = 0                 # 学习/练习次数
    last_study: str = ""                    # 最后学习时间 ISO
    next_review: str = ""                   # 下次复习时间 ISO
    review_stage: int = 0                   # 当前在 EBBINGHAUS_INTERVALS 中的位置
    related: List[str] = field(default_factory=list)  # 关联主题
    notes: str = ""                         # 学习笔记摘要

    def record_study(self, quality: float = 0.5):
        """
        记录一次学习。quality: 0=完全不理解, 1=完全掌握
        掌握度使用加权平均更新，避免一次学习就跳到很高
        """
        old_weight = self.evidence_count
        new_weight = 1
        total_weight = old_weight + new_weight
        self.mastery = (self.mastery * old_weight + quality * new_weight) / total_weight
        self.evidence_count += 1
        self.last_study = datetime.now().isoformat()

        # 艾宾浩斯复习调度
        if quality >= 0.6:
            # 理解了，推进复习间隔
            self.review_stage = min(self.review_stage + 1, len(EBBINGHAUS_INTERVALS) - 1)
        else:
            # 没理解，回到更短的间隔
            self.review_stage = max(0, self.review_stage - 2)

        days = EBBINGHAUS_INTERVALS[self.review_stage]
        self.next_review = (datetime.now() + timedelta(days=days)).isoformat()

    @property
    def mastery_label(self) -> str:
        """掌握度标签"""
        if self.mastery >= 0.85:
            return "精通"
        elif self.mastery >= 0.65:
            return "熟练"
        elif self.mastery >= 0.4:
            return "了解"
        elif self.mastery > 0:
            return "入门"
        return "未学"

    @property
    def needs_review(self) -> bool:
        """今天是否需要复习"""
        if not self.next_review:
            return True
        return datetime.fromisoformat(self.next_review) <= datetime.now()


class KnowledgeGraph:
    """知识图谱管理器"""

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self._load()

    def get_or_create(self, topic: str) -> KnowledgeNode:
        """获取或创建知识点"""
        topic = topic.strip()
        if topic not in self.nodes:
            self.nodes[topic] = KnowledgeNode(topic=topic)
        return self.nodes[topic]

    def record_learning(self, topic: str, quality: float = 0.5, related: List[str] = None):
        """记录学习了一个主题"""
        node = self.get_or_create(topic)
        node.record_study(quality)
        if related:
            for r in related:
                r = r.strip()
                if r and r not in node.related:
                    node.related.append(r)
                # 双向关联
                other = self.get_or_create(r)
                if topic not in other.related:
                    other.related.append(topic)
        self._save()

    def extract_topics(self, text: str) -> List[str]:
        """从文本中提取可能的知识点（启发式 + 质量过滤）"""
        # 1. 匹配关键技术名词和概念
        patterns = [
            r'(?:什么是|关于|学习|理解|掌握|介绍)(.{2,20}?)(?:[，。！？\n]|$)',
            r'\*\*([^*]{2,40})\*\*',          # Markdown 加粗
            r'`([^`]{2,30})`',                 # 行内代码
            r'#{1,3}\s+(.{2,50}?)$',           # 标题
            r'([A-Z][a-zA-Z]+(?:-[a-zA-Z]+)?)', # 英文专有名词
        ]
        topics = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                m = m.strip()
                # 质量过滤
                if len(m) < 3 or len(m) > 40:
                    continue
                if any(c in m for c in '：:】【[]（）()*#|'):
                    continue
                if m.startswith(('一个', '一种', '这个', '那个', '什么', '如何', '怎么')):
                    continue
                # 必须包含中文或有实际内容
                has_content = (
                    bool(re.search(r'[\u4e00-\u9fff]', m)) or     # 有中文
                    (len(m) >= 3 and re.match(r'^[A-Z]', m))       # 英文专有名词（允许缩写）
                )
                if has_content:
                    topics.append(m)

        # 去重 + 保留前 5 个
        seen = set()
        result = []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                result.append(t)
        return result[:5]

    def get_weak_topics(self, limit: int = 5) -> List[KnowledgeNode]:
        """获取掌握度最低的主题（需要加强）"""
        sorted_nodes = sorted(
            [n for n in self.nodes.values() if n.evidence_count > 0],
            key=lambda n: n.mastery
        )
        return sorted_nodes[:limit]

    def get_review_plan(self, limit: int = 5) -> List[KnowledgeNode]:
        """获取今天的复习计划"""
        due = [n for n in self.nodes.values() if n.needs_review and n.evidence_count > 0]
        # 优先复习掌握度低的、逾期最久的
        due.sort(key=lambda n: (n.mastery, n.next_review))
        return due[:limit]

    def get_related(self, topic: str) -> List[KnowledgeNode]:
        """获取关联主题"""
        node = self.nodes.get(topic)
        if not node:
            return []
        return [self.nodes[r] for r in node.related if r in self.nodes]

    def get_all_topics(self) -> List[Dict]:
        """获取所有主题的状态"""
        return [
            {
                "topic": n.topic,
                "mastery": round(n.mastery, 2),
                "label": n.mastery_label,
                "evidence_count": n.evidence_count,
                "needs_review": n.needs_review,
                "related": n.related,
            }
            for n in sorted(self.nodes.values(),
                          key=lambda n: (-n.mastery, n.topic))
        ]

    def _save(self):
        path = os.path.join(MEMORY_DIR, "knowledge_graph.json")
        data = {
            topic: {
                "topic": n.topic,
                "mastery": n.mastery,
                "evidence_count": n.evidence_count,
                "last_study": n.last_study,
                "next_review": n.next_review,
                "review_stage": n.review_stage,
                "related": n.related,
                "notes": n.notes,
            }
            for topic, n in self.nodes.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        path = os.path.join(MEMORY_DIR, "knowledge_graph.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for topic, d in data.items():
                self.nodes[topic] = KnowledgeNode(
                    topic=d["topic"],
                    mastery=d.get("mastery", 0),
                    evidence_count=d.get("evidence_count", 0),
                    last_study=d.get("last_study", ""),
                    next_review=d.get("next_review", ""),
                    review_stage=d.get("review_stage", 0),
                    related=d.get("related", []),
                    notes=d.get("notes", ""),
                )


# ============================================================
# 用户模型
# ============================================================

@dataclass
class UserProfile:
    """学习者档案"""
    learning_goal: str = ""           # 学习目标
    level: str = "初级"               # 当前水平
    preferred_style: str = "理论+实践"  # 偏好学习方式
    weak_areas: List[str] = field(default_factory=list)  # 薄弱领域
    strengths: List[str] = field(default_factory=list)   # 擅长领域
    total_sessions: int = 0           # 学习次数
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "goal": self.learning_goal,
            "level": self.level,
            "preferred_style": self.preferred_style,
            "weak_areas": self.weak_areas,
            "strengths": self.strengths,
            "total_sessions": self.total_sessions,
            "created": self.created,
        }


# ============================================================
# Agent 记忆系统 —— 整合所有模块
# ============================================================

class AgentMemory:
    """
    Agent 记忆系统
    ├── 工作记忆（短期对话）
    ├── 语义检索（TF-IDF）
    ├── 知识图谱（掌握度 + 复习）
    └── 用户模型（偏好 + 水平）
    """

    def __init__(self, short_term_size: int = 20):
        self.short_term_size = short_term_size
        self.short_term: List[Dict] = []
        self.index = SemanticIndex()
        self.kg = KnowledgeGraph()
        self.profile = UserProfile()
        self._load_profile()

    # ========== 短期记忆（滑动窗口）==========

    def add_message(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.short_term_size * 2:
            self.short_term = self.short_term[-self.short_term_size * 2:]

    def get_recent(self, n: int = None) -> List[Dict]:
        n = n or self.short_term_size
        return self.short_term[-n:]

    # ========== 长期记忆（语义检索）==========

    def save_memory(self, text: str, metadata: Dict = None):
        """保存一条长期记忆"""
        self.index.add(text)
        self._persist_index()

    def search_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义搜索长期记忆"""
        return self.index.search(query, top_k)

    # 兼容旧接口
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        return self.search_memory(query, top_k)

    # ========== 知识图谱接口 ==========

    def record_learning(self, topic: str, quality: float = 0.5, related: List[str] = None):
        """记录学习了一个知识点"""
        self.kg.record_learning(topic, quality, related)

    def auto_record_from_message(self, user_input: str, assistant_response: str):
        """从对话中自动提取知识点并记录（仅从用户问题提取）"""
        topics = self.kg.extract_topics(user_input)

        for topic in topics[:3]:
            self.kg.record_learning(topic, quality=0.45)

        summary = f"学习了: {user_input[:200]}"
        if topics:
            summary = f"主题: {', '.join(topics[:3])} — {user_input[:150]}"
        self.save_memory(summary)

    def get_weak_topics(self, limit: int = 5) -> List[Dict]:
        """哪些知识点需要加强"""
        nodes = self.kg.get_weak_topics(limit)
        return [{
            "topic": n.topic,
            "mastery": round(n.mastery, 2),
            "label": n.mastery_label,
            "evidence_count": n.evidence_count,
        } for n in nodes]

    def get_review_plan(self, limit: int = 5) -> List[Dict]:
        """今天该复习什么"""
        nodes = self.kg.get_review_plan(limit)
        return [{
            "topic": n.topic,
            "mastery": round(n.mastery, 2),
            "label": n.mastery_label,
            "next_review": n.next_review[:10] if n.next_review else "N/A",
        } for n in nodes]

    def get_knowledge_summary(self) -> Dict:
        """知识图谱总览"""
        all_topics = self.kg.get_all_topics()
        mastered = [t for t in all_topics if t["label"] in ("精通", "熟练")]
        learning = [t for t in all_topics if t["label"] in ("了解", "入门")]
        due_review = [t for t in all_topics if t["needs_review"]]
        weak_nodes = self.kg.get_weak_topics(3)

        return {
            "total_topics": len(all_topics),
            "mastered": len(mastered),
            "learning": len(learning),
            "due_review": len(due_review),
            "due_topics": [t["topic"] for t in due_review[:10]],
            "weakest": [{
                "topic": n.topic,
                "mastery": round(n.mastery, 2),
                "label": n.mastery_label
            } for n in weak_nodes],
        }

    def get_status(self) -> Dict:
        """获取记忆系统完整状态"""
        return {
            "short_term": len(self.short_term),
            "long_term": len(self.index.documents),
            "knowledge_graph": self.get_knowledge_summary(),
            "profile": self.profile.to_dict(),
        }

    # ========== 持久化 ==========

    def _persist_index(self):
        path = os.path.join(MEMORY_DIR, "long_term.json")
        data = {"texts": self.index.documents}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        path = os.path.join(MEMORY_DIR, "long_term.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for text in data.get("texts", []):
                self.index.add(text)

    def _load_profile(self):
        path = os.path.join(MEMORY_DIR, "profile.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profile = UserProfile(
                learning_goal=data.get("goal", ""),
                level=data.get("level", "初级"),
                preferred_style=data.get("preferred_style", "理论+实践"),
                weak_areas=data.get("weak_areas", []),
                strengths=data.get("strengths", []),
                total_sessions=data.get("total_sessions", 0),
                created=data.get("created", ""),
            )

    def save_profile(self):
        path = os.path.join(MEMORY_DIR, "profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.profile.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================
# 全局单例
# ============================================================

memory = AgentMemory()
# 加载已有长期记忆
memory._load_index()
