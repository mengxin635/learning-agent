"""
多轮出题 + 自动判卷系统

架构：
  QuizSession  — 单次测验会话（题目列表、当前进度、得分）
  QuizManager   — 会话管理器（创建/恢复/销毁会话）
  LLM 出题      — 根据主题/难度生成结构化题目
  LLM 判卷      — 对答案评分 + 给出解析
"""

import json
import uuid
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from .llm import chat_sync
from .memory import memory


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Question:
    """一道题目"""
    q_type: str          # choice / fill / coding / short
    text: str            # 题目文本
    options: List[str]   # 选项（仅 choice 有）
    answer: str          # 正确答案
    explanation: str     # 解析
    difficulty: str      # easy / medium / hard
    topic: str           # 所属知识点


@dataclass
class AnswerRecord:
    """一条作答记录"""
    question_num: int
    question: Question
    user_answer: str
    correct: bool
    scored_at: str = ""


@dataclass
class QuizSession:
    """一次测验会话"""
    session_id: str
    questions: List[Question] = field(default_factory=list)
    answers: List[AnswerRecord] = field(default_factory=list)
    current_index: int = 0
    topic: str = ""
    difficulty: str = "medium"
    q_type: str = "choice"
    score: int = 0
    total: int = 0
    created_at: str = ""
    finished: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def current_question(self) -> Optional[Question]:
        """当前题目"""
        if self.finished or self.current_index >= len(self.questions):
            return None
        return self.questions[self.current_index]

    def progress(self) -> Dict:
        """进度信息"""
        return {
            "total": self.total,
            "answered": len(self.answers),
            "current": self.current_index + 1,
            "correct": self.score,
            "done": self.finished,
        }


# ============================================================
# LLM 出题 prompt
# ============================================================

GEN_QUIZ_PROMPT = """你是一个专业出题老师。根据以下要求生成 {count} 道{type_desc}，难度 {difficulty}。

主题: {topic}
{weak_context}

题目要求：
1. 题目要有代表性，能真正考察知识掌握程度
2. 选项要有迷惑性（选择题），错误选项要像真的
3. 每道题都要有详细解析，解释为什么对、为什么错

输出严格的 JSON 数组格式（不要 markdown 代码块，只输出纯 JSON）:
[
  {{
    "q_type": "{q_type}",
    "text": "题目内容",
    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
    "answer": "A",
    "explanation": "详细解析",
    "difficulty": "{difficulty}",
    "topic": "具体知识点"
  }}
]

注意: options 字段仅 q_type=choice 时需要，其他类型可省略或为空数组。answer 是正确答案的文字或选项字母。"""


GEN_QUIZ_PROMPT_MIXED = """你是一个专业出题老师。根据以下要求生成 {count} 道题目，题型混搭（选择题、填空题、简答题），难度 {difficulty}。

主题: {topic}
{weak_context}

题目要求：
1. 选择题: 4个选项，有迷惑性，附解析
2. 填空题: 关键概念填空，答案简洁明确
3. 简答题: 需要2-5句话回答，考察理解深度
4. 每道题都要有详细解析

输出严格的 JSON 数组格式（不要 markdown 代码块，只输出纯 JSON）:
[
  {{
    "q_type": "choice",
    "text": "题目内容",
    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
    "answer": "A",
    "explanation": "详细解析",
    "difficulty": "{difficulty}",
    "topic": "具体知识点"
  }},
  {{
    "q_type": "fill",
    "text": "题目内容（用___标出填空位置）",
    "options": [],
    "answer": "正确答案",
    "explanation": "详细解析",
    "difficulty": "{difficulty}",
    "topic": "具体知识点"
  }}
]"""


GRADE_PROMPT = """你是一个严格的阅卷老师。根据以下题目和标准答案，评判学生的回答。

题目: {question}
题型: {q_type}
标准答案: {correct_answer}
学生回答: {user_answer}

评分标准：
- 选择题: 必须完全匹配标准答案（字母或内容）
- 填空题: 语义等价即可，宽松判分
- 简答题: 看是否涵盖关键点，部分正确给半对
- 编程题: 看核心逻辑是否正确，语法小错可忽略

输出 JSON:
{{
  "correct": true/false,
  "explanation": "评判理由，简要说明对/错在哪里"
}}"""


# ============================================================
# QuizManager — 会话管理
# ============================================================

QUIZ_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "quizzes")
os.makedirs(QUIZ_SESSIONS_DIR, exist_ok=True)


class QuizManager:
    """测验会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, QuizSession] = {}

    def _parse_json_response(self, text: str) -> list:
        """从 LLM 回复中提取 JSON 数组"""
        # 去掉可能的 markdown 代码块
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        # 尝试解析
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            return []
        except json.JSONDecodeError:
            # 尝试提取 [...] 部分
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return []

    def generate_questions(
        self,
        topic: str = "",
        difficulty: str = "medium",
        q_type: str = "choice",
        count: int = 5,
    ) -> List[Question]:
        """调用 LLM 生成题目"""
        type_desc = {
            "choice": "选择题",
            "fill": "填空题",
            "coding": "编程题",
            "mixed": "混搭题型（选择+填空+简答）",
        }.get(q_type, "选择题")

        # 获取薄弱环节作为出题参考
        weak_context = ""
        weak = memory.get_weak_topics(5)
        if weak:
            weak_topics = [w["topic"] for w in weak if w["mastery"] < 0.5]
            if weak_topics:
                weak_context = f"\n学生薄弱环节: {', '.join(weak_topics)}。优先考察这些知识点。"

        # 选择 prompt
        if q_type == "mixed":
            prompt = GEN_QUIZ_PROMPT_MIXED.format(
                count=count, difficulty=difficulty, topic=topic or "通用编程知识",
                weak_context=weak_context,
            )
        else:
            prompt = GEN_QUIZ_PROMPT.format(
                count=count, type_desc=type_desc, difficulty=difficulty,
                topic=topic or "通用编程知识",
                q_type=q_type,
                weak_context=weak_context,
            )

        messages = [
            {"role": "system", "content": "你是一个专业出题老师。只输出 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ]

        response = chat_sync(messages, temperature=0.9, max_tokens=4096, model_kind="flash")
        raw_questions = self._parse_json_response(response)

        questions = []
        for i, raw in enumerate(raw_questions):
            try:
                q = Question(
                    q_type=raw.get("q_type", q_type),
                    text=raw.get("text", f"题目 {i+1}"),
                    options=raw.get("options", []),
                    answer=raw.get("answer", ""),
                    explanation=raw.get("explanation", ""),
                    difficulty=raw.get("difficulty", difficulty),
                    topic=raw.get("topic", topic or "通用"),
                )
                questions.append(q)
            except Exception:
                continue

        return questions

    def start_quiz(
        self,
        topic: str = "",
        difficulty: str = "medium",
        q_type: str = "choice",
        count: int = 5,
    ) -> QuizSession:
        """开始新测验"""
        session_id = uuid.uuid4().hex[:12]
        questions = self.generate_questions(topic, difficulty, q_type, count)

        session = QuizSession(
            session_id=session_id,
            questions=questions,
            total=len(questions),
            topic=topic or "综合",
            difficulty=difficulty,
            q_type=q_type,
        )
        self.sessions[session_id] = session
        return session

    def submit_answer(self, session_id: str, answer: str) -> Dict:
        """提交答案并判卷"""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在或已过期"}

        if session.finished:
            return {"error": "测验已完成", "finished": True, "summary": self.get_summary(session_id)}

        question = session.current_question()
        if not question:
            return {"error": "没有更多题目"}

        # 判卷
        grading = self._grade_answer(question, answer)

        # 记录
        record = AnswerRecord(
            question_num=session.current_index + 1,
            question=question,
            user_answer=answer,
            correct=grading["correct"],
            scored_at=datetime.now().isoformat(),
        )
        session.answers.append(record)
        if grading["correct"]:
            session.score += 1

        session.current_index += 1
        if session.current_index >= len(session.questions):
            session.finished = True

        # 更新知识图谱
        quality = 0.8 if grading["correct"] else 0.25
        memory.kg.record_learning(question.topic, quality)

        # 检查是否还有下一题
        next_q = session.current_question()

        return {
            "session_id": session_id,
            "correct": grading["correct"],
            "user_answer": answer,
            "correct_answer": question.answer,
            "explanation": grading["explanation"],
            "progress": session.progress(),
            "next_question": self._question_to_dict(next_q) if next_q else None,
            "finished": session.finished,
        }

    def _grade_answer(self, question: Question, user_answer: str) -> Dict:
        """LLM 判卷"""
        # 选择题：直接比对（更快更准）
        if question.q_type == "choice":
            normalized_user = user_answer.strip().upper().replace(" ", "")
            normalized_correct = question.answer.strip().upper().replace(" ", "")
            # 尝试提取选项字母
            user_letter = re.match(r'^[A-D]', normalized_user)
            correct_letter = re.match(r'^[A-D]', normalized_correct)
            if user_letter and correct_letter:
                is_correct = user_letter.group() == correct_letter.group()
                exp = question.explanation or (
                    f"✅ 正确！答案就是 {question.answer}。" if is_correct
                    else f"❌ 正确答案是 {question.answer}。{question.explanation}"
                )
                return {"correct": is_correct, "explanation": exp}
            # 全内容比对
            is_correct = normalized_user == normalized_correct or user_answer.strip() == question.answer.strip()
            return {"correct": is_correct, "explanation": question.explanation or f"答案: {question.answer}"}

        # 填空/简答/编程：LLM 判卷
        prompt = GRADE_PROMPT.format(
            question=question.text,
            q_type=question.q_type,
            correct_answer=question.answer,
            user_answer=user_answer,
        )
        messages = [
            {"role": "system", "content": "你是严格的阅卷老师。只输出 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ]
        response = chat_sync(messages, temperature=0.3, max_tokens=512, model_kind="flash")
        try:
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            result = json.loads(text)
            return {
                "correct": result.get("correct", False),
                "explanation": result.get("explanation", question.explanation or ""),
            }
        except Exception:
            # 降级：简单比对
            return {
                "correct": user_answer.strip() == question.answer.strip(),
                "explanation": question.explanation or f"标准答案: {question.answer}",
            }

    def get_session(self, session_id: str) -> Optional[QuizSession]:
        return self.sessions.get(session_id)

    def get_summary(self, session_id: str) -> Dict:
        """获取测验总结"""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在"}

        total = session.total
        correct = session.score
        pct = round(correct / total * 100) if total > 0 else 0

        # 等级
        if pct >= 90:
            grade = "🏆 优秀"
        elif pct >= 70:
            grade = "👍 良好"
        elif pct >= 50:
            grade = "📖 继续努力"
        else:
            grade = "💪 需要加强"

        # 错误题目
        wrong_answers = [
            {
                "num": r.question_num,
                "topic": r.question.topic,
                "question": r.question.text[:80],
                "user_answer": r.user_answer,
                "correct_answer": r.question.answer,
                "explanation": r.question.explanation,
            }
            for r in session.answers if not r.correct
        ]

        # 知识点表现
        topic_performance = {}
        for r in session.answers:
            t = r.question.topic
            if t not in topic_performance:
                topic_performance[t] = {"correct": 0, "total": 0}
            topic_performance[t]["total"] += 1
            if r.correct:
                topic_performance[t]["correct"] += 1

        return {
            "session_id": session_id,
            "topic": session.topic,
            "total": total,
            "correct": correct,
            "percentage": pct,
            "grade": grade,
            "wrong_answers": wrong_answers,
            "topic_performance": [
                {"topic": t, "correct": d["correct"], "total": d["total"]}
                for t, d in topic_performance.items()
            ],
            "duration": "",
        }

    def _question_to_dict(self, q: Question) -> Dict:
        """题目转前端可用的 dict"""
        return {
            "q_type": q.q_type,
            "text": q.text,
            "options": q.options,
            "difficulty": q.difficulty,
            "topic": q.topic,
            # 不传 answer！防止作弊
        }


# ============================================================
# 全局实例
# ============================================================

quiz_manager = QuizManager()
