"""
Agent 节点：路由 / 辅导 / 出题 / 复习 / 规划
每个节点接收 AgentState，返回 AgentState

v1.3: LLM意图路由 + 上下文预算管理
"""
from .llm import chat_sync
from .memory import memory
from .rag import kb

# ============================================================
# 上下文注入策略 —— 按意图控制注入内容，避免 token 爆炸
# ============================================================

# 每个意图的注入策略: (rag_chunks, memory_hits, show_weak, show_related, recent_msgs)
INJECTION_POLICY = {
    "tutor":  (2, 1, True,  True,  4),   # 辅导：知识库 + 记忆 + 薄弱点 + 关联
    "quiz":   (0, 0, True,  False, 0),   # 出题：只注入薄弱点作为出题参考
    "review": (0, 2, True,  False, 0),   # 复习：只看记忆和薄弱点
    "plan":   (0, 0, True,  False, 0),   # 规划：只注入知识状态
    "chat":   (0, 1, False, False, 2),   # 闲聊：轻量，只看最近对话
}


def _build_context(user_input: str, intent: str, budget: int = 600) -> str:
    """
    按意图分层构建上下文，总 token 不超预算。
    优先级: RAG > 薄弱点 > 记忆 > 关联主题 > 近期对话
    """
    policy = INJECTION_POLICY.get(intent, INJECTION_POLICY["tutor"])
    rag_n, mem_n, show_weak, show_related, recent_n = policy

    def _est(s: str) -> int:
        return len(s) // 2  # 粗略: 1 token ≈ 2 中文字符

    parts = []
    remaining = budget

    # 1. RAG 知识库（最高优先级——回答问题靠它）
    if rag_n > 0:
        rag = kb.search_formatted(user_input, top_k=rag_n)
        if rag:
            cost = _est(rag)
            max_rag = int(budget * 0.6)
            if cost <= max_rag:
                parts.append(rag)
                remaining -= cost

    # 2. 薄弱环节（掌握度 < 0.6 的）
    if show_weak and remaining > 80:
        weak = [w for w in memory.get_weak_topics(3) if w["mastery"] < 0.6]
        if weak:
            text = "薄弱环节: " + " | ".join(
                f"{w['topic']}({w['label']})" for w in weak
            )
            cost = _est(text)
            if cost <= remaining:
                parts.append(text)
                remaining -= cost

    # 3. 语义记忆
    if mem_n > 0 and remaining > 100:
        hits = memory.search_memory(user_input, top_k=mem_n)
        if hits:
            text = "历史: " + hits[0]["text"][:120]
            cost = _est(text)
            if cost <= remaining:
                parts.append(text)
                remaining -= cost

    # 4. 关联主题
    if show_related and remaining > 100:
        topics = memory.kg.extract_topics(user_input)
        for topic in topics[:1]:
            related = memory.kg.get_related(topic)
            if related:
                text = "关联: " + ", ".join(
                    f"{n.topic}({n.mastery_label})" for n in related[:3]
                )
                cost = _est(text)
                if cost <= remaining:
                    parts.append(text)
                    remaining -= cost
                break

    return "\n".join(parts)


# ============================================================
# 路由节点 —— 增强关键词匹配（覆盖边界情况）
# ============================================================

_ROUTES = [
    # quiz: 出题相关
    (["出题", "出道", "出一道", "出几道", "做题", "题目", "测试", "练习", "quiz",
      "选择题", "简答题", "编程题", "来一题", "考考", "提问（出题"], "quiz"),
    # review: 复习回顾
    (["复习", "回顾", "之前学的", "掌握情况", "薄弱", "review",
      "我学了什么", "我的进度", "记忆"], "review"),
    # plan: 规划安排
    (["计划", "规划", "路线", "安排", "plan", "学习路径", "制定",
      "怎么学", "学习方向"], "plan"),
    # chat: 闲聊
    (["你好", "谢谢", "再见", "哈哈", "不错", "今天", "天气",
      "hello", "hi", "嗯", "哦"], "chat"),
]

def router_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    if not user_input.strip():
        state["intent"] = "chat"
        return state

    lower = user_input.lower()

    for keywords, intent in _ROUTES:
        if any(kw.lower() in lower for kw in keywords):
            state["intent"] = intent
            return state

    # 兜底走 tutor
    state["intent"] = "tutor"
    return state


def route_by_intent(state: dict) -> str:
    return state.get("intent", "tutor")


# ============================================================
# 辅导节点 —— 答疑解惑（集成知识图谱 + 语义记忆 + 上下文预算）
# ============================================================

TUTOR_SYSTEM = """你是一个专业的学习导师 AI。你的教学风格：

1. 用通俗语言解释概念，先给一句话核心定义
2. 用具体例子帮助理解
3. 如果涉及代码，给出可运行的示例
4. 引导学生思考，而非直接给答案
5. 鼓励提问，不评判学生水平
6. 如果提供了「学生学习档案」，根据其当前水平调整解释深度

注意：回答简洁有力，不堆砌信息。如果学生的问题不清晰，温和地请他补充细节。"""


def tutor_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    intent = state.get("intent", "tutor")
    context = state.get("context", "")

    # ── 按意图构建上下文（预算 600 tokens）──
    extra = _build_context(user_input, intent)

    # ── 构建消息 ──
    messages = [{"role": "system", "content": TUTOR_SYSTEM}]
    recent = memory.get_recent(6)
    messages.extend(recent)

    prompt = user_input
    if context:
        prompt = f"参考知识：\n{context}\n\n用户问题：{user_input}"
    if extra:
        prompt = f"{extra}\n\n**用户问题：**{user_input}"

    messages.append({"role": "user", "content": prompt})

    # ── 调用 LLM ──
    response = chat_sync(messages, temperature=0.7, model_kind="pro")
    state["response"] = response

    # ── 记忆存储 ──
    memory.add_message("user", user_input[:500])
    memory.add_message("assistant", response[:500])

    if len(user_input) > 10:
        memory.auto_record_from_message(user_input, response)
        memory.profile.total_sessions += 1
        memory.save_profile()

    state["finished"] = True
    return state


# ============================================================
# 出题节点 —— 生成练习题
# ============================================================

QUIZ_SYSTEM = """你是一个出题老师。根据学生的要求生成练习题。

规则：
1. 先判断学生想练习什么主题
2. 生成 1 道高质量题目（单选/简答/编程 均可）
3. 题目后附上正确答案和简要解析
4. 如果学生没指定主题，询问他想练习什么
5. 优先出学生薄弱环节的题目（如果有提供「薄弱环节」列表）

输出格式：
【题目】...
【答案】...
【解析】..."""


def quiz_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    intent = state.get("intent", "quiz")

    # 上下文：只注入薄弱点
    extra = _build_context(user_input, intent)
    prompt = (user_input or "请给我出一道题")
    if extra:
        prompt = f"{extra}\n\n用户要求：{prompt}"

    messages = [
        {"role": "system", "content": QUIZ_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    response = chat_sync(messages, temperature=0.8, model_kind="flash")
    state["response"] = response
    memory.add_message("user", user_input[:200])
    memory.add_message("assistant", response[:500])
    state["finished"] = True
    return state


# ============================================================
# 复习节点 —— 基于知识图谱的智能复习
# ============================================================

def review_node(state: dict) -> dict:
    user_input = state.get("user_input", "")

    kg_summary = memory.get_knowledge_summary()
    review_plan = memory.get_review_plan(5)

    if user_input.strip() and len(user_input) > 5:
        hits = memory.search_memory(user_input, top_k=5)
        if hits:
            review_text = f"## 📖 复习: {user_input}\n\n"
            for i, h in enumerate(hits, 1):
                review_text += f"{i}. {h['text']}\n"
            review_text += f"\n共找到 {len(hits)} 条相关记忆。"
        else:
            kb_hits = kb.search_formatted(user_input, top_k=3)
            if kb_hits:
                review_text = f"## 📖 从知识库复习: {user_input}\n\n{kb_hits}"
            else:
                review_text = f"关于「{user_input}」暂时没有学习记录。先学起来吧 😊"
    else:
        if review_plan:
            review_text = "## 📋 今日复习计划\n\n"
            for i, item in enumerate(review_plan, 1):
                review_text += f"{i}. **{item['topic']}** — {item['label']}（掌握度 {item['mastery']}）\n"
                review_text += f"   下次复习: {item['next_review']}\n\n"
            if kg_summary["weakest"]:
                review_text += "## ⚠️ 需要加强\n\n"
                for w in kg_summary["weakest"]:
                    review_text += f"- {w['topic']}: {w['label']}（{w['mastery']}）\n"
        else:
            review_text = "## 📊 知识图谱总览\n\n"
            review_text += f"- 📚 已学主题: **{kg_summary['total_topics']}** 个\n"
            review_text += f"- ✅ 已掌握: **{kg_summary['mastered']}** 个\n"
            review_text += f"- 📖 学习中: **{kg_summary['learning']}** 个\n"
            review_text += f"- ⏰ 待复习: **{kg_summary['due_review']}** 个\n\n"
            review_text += "目前还没有待复习的内容。继续学习新知识吧！"

    state["response"] = review_text
    memory.add_message("user", user_input[:200] or "查看复习计划")
    memory.add_message("assistant", review_text[:500])
    state["finished"] = True
    return state


# ============================================================
# 规划节点 —— 制定学习计划（结合用户水平）
# ============================================================

PLAN_SYSTEM = """你是一个学习规划师。帮学生制定学习计划。

规则：
1. 先了解学生想学什么、每天能投入多少时间
2. 制定 3-7 天的学习路径
3. 每天包含：学习目标 + 核心知识点 + 练习任务
4. 输出简洁，用 markdown 列表
5. 如果提供了学生当前水平信息和薄弱环节，据此调整计划

如果学生提供的信息不完整，先问他：
- 想学什么主题？
- 每天能学多久？
- 当前水平如何？"""


def plan_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    intent = state.get("intent", "plan")

    if not user_input.strip():
        kg_summary = memory.get_knowledge_summary()
        weak = memory.get_weak_topics(5)

        response = (
            "好的，我来帮你制定学习计划。请告诉我：\n"
            "1. 你想学什么？（如 Python、机器学习、Web 开发）\n"
            "2. 每天能投入多少时间？\n"
            "3. 当前基础如何？（零基础 / 有一定了解 / 较熟练）\n"
        )
        if kg_summary["total_topics"] > 0:
            response += (
                f"\n\n📊 你目前已学 **{kg_summary['total_topics']}** 个主题，"
                f"掌握 {kg_summary['mastered']} 个。"
            )
        if weak:
            response += f"\n⚠️ 需要加强: {', '.join(w['topic'] for w in weak[:3])}"
        state["response"] = response
        state["finished"] = True
        return state

    # 上下文：只注入知识状态
    extra = _build_context(user_input, intent)
    prompt = user_input
    if extra:
        prompt = f"{extra}\n\n用户需求：{user_input}"

    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    response = chat_sync(messages, temperature=0.7, model_kind="pro")
    state["response"] = response
    memory.add_message("user", user_input[:300])
    memory.add_message("assistant", response[:500])
    state["finished"] = True
    return state


# ============================================================
# 闲聊节点 —— 兜底
# ============================================================

def chat_node(state: dict) -> dict:
    user_input = state.get("user_input", "你好")
    messages = [
        {"role": "system", "content": "你是一个友好的学习助手。用简洁温暖的方式回应。"},
        {"role": "user", "content": user_input},
    ]
    response = chat_sync(messages, temperature=0.7, model_kind="flash")
    state["response"] = response
    state["finished"] = True
    return state
