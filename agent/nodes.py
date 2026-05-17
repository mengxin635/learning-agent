"""
Agent 节点：路由 / 辅导 / 出题 / 复习 / 规划
每个节点接收 AgentState，返回 AgentState

v1.2: 集成学习型记忆系统（语义检索 + 知识图谱 + 间隔重复）
"""
from .llm import chat_sync
from .memory import memory
from .rag import kb

# ============================================================
# 路由节点 —— 判断用户意图
# ============================================================

def router_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    if not user_input.strip():
        state["intent"] = "chat"
        return state

    lower = user_input.lower()
    if any(w in lower for w in ["题目", "出题", "测试", "练习", "做题", "quiz"]):
        state["intent"] = "quiz"
    elif any(w in lower for w in ["复习", "回顾", "之前", "review", "掌握", "薄弱"]):
        state["intent"] = "review"
    elif any(w in lower for w in ["计划", "规划", "路线", "安排", "plan"]):
        state["intent"] = "plan"
    else:
        state["intent"] = "tutor"

    return state


def route_by_intent(state: dict) -> str:
    return state.get("intent", "tutor")


# ============================================================
# 辅导节点 —— 答疑解惑（集成知识图谱 + 语义记忆）
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
    context = state.get("context", "")

    # ── RAG 知识库检索 ──
    rag_context = kb.search_formatted(user_input, top_k=3)

    # ── 语义记忆检索（TF-IDF，比旧 n-gram 准很多）──
    memory_hits = memory.search_memory(user_input, top_k=3)
    memory_text = ""
    if memory_hits:
        memory_text = "\n## 你的历史学习记录\n" + "\n".join(
            f"- {h['text']}（相关度: {h['score']:.2f}）" for h in memory_hits
        )
        state["memory_hits"] = memory_hits

    # ── 知识图谱上下文 ──
    kg_context = ""
    weak = memory.get_weak_topics(3)
    if weak:
        kg_context += "\n## 你的薄弱环节（需要加强）\n" + "\n".join(
            f"- {w['topic']}（{w['label']}，掌握度 {w['mastery']}）" for w in weak
        )

    related_check = memory.kg.extract_topics(user_input)
    if related_check:
        for topic in related_check[:2]:
            related_nodes = memory.kg.get_related(topic)
            if related_nodes:
                kg_context += f"\n\n## 已学过的关联主题（与「{topic}」相关）\n"
                kg_context += "\n".join(
                    f"- {n.topic}（{n.mastery_label}，掌握度 {n.mastery:.0%}）" 
                    for n in related_nodes[:5]
                )
                break

    # ── 构建消息 ──
    messages = [{"role": "system", "content": TUTOR_SYSTEM}]
    recent = memory.get_recent(6)
    messages.extend(recent)

    prompt = user_input
    if rag_context:
        prompt = f"{rag_context}\n\n**用户问题：**{user_input}"
    elif context:
        prompt = f"参考知识：\n{context}\n\n用户问题：{user_input}"
    if memory_text:
        prompt += f"\n{memory_text}"
    if kg_context:
        prompt += f"\n{kg_context}"

    messages.append({"role": "user", "content": prompt})

    # ── 调用 LLM ──
    response = chat_sync(messages, temperature=0.7)
    state["response"] = response

    # ── 记忆存储 ──
    memory.add_message("user", user_input[:500])
    memory.add_message("assistant", response[:500])

    # 自动提取知识点并更新知识图谱
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

    # 注入薄弱环节作为出题参考
    weak = memory.get_weak_topics(3)
    weak_hint = ""
    if weak and (not user_input.strip() or len(user_input) < 10):
        weak_hint = "\n\n建议优先出以下薄弱环节的题目：\n" + "\n".join(
            f"- {w['topic']}（{w['label']}）" for w in weak
        )

    messages = [
        {"role": "system", "content": QUIZ_SYSTEM},
        {"role": "user", "content": (user_input or "请给我出一道题") + weak_hint},
    ]
    response = chat_sync(messages, temperature=0.8)
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

    # ── 1. 知识图谱总览 ──
    kg_summary = memory.get_knowledge_summary()
    review_plan = memory.get_review_plan(5)

    # ── 2. 如果指定了主题，精确复习 ──
    if user_input.strip() and len(user_input) > 5:
        # 语义搜索相关记忆
        hits = memory.search_memory(user_input, top_k=5)
        if hits:
            review_text = f"## 📖 复习: {user_input}\n\n"
            for i, h in enumerate(hits, 1):
                review_text += f"{i}. {h['text']}\n"
            review_text += f"\n共找到 {len(hits)} 条相关记忆。"
        else:
            # 尝试从知识库检索
            kb_hits = kb.search_formatted(user_input, top_k=3)
            if kb_hits:
                review_text = f"## 📖 从知识库复习: {user_input}\n\n{kb_hits}"
            else:
                review_text = f"关于「{user_input}」暂时没有学习记录。先学起来吧 😊"
    else:
        # ── 3. 自动复习计划 ──
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

    if not user_input.strip():
        # 没有具体需求时，展示当前知识状态
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

    # 注入用户知识状态
    kg_context = ""
    kg_summary = memory.get_knowledge_summary()
    if kg_summary["total_topics"] > 0:
        kg_context = (
            f"\n\n【学生当前状态】\n"
            f"已学 {kg_summary['total_topics']} 个主题，"
            f"掌握 {kg_summary['mastered']} 个。\n"
        )
        weak = memory.get_weak_topics(3)
        if weak:
            kg_context += f"薄弱环节: {', '.join(w['topic'] for w in weak)}"

    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_input + kg_context},
    ]
    response = chat_sync(messages, temperature=0.7)
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
    response = chat_sync(messages, temperature=0.7)
    state["response"] = response
    state["finished"] = True
    return state
