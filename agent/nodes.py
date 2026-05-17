"""
Agent 节点：路由 / 辅导 / 出题 / 记忆 / 规划
每个节点接收 AgentState，返回 AgentState
"""
from .llm import chat_sync
from .memory import memory

# ============================================================
# 路由节点 —— 判断用户意图
# ============================================================

ROUTER_PROMPT = """你是一个学习助手的意图识别器。根据用户输入，判断意图：

- tutor: 用户提问、想学某个知识、需要解释概念
- quiz: 用户想做练习、测试、出题
- review: 用户想复习、回顾之前学过的内容
- plan: 用户想制定学习计划、规划路径
- chat: 闲聊或其他

只输出一个单词（tutor/quiz/review/plan/chat），不要其他内容。

用户输入: {user_input}
意图:"""


def router_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    if not user_input.strip():
        state["intent"] = "chat"
        return state

    # 简单关键词匹配，避免不必要的 API 调用
    lower = user_input.lower()
    if any(w in lower for w in ["题目", "出题", "测试", "练习", "做题", "quiz"]):
        state["intent"] = "quiz"
    elif any(w in lower for w in ["复习", "回顾", "之前", "review"]):
        state["intent"] = "review"
    elif any(w in lower for w in ["计划", "规划", "路线", "安排", "plan"]):
        state["intent"] = "plan"
    else:
        # 默认走 tutor 节点
        state["intent"] = "tutor"

    return state


def route_by_intent(state: dict) -> str:
    return state.get("intent", "tutor")


# ============================================================
# 辅导节点 —— 答疑解惑
# ============================================================

TUTOR_SYSTEM = """你是一个专业的学习导师 AI。你的教学风格：

1. 用通俗语言解释概念，先给一句话核心定义
2. 用具体例子帮助理解
3. 如果涉及代码，给出可运行的示例
4. 引导学生思考，而非直接给答案
5. 鼓励提问，不评判学生水平

注意：回答简洁有力，不堆砌信息。如果学生的问题不清晰，温和地请他补充细节。"""


def tutor_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    context = state.get("context", "")

    # 检索相关记忆
    hits = memory.search(user_input, top_k=3)
    memory_text = ""
    if hits:
        memory_text = "\n## 你的历史学习记录\n" + "\n".join(
            f"- {h['text']} (相关度: {h['score']:.2f})" for h in hits
        )
        state["memory_hits"] = hits

    # 构建消息
    messages = [{"role": "system", "content": TUTOR_SYSTEM}]

    # 注入近期对话
    recent = memory.get_recent(6)
    messages.extend(recent)

    # 注入记忆 + 上下文
    prompt = user_input
    if context:
        prompt = f"参考知识：\n{context}\n\n用户问题：{user_input}"
    if memory_text:
        prompt += f"\n{memory_text}"

    messages.append({"role": "user", "content": prompt})

    response = chat_sync(messages, temperature=0.7)
    state["response"] = response

    # 存入记忆
    memory.add_message("user", user_input[:500])
    memory.add_message("assistant", response[:500])
    # 提取关键知识点存为长期记忆
    if len(user_input) > 20:
        memory.save_memory(f"学习了: {user_input[:200]}")

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

输出格式：
【题目】...
【答案】...
【解析】..."""


def quiz_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    messages = [
        {"role": "system", "content": QUIZ_SYSTEM},
        {"role": "user", "content": user_input or "请给我出一道编程题"},
    ]
    response = chat_sync(messages, temperature=0.8)
    state["response"] = response
    memory.add_message("user", user_input[:200])
    memory.add_message("assistant", response[:500])
    state["finished"] = True
    return state


# ============================================================
# 复习节点 —— 回顾已学内容
# ============================================================

def review_node(state: dict) -> dict:
    user_input = state.get("user_input", "")

    # 检索长期记忆
    hits = memory.search(user_input or "学习", top_k=5)
    if not hits:
        state["response"] = "你还没有学习记录。先学点东西，我帮你记下来 😊\n试试说「教我 Python 的装饰器」"
        state["finished"] = True
        return state

    # 汇总记忆
    review_text = "## 📚 你的学习回顾\n\n"
    for i, h in enumerate(hits, 1):
        review_text += f"{i}. {h['text']}\n"

    review_text += f"\n共检索到 {len(hits)} 条相关记忆。你可以针对其中任何一点深入复习。"

    state["response"] = review_text
    memory.add_message("user", user_input[:200])
    memory.add_message("assistant", review_text[:500])
    state["finished"] = True
    return state


# ============================================================
# 规划节点 —— 制定学习计划
# ============================================================

PLAN_SYSTEM = """你是一个学习规划师。帮学生制定学习计划。

规则：
1. 先了解学生想学什么、每天能投入多少时间
2. 制定 3-7 天的学习路径
3. 每天包含：学习目标 + 核心知识点 + 练习任务
4. 输出简洁，用 markdown 列表

如果学生提供的信息不完整，先问他：
- 想学什么主题？
- 每天能学多久？
- 当前水平如何？"""


def plan_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    if not user_input.strip():
        state["response"] = (
            "好的，我来帮你制定学习计划。请告诉我：\n"
            "1. 你想学什么？（如 Python、机器学习、Web 开发）\n"
            "2. 每天能投入多少时间？\n"
            "3. 当前基础如何？（零基础 / 有一定了解 / 较熟练）"
        )
        state["finished"] = True
        return state

    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": user_input},
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
