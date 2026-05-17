"""
轻量 Agent 运行时 —— StateGraph + 条件路由
不依赖 LangGraph，纯 Python 实现，方便理解每一行
"""
from typing import TypedDict, Literal, Callable, Dict, List, Any
from dataclasses import dataclass, field

# ============================================================
# 状态定义
# ============================================================

class AgentState(TypedDict, total=False):
    """贯穿所有节点的共享状态"""
    user_input: str               # 本轮用户输入
    intent: str                   # 路由判断结果: tutor / quiz / review / plan
    messages: List[Dict]          # 对话历史 [{"role":"user"/"assistant","content":"..."}]
    context: str                  # RAG 检索到的知识上下文
    response: str                 # 当前节点生成的回答
    quiz_question: str            # 生成的题目
    user_answer: str              # 用户回答
    quiz_result: Dict             # 判题结果
    learning_plan: str            # 学习计划
    progress: Dict                # 学习进度
    memory_hits: List[Dict]       # 记忆检索结果
    error: str                    # 错误信息
    finished: bool                # 是否终止

def empty_state() -> AgentState:
    return AgentState(
        user_input="",
        intent="tutor",
        messages=[],
        context="",
        response="",
        progress={},
        memory_hits=[],
        finished=False,
    )

# ============================================================
# 状态图引擎
# ============================================================

@dataclass
class StateGraph:
    """最简状态图：节点 + 条件边"""
    nodes: Dict[str, Callable] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)                # from → to
    cond_edges: Dict[str, tuple] = field(default_factory=dict)         # from → (router_fn, {"a":"to_a","b":"to_b"})
    entry: str = ""

    def add_node(self, name: str, fn: Callable):
        self.nodes[name] = fn
        return self

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node
        return self

    def add_conditional_edges(self, from_node: str, router: Callable, mapping: Dict[str, str]):
        self.cond_edges[from_node] = (router, mapping)
        return self

    def set_entry_point(self, name: str):
        self.entry = name
        return self

    def invoke(self, state: AgentState, max_steps: int = 10) -> AgentState:
        """执行图：入口 → 节点 → 边 → 下一个节点 → ... 直到 finished"""
        current = self.entry
        steps = 0

        while current and steps < max_steps:
            if current not in self.nodes:
                state["error"] = f"节点 '{current}' 不存在"
                break

            # 执行节点
            try:
                state = self.nodes[current](state)
            except Exception as e:
                state["error"] = f"节点 '{current}' 执行失败: {e}"
                break

            # 检查终止
            if state.get("finished"):
                break

            # 找下一个节点
            if current in self.cond_edges:
                router_fn, mapping = self.cond_edges[current]
                result = router_fn(state)
                current = mapping.get(result)
            elif current in self.edges:
                current = self.edges[current]
            else:
                break

            steps += 1

        return state
