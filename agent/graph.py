"""
学习 Agent 主图 —— 组装节点，创建可调用的 Agent
"""
from .runtime import StateGraph, empty_state, AgentState
from .nodes import (
    router_node, route_by_intent,
    tutor_node, quiz_node, review_node, plan_node, chat_node,
)


def build_agent() -> StateGraph:
    """构建学习 Agent 状态图"""
    graph = StateGraph()

    # 注册节点
    graph.add_node("router", router_node)
    graph.add_node("tutor", tutor_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("review", review_node)
    graph.add_node("plan", plan_node)
    graph.add_node("chat", chat_node)

    # 入口
    graph.set_entry_point("router")

    # 条件路由：根据意图分发到不同节点
    graph.add_conditional_edges("router", route_by_intent, {
        "tutor": "tutor",
        "quiz": "quiz",
        "review": "review",
        "plan": "plan",
        "chat": "chat",
    })

    return graph


# 全局 Agent 实例
agent = build_agent()


def run_agent(user_input: str, context: str = "") -> dict:
    """运行一次 Agent 对话，返回完整状态"""
    state = empty_state()
    state["user_input"] = user_input
    state["context"] = context
    state["messages"] = []
    result = agent.invoke(state)
    return {
        "response": result.get("response", ""),
        "intent": result.get("intent", "tutor"),
        "memory_hits": result.get("memory_hits", []),
        "error": result.get("error", ""),
        "progress": result.get("progress", {}),
    }
