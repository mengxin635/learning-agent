# LangChain 与 LlamaIndex 框架中级指南

> 本文档为中级难度，面向已有大模型基础、正在学习 LLM 应用开发的读者。内容涵盖 LangChain 核心组件与生态（LCEL、LangGraph、LangSmith、LangServe）以及 LlamaIndex 核心概念与高级特性（数据连接器、高级检索、Agent 集成），最后给出两个框架的对比与选择建议。

---

## 目录

### 第一部分：LangChain

1. [LangChain 核心组件](#1-langchain-核心组件)
   - 1.1 [Chain（链）](#11-chain链)
   - 1.2 [Agent（智能体）](#12-agent智能体)
   - 1.3 [Memory（记忆）](#13-memory记忆)
   - 1.4 [Retriever（检索器）](#14-retriever检索器)
2. [LCEL 表达式语言](#2-lcel-表达式语言)
3. [LangGraph 状态图](#3-langgraph-状态图)
4. [LangSmith 调试与观测](#4-langsmith-调试与观测)
5. [LangServe 部署](#5-langserve-部署)

### 第二部分：LlamaIndex

6. [LlamaIndex 核心概念](#6-llamaindex-核心概念)
   - 6.1 [Document 与 Node](#61-document-与-node)
   - 6.2 [Index（索引）](#62-index索引)
   - 6.3 [QueryEngine（查询引擎）](#63-queryengine查询引擎)
7. [数据连接器](#7-数据连接器)
8. [高级检索](#8-高级检索)
9. [LlamaIndex Agent 集成](#9-llamaindex-agent-集成)

### 第三部分：对比与选择

10. [框架对比与选择建议](#10-框架对比与选择建议)

---

## 1. LangChain 核心组件

LangChain 是构建 LLM 驱动应用的"智能体工程平台"，当前最新版本将核心定位从"Chain 框架"升级为以 **Agent 为中心的架构**，底层由 LangGraph 提供状态图编排能力。

### 1.1 Chain（链）

Chain 是 LangChain 中最基础的**组合单元**，它将多个组件（LLM 调用、Prompt 模板、输出解析器等）串联成一条可执行的流水线。

#### 核心类型

| Chain 类型 | 说明 | 典型场景 |
|------------|------|----------|
| `LLMChain` | 最基本的链：Prompt + LLM + 可选 OutputParser | 单轮问答、文本生成 |
| `SequentialChain` | 多个 Chain 顺序执行，前一个输出作为后一个输入 | 多步骤流水线 |
| `RouterChain` | 根据输入动态选择下游 Chain | 意图路由、条件分支 |
| `RetrievalQA` | 检索 + 问答的组合链 | 知识库问答 |
| `ConversationalRetrievalChain` | 带对话历史的检索问答 | 多轮知识对话 |
| `StuffDocumentsChain` | 将所有检索到的文档直接拼入 Prompt | 简单 RAG |
| `MapReduceChain` | 先分别总结每篇文档，再合并总结 | 长文档摘要 |
| `RefineChain` | 逐篇迭代优化回答 | 逐步精炼生成 |

#### Chain 的工作原理

```
用户输入 → Prompt模板 → LLM调用 → 输出解析 → 下一环节 → ... → 最终输出
```

#### 代码示例（LCEL 风格）

```python
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

model = init_chat_model("openai:gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}。"),
    ("user", "{input}")
])

chain = prompt | model
result = chain.invoke({"role": "Python导师", "input": "什么是装饰器？"})
```

> **注意**：在最新的 LangChain 架构中，推荐使用 **LCEL**（`|` 管道操作符）而非旧式的 `LLMChain` 类来构建链，LCEL 提供更好的类型推断、流式支持、异步调用和自动并行化。

---

### 1.2 Agent（智能体）

Agent 是 LangChain 中**最核心的高级抽象**。与固定流程的 Chain 不同，Agent 具有**自主决策能力**：它根据任务目标和当前状态，动态决定使用哪个工具、调用多少次、何时停止。

#### Agent 核心架构

```
         ┌──────────────────────────────────┐
         │            Agent 循环            │
         │                                  │
  输入 ──▶│  思考（LLM推理）                 │
         │     │                            │
         │     ▼                            │
         │  决策（选择动作：工具/完成）        │
         │     │                            │
         │     ├── 工具调用 ──▶ 观察结果 ──┐  │
         │     │                          │  │
         │     └── 最终回答 ◀──────────────┘  │
         │                    │               │
         └────────────────────┼───────────────┘
                              ▼
                          用户输出
```

#### Agent 的关键组成

| 组件 | 作用 | 说明 |
|------|------|------|
| **LLM（推理引擎）** | 分析任务、制定计划、选择工具 | 需要支持 Function Calling |
| **Tools（工具集）** | Agent 可调用的外部能力 | 搜索、计算、API、数据库、文件系统等 |
| **ToolSelector** | 根据任务选择合适工具 | LLM 自主决策或预设路由 |
| **Scratchpad（中间记录）** | 保存推理链和工具调用历史 | 避免重复、提供上下文 |
| **StoppingCondition** | 判断何时终止 | 达到最终答案 / 超时 / 步数限制 |

#### Agent 类型演进

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **ReAct Agent** | 推理+行动交替：Thought → Action → Observation | 经典模式，可解释性强 |
| **OpenAI Functions Agent** | 基于 Function Calling，LLM 直接输出 JSON | 高效、结构化 |
| **Tool Calling Agent** | 通用化 Function Calling 抽象 | 跨模型厂商 |
| **Structured Chat Agent** | 支持多参数工具 | 复杂工具调用 |
| **Plan-and-Execute Agent** | 先制定完整计划再逐步执行 | 复杂多步骤任务 |
| **Deep Agents** (最新) | 内置规划、子代理、文件系统能力 | 复杂自主任务（LangChain 最新旗舰） |

#### 代码示例

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o")

# 定义工具
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果: {query}的相关信息..."

def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

agent = create_agent(
    model=model,
    tools=[search, calculate],
    system_prompt="你是一个有帮助的AI助手，可以使用搜索和计算工具。"
)

result = agent.invoke({"messages": [{"role": "user", "content": "2024年GDP增长率乘以2是多少？"}]})
```

#### 关键设计原则

1. **工具描述即契约**：工具的 docstring 和参数说明是 Agent 理解工具的唯一依据，务必清晰描述
2. **错误处理**：工具应返回有用的错误信息，帮助 Agent 自我修正
3. **权限控制**：敏感操作（删除、写文件）需要确认机制
4. **步数限制**：设置 `max_iterations` 防止无限循环

---

### 1.3 Memory（记忆）

Memory 让 LLM 应用具备**上下文保持**能力，在多次交互间维护状态。

#### Memory 类型体系

```
                    Memory 基类
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  对话记忆          实体记忆         摘要记忆
        │               │               │
   ┌────┴────┐    ┌─────┴─────┐    ┌────┴────┐
   ▼         ▼    ▼           ▼    ▼         ▼
Buffer   Window  EntityStore  KG  Summary  SummaryBuffer
```

| Memory 类型 | 机制 | 适用场景 | 优缺点 |
|-------------|------|----------|--------|
| **ConversationBufferMemory** | 完整保存所有对话轮次 | 短对话、调试 | ⚠️ Token 消耗线性增长 |
| **ConversationBufferWindowMemory** | 仅保留最近 K 轮对话 | 需限制上下文的场景 | 权衡：丢失早期信息 |
| **ConversationSummaryMemory** | 用 LLM 逐轮总结历史 | 长对话 | 节省 Token，但总结可能丢失细节 |
| **ConversationSummaryBufferMemory** | 窗口+摘要混合 | 平衡方案 | 保留最近细节+早期摘要 |
| **ConversationTokenBufferMemory** | 按 Token 数量截断 | 精确控制上下文长度 | 简单直接 |
| **VectorStoreRetrieverMemory** | 将历史存入向量库，检索最相关片段 | 大规模历史检索 | 语义相关而非时间相关 |
| **ConversationKGMemory** | 从对话中提取实体和关系构建知识图谱 | 需要推理关系的场景 | 可发现隐含关联 |

#### Memory 在 Agent 中的演进

在最新的 LangChain Agent 架构中，Memory 机制已深度融合进 LangGraph 的状态管理：

```python
from langgraph.checkpoint.memory import MemorySaver

# LangGraph 方式：checkpointer 持久化整个状态图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 同一 thread_id 的多次调用自动继承历史状态
config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "我刚才说了什么？"}]}, config)
```

#### 记忆策略选择指南

| 场景 | 推荐 Memory | 原因 |
|------|------------|------|
| 客服对话 | ConversationBufferWindowMemory (K=10) | 保留最近上下文，防止 Token 溢出 |
| 个人助手 | ConversationSummaryBufferMemory | 长期记忆 + 近期细节 |
| 知识库问答 | VectorStoreRetrieverMemory | 按语义检索历史 |
| 简单原型 | ConversationBufferMemory | 简单直接 |

---

### 1.4 Retriever（检索器）

Retriever 是 LangChain 中负责**从外部知识源获取相关文档**的组件，是 RAG 架构的核心一环。

#### Retriever 的抽象接口

```python
# 核心接口极其简洁
def retrieve(query: str) -> List[Document]:
    """给定查询字符串，返回相关文档列表"""
```

#### 常见 Retriever 类型

```python
# 1. 向量存储检索器
from langchain_chroma import Chroma
retriever = Chroma.from_documents(docs, embeddings).as_retriever(
    search_type="similarity",  # similarity | mmr | similarity_score_threshold
    search_kwargs={"k": 4}
)

# 2. 多查询检索器 — 从多角度查询，合并结果
from langchain.retrievers import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=model
)

# 3. 上下文压缩检索器 — 检索后过滤无关内容
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(model)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. 自查询检索器 — 从自然语言中提取元数据过滤
from langchain.retrievers import SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm=model,
    vectorstore=vectorstore,
    document_content_description="技术文档",
    metadata_field_info=[...]
)

# 5. Ensemble 检索器 — 融合多种检索结果
from langchain.retrievers import EnsembleRetriever
ensemble = EnsembleRetriever(
    retrievers=[sparse_retriever, dense_retriever],
    weights=[0.3, 0.7]  # 混合权重
)

# 6. 父文档检索器 — 检索小块、返回大块
from langchain.retrievers import ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=small_splitter,   # 用于检索的小块
    parent_splitter=large_splitter,  # 返回给 LLM 的大块
)
```

#### 检索增强策略总结

| 策略 | 解决的问题 | 实现方式 |
|------|-----------|----------|
| **MultiQuery** | 单一查询角度有限 | 生成多个变体查询，合并去重 |
| **HyDE** | 查询与文档用词不匹配 | 先生成假设答案，再检索 |
| **Contextual Compression** | 检索结果含噪 | LLM 过滤无关片段 |
| **Self-Query** | 需元数据过滤 | 从自然语言提取过滤条件 |
| **Parent Document** | 小块检索 vs 大块上下文矛盾 | 检索用小块，返回大块 |
| **Ensemble** | 单一检索器有盲区 | 多检索器加权融合 (RRF) |
| **Re-ranking** | 初次检索排序不准 | 用 Cross-Encoder 重排序 |

---

## 2. LCEL 表达式语言

**LCEL（LangChain Expression Language）** 是 LangChain 的声明式组合语言，使用 Unix 管道符 `|` 将组件串联成可运行的链。它是 LangChain 推荐的标准构建方式。

### 2.1 核心语法

```python
# 基本语法：component1 | component2 | component3
chain = prompt | model | output_parser

# 等价于：output_parser(model(prompt(input)))
```

### 2.2 LCEL 的核心优势

| 特性 | 说明 |
|------|------|
| **流式支持** | 自动逐 Token 输出，`chain.stream()` |
| **异步支持** | 自动提供 `ainvoke()`, `astream()`, `abatch()` |
| **并行执行** | `RunnableParallel` 自动并行化独立步骤 |
| **回退机制** | `.with_fallbacks()` 在失败时切换到备用模型 |
| **可观测性** | 自动与 LangSmith 集成，追踪每一步 |
| **类型推断** | 链式调用中自动推断输入输出类型 |

### 2.3 LCEL 核心原语

```python
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RunnableMap,
)

# --- RunnablePassthrough: 透传数据 ---
chain = {"input": RunnablePassthrough()} | prompt | model

# --- RunnableParallel: 并行执行 ---
chain = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain,
)
# 三条子链并行执行，结果合并为 {"summary": ..., "keywords": ..., "sentiment": ...}

# --- RunnableLambda: 包装自定义函数 ---
def custom_transform(x: str) -> str:
    return x.upper()

chain = RunnableLambda(custom_transform) | model

# --- RunnableBranch: 条件分支 ---
chain = RunnableBranch(
    (lambda x: len(x) < 10, short_chain),
    (lambda x: len(x) < 100, medium_chain),
    long_chain,  # 默认分支
)

# --- RunnableMap: 构造字典输入 ---
chain = RunnableMap({
    "context": retriever,
    "question": RunnablePassthrough()
}) | prompt | model

# --- .with_fallbacks(): 容错回退 ---
chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

# --- .bind(): 绑定运行时参数 ---
chain = prompt | model.bind(stop=["\nObservation"], temperature=0)
```

### 2.4 复杂 LCEL 示例

```python
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

model = init_chat_model("openai:gpt-4o")

# 多步骤 RAG Pipeline
retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下上下文回答问题：\n\n{context}"),
    ("user", "{question}")
])

# 预处理步骤
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,   # 检索+格式化
        "question": RunnablePassthrough()      # 透传问题
    })
    | retrieval_prompt
    | model
    | StrOutputParser()
)

# 调用方式多样
result = rag_chain.invoke("什么是RAG？")
async for chunk in rag_chain.astream("什么是RAG？"):
    print(chunk, end="", flush=True)
```

### 2.5 LCEL 执行模型

```
invoke(input)
    │
    ▼
┌────────────────────────────────────────────────┐
│               Runnable Sequence                 │
│                                                │
│  input ──▶ step1 ──▶ step2 ──▶ step3 ──▶ output│
│            (prompt)   (model)   (parser)        │
│                                                │
│  每个 step 支持:                                │
│  - .invoke()  同步调用                          │
│  - .ainvoke() 异步调用                          │
│  - .stream()  流式输出                          │
│  - .batch()   批量处理                          │
└────────────────────────────────────────────────┘
```

---

## 3. LangGraph 状态图

**LangGraph** 是 LangChain 生态中的**低层级 Agent 编排框架**，用于构建可控、可靠的智能体工作流。它把 Agent 建模为**有向状态图**，节点执行动作，边控制流转。

### 3.1 核心概念

```
┌─────────────────────────────────────────────────┐
│                  LangGraph                       │
│                                                 │
│   State（状态）  ──  贯穿全图的共享数据结构       │
│   Nodes（节点）  ──  执行具体逻辑（LLM调用/工具）  │
│   Edges（边）    ──  控制节点间的流转方向          │
│   Conditional    ──  根据状态动态选择下一节点      │
│   Edges（条件边）                                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 3.2 状态 (State)

State 是 LangGraph 的核心——贯穿整个图执行的**共享数据对象**，通常使用 `TypedDict` 或 Pydantic 模型定义：

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # 使用 add_messages reducer
    next_step: str
    tool_results: dict
```

关键机制：
- **Reducer 函数**：定义状态如何被更新。`add_messages` 将新消息追加到消息列表（而非覆盖）
- **状态持久化**：通过 Checkpointer 自动保存/恢复状态

### 3.3 图构建

```python
from langgraph.graph import StateGraph, START, END

# 1. 定义图
builder = StateGraph(AgentState)

# 2. 添加节点
builder.add_node("agent", call_model)        # Agent推理节点
builder.add_node("tools", call_tools)        # 工具执行节点

# 3. 添加边
builder.add_edge(START, "agent")              # 入口 → agent
builder.add_conditional_edges(                # agent → tools 或 END
    "agent",
    should_continue,                          # 条件路由函数
    {"continue": "tools", "end": END}
)
builder.add_edge("tools", "agent")            # tools → agent（循环）

# 4. 编译（可选 checkpointer 持久化）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 3.4 条件路由

```python
def should_continue(state: AgentState) -> str:
    """决策路由逻辑"""
    last_message = state["messages"][-1]

    # 如果 LLM 请求工具调用 → 进入 tools 节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # 否则结束
    return "end"
```

### 3.5 常见图模式

#### 模式一：Agent 循环（ReAct 模式）

```
      ┌──────────┐
      │  agent   │◀────┐
      └────┬─────┘     │
           │           │
    [条件判断]         │
      │        │       │
   需要工具   完成      │
      │        │       │
      ▼        ▼       │
  ┌───────┐  END       │
  │ tools │────────────┘
  └───────┘
```

#### 模式二：并行 Fan-out / Fan-in

```
              ┌──────────┐
       ┌─────▶│ sub_task1 ├─────┐
       │      └──────────┘     │
START ─┼───────────────────────┼──▶ END
       │      ┌──────────┐     │
       └─────▶│ sub_task2 ├─────┘
              └──────────┘
```

#### 模式三：多 Agent 协作（Supervisor 模式）

```
              ┌──────────┐
              │supervisor│
              └────┬─────┘
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │Agent_A │ │Agent_B │ │Agent_C │
  └────────┘ └────────┘ └────────┘
```

### 3.6 LangGraph vs 传统 Agent

| 维度 | 传统 Agent (create_agent) | LangGraph |
|------|---------------------------|-----------|
| **控制粒度** | 粗粒度（Agent 自主决策） | 细粒度（开发者定义每步流程） |
| **可定制性** | 有限（主要配置提示词和工具） | 极高（自定义节点、边、状态） |
| **可靠性** | 依赖 LLM 推理质量 | 开发者控制关键路径 |
| **复杂度** | 低，几行代码 | 高，需要设计图结构 |
| **适用场景** | 通用 Agent 任务 | 需要精确控制的生产级工作流 |
| **人机交互** | 有限 | 支持 Human-in-the-loop（中断、审批） |

### 3.7 Human-in-the-Loop

LangGraph 支持在工作流中插入人工审批节点：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

def sensitive_node(state):
    # 暂停执行，等待人工审批
    approval = interrupt("是否批准此操作？(yes/no)")
    if approval.lower() != "yes":
        return {"messages": [{"role": "assistant", "content": "操作已取消"}]}
    # 继续执行...
    return result

# 恢复时传入批准值
graph.invoke(None, config, interrupt_before=["sensitive_node"])
# 人工审批后：
graph.invoke(Command(resume="yes"), config)
```

---

## 4. LangSmith 调试与观测

**LangSmith** 是 LangChain 官方的 LLM 应用**调试、测试、评估和监控平台**。

### 4.1 核心功能

```
┌─────────────────────────────────────────────────────┐
│                    LangSmith                         │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  追踪    │  │  评估    │  │  监控    │          │
│  │ Tracing  │  │ Eval     │  │ Monitor  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│       │              │              │               │
│  ┌────┴────┐    ┌────┴────┐   ┌────┴────┐          │
│  │ 延迟分析 │    │ 正确性   │   │ 漂移检测 │          │
│  │ 错误追踪 │    │ 幻觉检测 │   │ 成本追踪 │          │
│  │ Token统计│    │ 回归测试 │   │ 数据标注 │          │
│  └─────────┘    └─────────┘   └─────────┘          │
└─────────────────────────────────────────────────────┘
```

### 4.2 快速集成

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."  # 从 smith.langchain.com 获取
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 之后所有 LangChain / LangGraph 调用自动追踪
```

### 4.3 核心概念：Run 与 Trace

| 概念 | 说明 |
|------|------|
| **Trace** | 一次完整的端到端调用（用户输入 → 最终输出） |
| **Run** | Trace 中的一个步骤（一次 LLM 调用、一次检索等） |
| **Span** | Run 的子步骤 |

```
Trace: "用户问2024年GDP增长率"
  ├── Run: LLM 调用 (Agent推理)
  │   ├── Span: Prompt 构造
  │   └── Span: Token 生成
  ├── Run: 工具调用 (搜索)
  │   └── Span: API 请求
  └── Run: LLM 调用 (最终回答)
```

### 4.4 评估功能

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 创建数据集
dataset = client.create_dataset("QA-test", description="问答测试")

# 定义评估器
def correctness_evaluator(run, example):
    """评估回答是否正确"""
    predicted = run.outputs["output"]
    expected = example.outputs["answer"]
    # 使用 LLM-as-Judge 评分
    score = llm_judge(predicted, expected)
    return {"score": score, "key": "correctness"}

# 运行评估
results = evaluate(
    lambda x: my_app.invoke(x["question"]),
    data="QA-test",
    evaluators=[correctness_evaluator],
    experiment_prefix="v1.0",
)
```

### 4.5 调试工作流

```
发现问题
    │
    ▼
1. 在 LangSmith 找到失败的 Trace
    │
    ▼
2. 展开 Run 树，定位故障步骤
    │
    ▼
3. 查看 Prompt 内容、LLM 参数、中间输出
    │
    ▼
4. 在 UI 中"Playground"重现问题
    │
    ▼
5. 修改代码或 Prompt
    │
    ▼
6. 用数据集运行回归测试
    │
    ▼
7. 确认修复 → 重新部署
```

### 4.6 高级功能

- **在线评估**：自动对每条 Trace 打分
- **人工标注**：标记"满意/不满意"用于 RLHF
- **A/B 对比**：对比两个版本的链路性能
- **成本分析**：按模型、项目统计 Token 消费和费用
- **数据导出**：将 Trace 导出为数据集用于微调

---

## 5. LangServe 部署

**LangServe** 将 LangChain 的 Chain/Runnable 一键部署为 REST API。

### 5.1 快速部署

```python
# server.py
from fastapi import FastAPI
from langserve import add_routes
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o")

app = FastAPI(title="My LLM API")

add_routes(
    app,
    model,
    path="/chat",
)
```

```bash
langserve start server:app
# API 自动可用：http://localhost:8000
```

### 5.2 自动生成的功能

| 端点 | 说明 |
|------|------|
| `POST /chat/invoke` | 同步调用 |
| `POST /chat/stream` | SSE 流式输出 |
| `POST /chat/batch` | 批量处理 |
| `POST /chat/astream_events` | 异步事件流 |
| `GET /chat/playground` | 自动生成的 Web 测试页面 |
| `GET /chat/input_schema` | 输入 JSON Schema |
| `GET /chat/output_schema` | 输出 JSON Schema |
| `GET /docs` | Swagger API 文档 |

### 5.3 生产化配置

```python
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

# 显式定义输入输出 Schema
class QueryInput(BaseModel):
    question: str = Field(description="用户问题")
    user_id: str = Field(description="用户标识")

class QueryOutput(BaseModel):
    answer: str = Field(description="模型回答")
    sources: list = Field(description="引用的来源")

# 配置路由
add_routes(
    app,
    chain.with_types(input_type=QueryInput, output_type=QueryOutput),
    path="/qa",
    config_keys=["configurable"],     # 允许运行时传入配置
    enable_feedback_endpoint=True,      # 启用用户反馈收集
    enable_public_trace_link_endpoint=True,  # 公开 Trace 链接
)
```

### 5.4 LangSmith Deployment（最新）

LangChain 最新推出了 **LangSmith Deployment**，一个专为长期运行、有状态工作流设计的部署平台：

- **原生 LangGraph 支持**：直接部署 LangGraph 编译后的图
- **持久化执行**：自动持久化状态，支持中断恢复
- **Cron 任务**：支持定时触发 Agent 运行
- **Webhook 触发**：通过 HTTP 回调触发工作流
- **水平伸缩**：生产级可扩展性

---

## 6. LlamaIndex 核心概念

**LlamaIndex**（原名 GPT Index）是一个专为 **LLM 数据增强** 设计的"数据框架"，核心目标是帮助 LLM 高效连接、索引和查询**私有数据**。

### 6.1 Document 与 Node

#### Document（文档）

Document 是 LlamaIndex 中**数据的最小入口单元**，代表一个完整的数据源：

```python
from llama_index.core import Document

doc = Document(
    text="这是一段需要被索引的文本内容。",
    metadata={
        "source": "knowledge_base/tech_doc.md",
        "author": "张三",
        "date": "2024-05-01",
        "page": 5,
    },
    doc_id="doc_001",
    excluded_llm_metadata_keys=["author"],   # 不发给 LLM 的元数据
    excluded_embed_metadata_keys=["date"],   # 不传给 Embedding 的元数据
)
```

#### Node（节点）

Node 是 Document 经过**解析和分块**后生成的更细粒度单元，是索引和检索的**基本单位**：

```
Document ──(解析/分块)──▶ [Node_1, Node_2, Node_3, ...]

每个 Node:
  - text: "分块后的文本片段"
  - metadata: {继承自 Document + 分块位置信息}
  - relationships: {PARENT → Document, PREVIOUS → Node_1, NEXT → Node_3}
  - embedding: [0.023, -0.451, ...]  (向量)
```

```python
from llama_index.core.schema import Node, TextNode, ImageNode

# 文本节点
text_node = TextNode(
    text="Transformer架构由Vaswani等人在2017年提出...",
    metadata={"chunk": 3, "page": 5},
)

# 图片节点（多模态场景）
image_node = ImageNode(
    image_path="figures/transformer.png",
    metadata={"caption": "Transformer架构图"},
)
```

#### 关键区别

| 维度 | Document | Node |
|------|----------|------|
| 粒度 | 粗（完整文档） | 细（文本块） |
| 用途 | 数据加载入口 | 索引和检索的基本单元 |
| 关系 | 可能被拆分为多个 Node | 通过 `relationships` 关联回 Document |
| 索引 | 不直接索引 | 嵌入向量并存入索引 |

---

### 6.2 Index（索引）

Index 是 LlamaIndex 中**组织和管理 Node** 的数据结构，决定了检索的方式和效率。

#### 索引类型对比

```python
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex,
)

# 1. VectorStoreIndex — 向量索引（最常用）
index = VectorStoreIndex.from_documents(
    documents,
    embed_model="local:BAAI/bge-small-zh-v1.5",  # 指定嵌入模型
)

# 2. SummaryIndex — 摘要索引（适合顺序处理）
index = SummaryIndex.from_documents(documents)

# 3. TreeIndex — 树形索引（层级聚类）
index = TreeIndex.from_documents(documents)

# 4. KeywordTableIndex — 关键词表索引
index = KeywordTableIndex.from_documents(documents)

# 5. KnowledgeGraphIndex — 知识图谱索引
index = KnowledgeGraphIndex.from_documents(documents)
```

#### 索引选型指南

| 索引类型 | 检索方式 | 适用场景 | 优势 | 劣势 |
|----------|----------|----------|------|------|
| **VectorStoreIndex** | 语义相似度 | 通用 RAG（最推荐） | 语义理解强、成熟 | 需要 Embedding 成本 |
| **SummaryIndex** | 顺序遍历 | 文档摘要、完整提取 | 不丢失信息 | 规模大时慢 |
| **TreeIndex** | 树形遍历 | 长文档批处理 | 高效处理大量文档 | 构建慢 |
| **KeywordTableIndex** | 关键词匹配 | 精确关键词查询 | 快速、精确 | 无语义理解 |
| **KnowledgeGraphIndex** | 图谱遍历 | 关系推理、多跳问答 | 支持复杂关系 | 构建复杂、成本高 |

#### 索引持久化

```python
# 保存索引到磁盘
index.storage_context.persist(persist_dir="./storage")

# 从磁盘加载
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

#### 索引组合模式

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.composability import ComposableGraph

# 多个子索引组合
graph = ComposableGraph.from_indices(
    root_index_cls=KeywordTableIndex,
    children_indices=[doc_index, code_index, faq_index],
    index_summaries=["技术文档", "代码仓库", "常见问题"],
)
```

---

### 6.3 QueryEngine（查询引擎）

QueryEngine 是 LlamaIndex 对外的**统一查询接口**，封装了检索+生成的全流程。

#### 内置 QueryEngine

```python
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    SubQuestionQueryEngine,
    RouterQueryEngine,
    TransformQueryEngine,
    CitationQueryEngine,
)

# 1. 基础查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,               # 检索 Top-K
    response_mode="compact",          # 响应模式
    streaming=True,                   # 流式输出
)

# 2. 子问题查询引擎 — 分解复杂问题
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(query_engine=doc_engine, metadata=ToolMetadata(name="docs", description="技术文档")),
    QueryEngineTool(query_engine=code_engine, metadata=ToolMetadata(name="code", description="代码库")),
]

sq_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    llm=llm,
)

# 3. 路由查询引擎 — 根据问题类型分发
from llama_index.core.selectors import LLMSingleSelector
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
)
```

#### Response Mode（响应模式）

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `compact` | 尽可能拼接上下文后一次性生成 | 短上下文（推荐） |
| `refine` | 用第一个块生成初步回答，逐块精炼 | 需要综合多个片段 |
| `tree_summarize` | 递归两两合并生成 | 大量文档 |
| `simple_summarize` | 截断后直接生成 | 简单场景 |
| `accumulate` | 逐块追加生成 | 需要覆盖全部内容 |
| `no_text` | 只返回检索结果不生成 | 仅需检索 |

#### 高级配置

```python
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="compact",
    # 生成参数
    temperature=0.1,
    max_tokens=1024,
    # 节点后处理
    node_postprocessors=[
        SentenceEmbeddingOptimizer(embed_model=embed_model, percentile_cutoff=0.5),
    ],
    # 提示词模板
    text_qa_template=qa_prompt,
    refine_template=refine_prompt,
)
```

---

## 7. 数据连接器

LlamaIndex 的数据连接器（Data Connectors / Readers）是其**最大优势之一**，提供开箱即用的 160+ 数据源接入能力。

### 7.1 SimpleDirectoryReader（最常用）

```python
from llama_index.core import SimpleDirectoryReader

# 自动检测并加载一个目录中的所有支持格式
documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,              # 递归子目录
    required_exts=[".pdf", ".md", ".docx"],  # 只加载指定格式
    exclude=["*.tmp", "~*"],     # 排除文件
    file_metadata=lambda path: {"filename": path},  # 自定义元数据
).load_data()
```

### 7.2 支持的数据源类型

| 类别 | 示例 |
|------|------|
| **文档格式** | PDF, Word, Markdown, HTML, PPT, Excel, CSV, JSON |
| **数据库** | MySQL, PostgreSQL, MongoDB, Elasticsearch, Pinecone, ChromaDB |
| **云存储** | AWS S3, GCS, Azure Blob, Dropbox |
| **SaaS 平台** | Notion, Slack, Discord, Google Docs, Confluence, Jira |
| **API/网站** | RSS Feed, Web Scraper, YouTube Transcript, Wikipedia |
| **代码仓库** | GitHub, GitLab, Bitbucket |
| **多模态** | 图片（含 OCR）、音频（含转录）、视频 |

### 7.3 专用 Reader 示例

```python
# PDF 解析（支持表格、图片提取）
from llama_index.readers.file import PDFReader
parser = PDFReader()

# Notion 集成
from llama_index.readers.notion import NotionPageReader
reader = NotionPageReader(integration_token="secret_...")

# 数据库连接
from llama_index.readers.database import DatabaseReader
reader = DatabaseReader(
    engine="postgresql",
    host="localhost",
    database="mydb",
    user="user",
    password="pass",
)

# 网页抓取
from llama_index.readers.web import SimpleWebPageReader
reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://example.com"])

# YouTube 字幕
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
reader = YoutubeTranscriptReader()
docs = reader.load_data(ytlinks=["https://youtube.com/watch?v=..."])
```

### 7.4 自定义数据连接器

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

class MyCustomReader(BaseReader):
    def load_data(self, *args, **kwargs) -> List[Document]:
        # 自定义逻辑
        documents = []
        # ... 从任何数据源加载 ...
        return documents
```

### 7.5 LlamaParse（高级文档解析）

LlamaIndex 的云服务 **LlamaParse** 提供企业级文档解析能力：

- **130+ 格式支持**：PDF、扫描件、图片、Office 文档等
- **表格提取**：准确识别和提取复杂表格
- **布局保持**：保留 Markdown 结构，包括标题、列表、图片
- **OCR 能力**：支持扫描件和包含图片的 PDF
- **Agentic 解析**：用 Agent 自我修正解析结果

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-...",
    result_type="markdown",         # 输出 Markdown
    parsing_instruction="提取所有技术参数表格",
    premium_mode=True,              # 高精度模式
)

documents = parser.load_data("complex_report.pdf")
```

---

## 8. 高级检索

LlamaIndex 提供了丰富的**检索增强策略**，远超过简单的向量相似度检索。

### 8.1 检索模式

```python
from llama_index.core.retrievers import VectorIndexRetriever

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    vector_store_query_mode="default",  # default | sparse | hybrid | text_search
)
```

| 模式 | 机制 | 说明 |
|------|------|------|
| `default` | Dense 向量相似度 | 语义匹配（Embedding 搜） |
| `sparse` | 稀疏向量 (BM25) | 关键词精确匹配 |
| `hybrid` | Dense + Sparse 混合 | 语义 + 关键词融合 |
| `text_search` | 纯文本搜索 | 不使用向量 |

### 8.2 高级检索器类型

```python
# 1. 自动合并检索器 — 检索小块，自动向上合并到大块
from llama_index.core.indices.managed.vectara import VectaraIndex

# 2. 递归检索器 — 多级索引，先找文档再找具体片段
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexAutoRetriever

retriever = VectorIndexAutoRetriever(
    index=index,
    vector_store_info=vector_store_info,
    similarity_top_k=10,
    empty_query_top_k=10,
    verbose=True,
)

# 3. 融合检索器 — 合并多个检索结果
from llama_index.core.retrievers import QueryFusionRetriever

fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, keyword_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,                         # 生成4个变体查询
    mode="reciprocal_rerank",              # 使用 RRF 融合
    use_async=True,                        # 并行执行
)
```

### 8.3 Node Postprocessor（检索后处理）

检索后的 Node 需要经过一系列后处理器进行**过滤、排序、压缩**：

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,       # 按相似度过滤
    KeywordNodePostprocessor,      # 按关键词过滤
    SentenceTransformerRerank,     # Cross-Encoder 重排序
    LongContextReorder,            # 长上下文重排序
    SentenceEmbeddingOptimizer,    # 句子级嵌入优化（去噪）
    MetadataReplacementPostProcessor,  # 元数据替换
)

# 示例：完整的后处理流水线
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.7),  # 过滤低分节点
    KeywordNodePostprocessor(
        required_keywords=["transformer"],            # 必须包含关键词
        exclude_keywords=["deprecated"],              # 排除某些关键词
    ),
    SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # 重排序
        top_n=5,
    ),
    LongContextReorder(),                              # 优化长上下文顺序
]
```

### 8.4 高级检索策略

#### 混合检索（Hybrid Search）

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Dense + Sparse 混合
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=10,
)
vector_retriever = index.as_retriever(similarity_top_k=10)
```

#### 多步检索（Recursive Retrieval）

```
用户查询 → 第一步：检索相关文档摘要 → 第二步：在选定的文档中精检索具体片段
```

#### 时间衰减检索

```python
from llama_index.core.postprocessor import TimeWeightedPostprocessor

postprocessor = TimeWeightedPostprocessor(
    time_decay=0.99,          # 时间衰减因子
    time_access_refresh=True, # 访问后刷新时间
)
```

---

## 9. LlamaIndex Agent 集成

LlamaIndex 提供了自己的 Agent 系统，可以与查询引擎、检索器、工具无缝集成，也支持与 LangChain Agent 互操作。

### 9.1 LlamaIndex 原生 Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool

# 定义工具
def multiply(a: int, b: int) -> int:
    """将两个整数相乘并返回结果"""
    return a * b

def add(a: int, b: int) -> int:
    """将两个整数相加并返回结果"""
    return a + b

# 创建工具对象
math_tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=add),
]

# 创建查询引擎工具
query_tool = QueryEngineTool.from_defaults(
    query_engine=your_query_engine,
    name="knowledge_base",
    description="搜索内部知识库",
)

# 创建 ReAct Agent
agent = ReActAgent.from_tools(
    tools=math_tools + [query_tool],
    llm=llm,
    verbose=True,
    max_iterations=10,
)

# 运行
response = agent.chat("用乘法计算5*7，然后在知识库中搜索相关文档")
```

### 9.2 Agent 类型

| Agent 类型 | 推理模式 | 适用场景 |
|------------|----------|----------|
| **ReActAgent** | 思考-行动-观察循环 | 通用型，可解释性强 |
| **OpenAIAgent** | 原生 Function Calling | 高效，仅 OpenAI 模型 |
| **FunctionCallingAgent** | 通用 Function Calling | 跨模型厂商 |
| **StructuredPlannerAgent** | 先计划后执行 | 复杂多步骤任务 |

### 9.3 LlamaAgents（分布式 Agent）

LlamaIndex 的 **LlamaAgents** 框架支持将 Agent 拆分为微服务：

```
┌──────────────────────────────────────────┐
│             LlamaAgents 架构              │
│                                          │
│  ┌──────────────┐   ┌──────────────┐     │
│  │   Agent A    │   │   Agent B    │     │
│  │ (文档分析)    │   │ (代码生成)    │     │
│  └──────┬───────┘   └──────┬───────┘     │
│         │                  │             │
│         └──────┬───────────┘             │
│                ▼                         │
│     ┌─────────────────────┐              │
│     │   Message Queue     │              │
│     │   (消息队列)         │              │
│     └─────────┬───────────┘              │
│               ▼                          │
│     ┌─────────────────────┐              │
│     │   Orchestrator      │              │
│     │   (编排器)           │              │
│     └─────────────────────┘              │
└──────────────────────────────────────────┘
```

### 9.4 LlamaIndex + LangChain 互操作

```python
# LlamaIndex 检索器 → LangChain 兼容
from llama_index.core.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
    create_llama_chat_agent,
)

# LlamaIndex 工具包装为 LangChain Tool
tool = LlamaIndexTool.from_tool(query_tool)

# 在 LangChain Agent 中使用
from langchain.agents import create_agent
agent = create_agent(
    model=langchain_model,
    tools=[tool, other_langchain_tools],
)

# 反之：LangChain Tool 也可以在 LlamaIndex Agent 中使用
# 通过 LlamaIndex 的 LangChainToolSpec
```

---

## 10. 框架对比与选择建议

### 10.1 定位对比

```
LangChain                          LlamaIndex
─────────────────────────          ────────────────────────
  通用 Agent 工程平台               LLM 数据增强框架
  侧重：编排、推理、工具调用        侧重：索引、检索、数据连接

  "如何让 LLM 做事？"               "如何让 LLM 获取数据？"
```

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **核心定位** | Agent 工程平台 | 数据框架（Data Framework） |
| **设计哲学** | 模块化组件 + 灵活编排 | 以数据为中心的端到端流程 |
| **主要优势** | Agent 编排、多模型互操作、生态丰富 | 数据连接器、索引结构、检索优化 |
| **学习曲线** | 中等偏高（概念多、版本变化快） | 中等（API 一致性好） |
| **社区规模** | 最大（100k+ GitHub Stars） | 快速增长（40k+ Stars） |

### 10.2 功能对比

| 功能 | LangChain | LlamaIndex |
|------|-----------|------------|
| **LLM 调用** | ✅ `init_chat_model` 统一接口 | ✅ `Settings.llm` 全局配置 |
| **Prompt 管理** | ✅ ChatPromptTemplate, MessagesPlaceholder | ✅ ChatMessage 模板 |
| **Agent 框架** | ⭐⭐⭐⭐⭐ LangGraph + Deep Agents | ⭐⭐⭐ ReAct / FunctionCalling |
| **RAG 管道** | ⭐⭐⭐ 基础 RAG 链 | ⭐⭐⭐⭐⭐ 丰富索引+检索策略 |
| **数据连接器** | ⭐⭐ 200+ DocumentLoader | ⭐⭐⭐⭐⭐ 160+ Readers, LlamaParse |
| **向量存储** | ✅ 50+ 集成 | ✅ 20+ 集成 |
| **工作流编排** | ⭐⭐⭐⭐⭐ LangGraph 状态图 | ⭐⭐⭐ Workflows（新） |
| **调试/监控** | ⭐⭐⭐⭐⭐ LangSmith | ⭐⭐⭐ 第三方工具 |
| **部署** | ⭐⭐⭐⭐ LangServe + LangSmith Deploy | ⭐⭐ 自行部署 |
| **多模态** | ⭐⭐⭐ 支持图片/音频 | ⭐⭐⭐⭐ 原生 ImageNode, 多模态索引 |
| **结构化提取** | ⭐⭐ 通过 PydanticOutputParser | ⭐⭐⭐⭐ LlamaExtract |

### 10.3 适用场景建议

#### 选 LangChain / LangGraph 当：

```
✅ 需要构建复杂的 Agent 系统（多步推理、工具编排）
✅ 需要精确控制执行流程（条件分支、循环、并行）
✅ 需要 Human-in-the-Loop（人工审批节点）
✅ 需要多 Agent 协作（Supervisor / Swarm 模式）
✅ 需要生产级部署和监控（LangSmith + LangServe）
✅ 需要多模型供应商灵活切换
✅ 项目强调推理和决策而非数据检索
```

#### 选 LlamaIndex 当：

```
✅ 核心需求是构建 RAG 系统（检索增强生成）
✅ 需要连接多种异构数据源（PDF、数据库、API、Notion...）
✅ 需要高级检索策略（混合检索、递归检索、融合检索）
✅ 需要专业的文档解析（表格、扫描件、复杂排版）
✅ 需要多种索引类型（向量、关键词、知识图谱、树形）
✅ 项目以数据/文档为中心
✅ 团队追求开箱即用的体验
```

#### 两者结合使用（常见最佳实践）：

```
┌─────────────────────────────────────────────┐
│              LLM Application                │
│                                             │
│  ┌──────────────────────────────────┐       │
│  │     LlamaIndex                   │       │
│  │  - 数据加载 (160+ Readers)        │       │
│  │  - 文档解析 (LlamaParse)          │       │
│  │  - 索引构建 (多种索引类型)         │       │
│  │  - 高级检索 (混合/递归/融合)       │       │
│  └──────────────┬───────────────────┘       │
│                 │                            │
│                 ▼ (检索结果作为工具)           │
│  ┌──────────────────────────────────┐       │
│  │     LangChain / LangGraph         │       │
│  │  - Agent 编排                    │       │
│  │  - 多步推理与工具调用              │       │
│  │  - 工作流控制 (StateGraph)        │       │
│  │  - 部署与监控 (LangServe/Smith)   │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

### 10.4 技术选型决策树

```
需要什么？
│
├── 主要做 RAG / 文档问答
│   │
│   ├── 数据源复杂多样 → LlamaIndex
│   │
│   ├── 需要复杂工作流控制 → LangChain + LlamaIndex
│   │
│   └── 简单场景 → 两个都可以，LlamaIndex 更快上手
│
├── 主要做 Agent / 工具调用
│   │
│   ├── 流程简单 → LangChain create_agent
│   │
│   ├── 需要精确控制/Human-in-the-loop → LangGraph
│   │
│   └── 需要生产级部署监控 → LangChain + LangSmith
│
├── 既要 RAG 又要 Agent
│   │
│   └── 推荐：LlamaIndex（数据层） + LangGraph（编排层）
│
└── 快速原型验证
    │
    ├── RAG 原型 → LlamaIndex (5行代码)
    │
    └── Agent 原型 → LangChain create_agent (10行代码)
```

### 10.5 最新趋势（2025-2026）

| 趋势 | 说明 |
|------|------|
| **LangChain 全面 Agent 化** | 从 Chain 框架转型为 Agent 平台，Deep Agents 是新一代旗舰 |
| **LlamaIndex 多模态化** | 从纯文本扩展到图片、音频、视频的索引和检索 |
| **两框架互操作性增强** | 工具可以互相调用，Agent 可以跨框架协作 |
| **MCP 协议支持** | 两者都支持 Model Context Protocol，统一工具标准 |
| **LlamaParse 生态** | 企业级文档解析成为 LlamaIndex 的核心竞争力 |
| **LangSmith 平台化** | 从调试工具发展为完整的 LLMOps 平台（评估+部署+监控） |

---

## 参考资源

### LangChain

- **官方文档**：https://docs.langchain.com
- **API 参考**：https://reference.langchain.com/python
- **LangChain Academy（免费课程）**：https://academy.langchain.com
- **GitHub**：https://github.com/langchain-ai/langchain
- **LangSmith**：https://smith.langchain.com
- **社区论坛**：https://forum.langchain.com

### LlamaIndex

- **官方文档**：https://developers.llamaindex.ai
- **LlamaHub（集成中心）**：https://llamahub.ai
- **LlamaParse**：https://cloud.llamaindex.ai
- **GitHub**：https://github.com/run-llama/llama_index
- **Discord 社区**：https://discord.gg/dGcwcsnxhU
- **Reddit**：https://reddit.com/r/LlamaIndex

### 中文资源

- **知乎专栏 - LangChain 实战**：搜索"LangChain 教程"
- **CSDN - LlamaIndex 系列**：搜索"LlamaIndex RAG"
- **B站 - 大模型应用开发**：LangChain + LlamaIndex 实战视频

---

> **文档版本**：v1.0 | **最后更新**：2025年5月 | **目标读者**：中级 LLM 应用开发者
