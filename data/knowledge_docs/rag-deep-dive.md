# RAG（检索增强生成）中级深度指南

> 本文档为中级难度，面向已有大模型基础、正在学习应用开发的读者。内容涵盖 RAG 原理与架构、文档解析分块、Embedding 模型选型、向量数据库对比、检索优化、高级 RAG 模式及评估方法。

---

## 目录

1. [RAG 原理与架构](#1-rag-原理与架构)
2. [文档解析与分块策略](#2-文档解析与分块策略)
3. [Embedding 模型选型](#3-embedding-模型选型)
4. [向量数据库对比](#4-向量数据库对比)
5. [检索优化](#5-检索优化)
6. [高级 RAG 模式](#6-高级-rag-模式)
7. [评估方法](#7-评估方法)
8. [参考资源](#8-参考资源)

---

## 1. RAG 原理与架构

### 1.1 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将**信息检索**与**大语言模型生成**相结合的架构。其核心思想是：**在 LLM 生成回答之前，先从外部知识库中检索相关文档片段，将这些片段作为上下文注入 Prompt，让模型基于检索到的"证据"进行生成。**

这解决了 LLM 的三大固有问题：

| 问题 | RAG 的解决方式 |
|------|---------------|
| **知识截止日期** | 实时检索外部知识库，无需重新训练 |
| **幻觉（Hallucination）** | 用检索到的真实文档约束生成，可溯源 |
| **领域知识不足** | 注入私有/专业文档，无需微调模型 |

### 1.2 标准 RAG 架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   用户查询    │────▶│  查询重写/   │────▶│  Embedding   │
│   (Query)    │     │  扩展        │     │  模型编码    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│   Prompt     │◀────│  向量数据库   │
│  最终回答    │     │  拼接        │     │  相似度检索   │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### 离线阶段（Indexing）

1. **文档加载**：读取 PDF、Word、Markdown、网页等多种格式
2. **文档解析**：提取纯文本、表格、图片描述等结构化信息
3. **文本分块（Chunking）**：将长文档切分为合适大小的片段
4. **向量化（Embedding）**：用 Embedding 模型将每个 chunk 转为向量
5. **存储索引**：将向量和原始文本存入向量数据库

#### 在线阶段（Retrieval & Generation）

1. **查询处理**：对用户 query 进行重写、扩展或分解
2. **查询向量化**：将 query 用同一 Embedding 模型编码
3. **相似度检索**：在向量数据库中找到 Top-K 最相似的 chunks
4. **后处理**：重排序（Reranking）、去重、过滤
5. **Prompt 拼接**：将检索结果与 query 组装成 Prompt
6. **LLM 生成**：调用大模型生成带引用的最终回答

### 1.3 进阶架构：模块化 RAG

现代 RAG 系统通常采用模块化设计，各组件可独立升级：

- **Query Processing Module**：意图识别、查询重写、HyDE（假设文档嵌入）
- **Retrieval Module**：稀疏检索（BM25）+ 稠密检索（Dense）混合
- **Reranking Module**：Cross-encoder 精排
- **Post-Processing Module**：去重、上下文压缩、引用标注
- **Generation Module**：Prompt 模板、思维链（CoT）、自我反思

---

## 2. 文档解析与分块策略

### 2.1 文档解析

#### 常见文档格式处理

| 格式 | 推荐工具 | 注意事项 |
|------|---------|---------|
| PDF | PyMuPDF (fitz), Unstructured, PDFPlumber | 扫描件需 OCR（Tesseract/PaddleOCR） |
| Word | python-docx, Unstructured | 保留层级结构 |
| Markdown | 原生解析 | 代码块与表格需特殊处理 |
| HTML | BeautifulSoup, Unstructured | 去噪（导航栏、广告） |
| 图片/扫描件 | PaddleOCR, Tesseract | 中文推荐 PaddleOCR |
| PPT | python-pptx | 提取文本框和备注 |

#### 核心挑战

- **复杂布局**：多栏 PDF、表格、图片混排
- **表格处理**：需转为 Markdown 表格或结构化 JSON
- **层级结构**：保留标题层级（H1-H6）以辅助分块
- **元数据提取**：作者、日期、来源 URL 等

> **推荐工具链**：[Unstructured](https://github.com/Unstructured-IO/unstructured) + [LlamaIndex](https://github.com/run-llama/llama_index) 的 IngestionPipeline 或 [LangChain](https://github.com/langchain-ai/langchain) 的 Document Loaders。

### 2.2 分块策略（Chunking Strategies）

分块是 RAG 中最关键的超参数之一，直接影响检索质量。

#### 2.2.1 固定大小分块（Fixed-size Chunking）

```python
# 最简单：按字符数拆分，可设重叠
chunk_size = 512   # 字符数
chunk_overlap = 50 # 重叠字符数
```

- ✅ 简单、计算效率高
- ❌ 可能切断句子或语义单元

#### 2.2.2 语义分块（Semantic Chunking）

基于句子边界和语义相似度进行拆分：

```python
# 计算相邻句子的 Embedding 相似度
# 在相似度"断点"处切分
breakpoint_threshold = 0.8  # 相邻句相似度低于此值时分块
```

- ✅ 保持语义完整性
- ❌ 计算量较大，需要额外的 Embedding 调用

#### 2.2.3 递归分块（Recursive Chunking）

LangChain 默认策略，按分隔符优先级依次尝试拆分：

```
分隔符优先级：["\n\n", "\n", "。", ".", " ", ""]
```

- ✅ 尽量在自然边界切分
- ✅ 对 Markdown/代码友好

#### 2.2.4 结构感知分块（Structure-aware）

按文档结构（标题、章节）分块：

- Markdown：按 `#` 标题层级切分
- HTML：按 DOM 标签切分
- PDF：按章节/段落检测

#### 2.2.5 小2大（Small-to-Big） / 父子块

```
┌─────────────────────────────┐  ← 父块 (Parent Chunk, ~1024 tokens)
│  ┌───────────────────────┐  │
│  │   子块 (Child, 256t)  │  │  ← 用于检索（索引小粒度）
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │   子块 (Child, 256t)  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

**检索时用小粒度（子块），生成时返回大粒度（父块）**，兼顾检索精度和上下文完整性。

#### 2.2.6 分块大小选择指南

| 场景 | 推荐 Chunk Size | 推荐 Overlap |
|------|----------------|-------------|
| 问答（短事实） | 256-512 tokens | 10-15% |
| 摘要/长文档 | 1000-2000 tokens | 10-15% |
| 代码检索 | 按函数/类边界 | N/A |
| 多语言混合 | 512-768 tokens | 15-20% |
| 知识库 FAQ | 128-256 tokens | 0-5% |

> **核心原则**：Chunk 大小应匹配你的"最小检索单元"。太小丢失上下文，太大稀释语义信号。

---

## 3. Embedding 模型选型

### 3.1 主流中文 Embedding 模型

| 模型 | 维度 | 最大长度 | MTEB-C 排名 | 特点 |
|------|------|---------|------------|------|
| **BGE-M3** (BAAI) | 1024 | 8192 | ⭐⭐⭐⭐⭐ | 多语言、稀疏+稠密、长文本 |
| **BGE-Large-ZH** (BAAI) | 1024 | 512 | ⭐⭐⭐⭐⭐ | 中文SOTA，但最大512 tokens |
| **GTE-Qwen2-7B** (Alibaba) | 3584 | 32768 | ⭐⭐⭐⭐⭐ | 基于Qwen2，超长文本 |
| **GTE-Large-ZH** (Alibaba) | 1024 | 512 | ⭐⭐⭐⭐ | 性价比高 |
| **M3E-Large** (moka-ai) | 1024 | 512 | ⭐⭐⭐⭐ | 开源中文模型，社区活跃 |
| **stella-mrl-large-zh** (infgrad) | 1024 | 1024 | ⭐⭐⭐⭐ | 支持 Matryoshka |
| **text2vec-large-chinese** | 1024 | 512 | ⭐⭐⭐ | 经典中文模型 |
| **Jina Embeddings v3** | 1024 | 8192 | ⭐⭐⭐⭐ | 多语言、任务特定 LoRA |
| **Cohere Embed v3** | 1024 | 512 | ⭐⭐⭐⭐ | 商用API，多语言 |

> 排名参考 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 和 C-MTEB。

### 3.2 选型核心考量

#### 3.2.1 维度 vs 性能权衡

```
高维度 (1536-3584)：精度高，存储和计算成本大
中间维度 (768-1024)：最佳性价比，大多数场景推荐
低维度 (384-512)：速度快，适合边缘设备或大规模场景
```

**Matryoshka Embeddings**：支持从高维向量"截取"前 N 维使用，一个模型满足多种精度/速度需求（如 `stella-mrl-large-zh`）。

#### 3.2.2 最大序列长度

| 你的 Chunk 大小 | 推荐模型最小长度 |
|----------------|----------------|
| ≤ 512 tokens | BGE-Large-ZH, GTE-Large-ZH, M3E-Large |
| 512-2048 tokens | BGE-M3, stella-mrl, Jina v3 |
| > 2048 tokens | GTE-Qwen2-7B（32K）, BGE-M3（8K） |

#### 3.2.3 Query/Document 不对称性

部分模型对 Query 和 Document 使用不同的编码指令：

```python
# BGE 系列需要加 instruction prefix
query_instruction = "为这个句子生成表示以用于检索相关文章："
document_instruction = ""  # 文档侧不加 prefix

# Jina Embeddings 使用 task 参数
# task="retrieval.query" vs task="retrieval.passage"
```

#### 3.2.4 部署建议

| 规模 | 推荐方案 |
|------|---------|
| 小规模（< 10万篇） | CPU + ONNX Runtime / FastEmbed |
| 中规模（10万-100万） | GPU (T4/A10) + Sentence-Transformers |
| 大规模（> 100万） | GPU 集群 / TEI（Text Embeddings Inference） |

- [**TEI (HuggingFace)**](https://github.com/huggingface/text-embeddings-inference)：高性能嵌入推理服务，支持 Flash Attention、动态批处理
- [**Infinity**](https://github.com/michaelfeil/infinity)：同样高性能，支持多种模型格式
- [**FastEmbed**](https://github.com/qdrant/fastembed)：轻量级，适合原型开发

### 3.3 微调 Embedding 模型

当通用模型在特定领域表现不佳时，可以微调：

1. **数据准备**：(Query, Positive, Negative) 三元组
2. **微调范式**：
   - 对比学习（Contrastive Loss）
   - 知识蒸馏（从 Cross-encoder 蒸馏到 Bi-encoder）
   - 使用 LlamaIndex / BGE 官方的微调脚本
3. **常用框架**：`sentence-transformers`、`FlagEmbedding`（BGE 官方）

---

## 4. 向量数据库对比

### 4.1 主流向量数据库一览

| 数据库 | 类型 | 开源 | GPU 加速 | 过滤能力 | 适用规模 | 特点 |
|--------|------|------|---------|---------|---------|------|
| **Milvus** | 分布式向量DB | ✅ | ✅ | 标量+向量混合过滤 | >亿级 | 云原生，功能最全 |
| **Zilliz Cloud** | 云服务 | ❌ | ✅ | 同上 | >亿级 | Milvus 全托管版 |
| **Qdrant** | 向量DB | ✅ | ❌ | 丰富 Payload 过滤 | 百万-亿级 | Rust 编写，性能优异 |
| **Weaviate** | 向量DB | ✅ | ❌ | GraphQL + 过滤 | 百万-亿级 | 内置多模态支持 |
| **Chroma** | 嵌入式向量DB | ✅ | ❌ | 基础元数据过滤 | 百万级 | 极简API，适合原型 |
| **FAISS** (Meta) | 向量检索库 | ✅ | ✅ | 有限 | 十亿级 | 极致性能，不是DB |
| **Elasticsearch** | 全文搜索+向量 | ✅ | ❌ | 强大 | 亿级 | 全文+向量混合 |
| **Vespa** | 综合搜索平台 | ✅ | ❌ | 非常强大 | 十亿级 | 支持结构化+非结构化 |
| **Pinecone** | 云服务 | ❌ | ❌ | 元数据过滤 | 十亿级 | 全托管，零运维 |
| **PGVector** | PostgreSQL扩展 | ✅ | ❌ | SQL 全部过滤能力 | 百万级 | 与现有PG生态集成 |

### 4.2 关键对比维度

#### 4.2.1 FAISS vs 向量数据库

```
FAISS = 向量索引库（需要自己管理元数据、持久化、分布式）
Milvus/Chroma = 向量数据库（一站式管理向量+元数据+索引）
```

**选择 FAISS 当**：需要极致性能、离线批量检索、已有存储方案
**选择向量数据库当**：需要生产级持久化、元数据过滤、水平扩展

#### 4.2.2 选型决策树

```
是否需要全托管服务？
├── 是 → Pinecone / Zilliz Cloud / Weaviate Cloud
└── 否
    ├── 数据量 > 亿级，需要分布式？
    │   ├── 是 → Milvus / Qdrant 集群
    │   └── 否
    │       ├── 已有 PostgreSQL？
    │       │   └── 是 → PGVector
    │       ├── 需要极简 API，快速原型？
    │       │   └── 是 → Chroma
    │       └── 需要全文+向量混合检索？
    │           └── 是 → Elasticsearch / Vespa
```

#### 4.2.3 索引类型选择

| 索引类型 | 速度 | 召回率 | 内存 | 适用场景 |
|---------|------|--------|------|---------|
| **FLAT** | 慢 | 100% | 低 | 精确检索基线 |
| **IVF_FLAT** | 中 | 95-99% | 中 | 通用场景首选 |
| **IVF_PQ** | 快 | 90-95% | 低 | 内存受限场景 |
| **HNSW** | 最快 | 98-99% | 高 | 低延迟在线服务 |
| **DiskANN** | 快 | 95-99% | 极低 | 超大规模+SSD预算 |

> **经验法则**：百万级用 HNSW，亿级用 IVF+PQ，极致大规模用 DiskANN。

### 4.3 性能基准参考

| 场景 | 推荐配置 | 预期 QPS（单机） |
|------|---------|-----------------|
| 原型/小项目 | Chroma + HNSW | 100-500 |
| 中等生产 | Qdrant/Milvus + HNSW + 量化 | 500-2000 |
| 大规模生产 | Milvus 集群 + IVF + GPU | 2000-10000+ |

---

## 5. 检索优化

### 5.1 混合检索（Hybrid Search）

**核心思想**：结合稀疏检索（关键词匹配）和稠密检索（语义匹配）的优势。

| 方法 | 优点 | 缺点 |
|------|------|------|
| **BM25（稀疏）** | 精确匹配、专有名词、领域术语 | 语义理解弱、同义词盲区 |
| **Dense Embedding（稠密）** | 语义理解、多语言、泛化强 | 精确匹配弱、需训练 |

#### 5.1.1 实现方案

```python
# 方案A: 线性加权融合
final_score = alpha * dense_score + (1 - alpha) * sparse_score

# 方案B: 倒数排名融合（RRF）
RRF_score(doc) = Σ 1/(k + rank_i(doc))  # k 常为 60
```

**RRF (Reciprocal Rank Fusion)** 不需要分数归一化，在实践中比线性加权更鲁棒。

#### 5.1.2 工具支持

| 工具 | 混合检索支持 |
|------|------------|
| **Elasticsearch 8.x+** | 原生支持，内置 RRF |
| **Milvus 2.4+** | 支持 BM25 + Dense 混合 |
| **Weaviate** | `hybrid` search 参数 |
| **Qdrant** | 需自行实现组合 |
| **LlamaIndex** | `HybridFusionRetriever` |
| **LangChain** | `EnsembleRetriever` |

#### 5.1.3 BGE-M3 统一模型方案

BGE-M3 单个模型同时输出**稠密向量**和**稀疏向量（lexical weights）**，无需维护两套检索系统：

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 同时输出稠密和稀疏表示
dense_embeddings = model.encode(sentences)['dense_vecs']
sparse_embeddings = model.encode(sentences)['lexical_weights']
```

### 5.2 重排序（Reranking）

**为什么需要 Reranking？**

- 初检索（Bi-encoder）速度快但精度有限
- 重排序（Cross-encoder）精度高但速度慢
- **两阶段策略**：粗筛（Top-K, K=100-200）→ 精排（Top-N, N=5-10）

#### 5.2.1 重排序模型对比

| 模型 | 参数 | 最大长度 | 中文支持 | 特点 |
|------|------|---------|---------|------|
| **BGE-Reranker-v2-m3** | 568M | 8192 | ✅ 多语言 | BGE 系列推荐 |
| **BGE-Reranker-Large** | 326M | 512 | ✅ | 中文 Reranker SOTA |
| **Cohere Rerank v3** | 闭源 | 4096 | ✅ | 商用 API，多语言 |
| **Jina Reranker v2** | 278M | 8192 | ✅ | 多语言，开源 |
| **bce-reranker-base_v1** | 278M | 512 | ✅ | 网易有道开源 |

#### 5.2.2 Reranker 使用示例

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# 计算 query 和每个 document 的相关性分数
scores = reranker.compute_score([
    [query, doc] for doc in retrieved_docs
])

# 按分数排序，取 Top-N
ranked_docs = sorted(
    zip(retrieved_docs, scores), 
    key=lambda x: x[1], 
    reverse=True
)[:top_n]
```

### 5.3 查询优化技术

#### 5.3.1 查询重写（Query Rewriting）

- **LLM 重写**：用 LLM 将口语化查询转为更适合检索的表达
- **多轮对话上下文**：将对话历史压缩为独立的检索查询
- **Step-back Prompting**：先问更抽象的问题，再结合具体问题

#### 5.3.2 查询扩展（Query Expansion）

- **同义词扩展**：用词典或 LLM 生成同义查询
- **HyDE (Hypothetical Document Embeddings)**：
  1. 让 LLM 生成"假设的完美回答文档"
  2. 用该文档的 Embedding 去检索
  3. 效果惊人地好，特别是对模糊查询

#### 5.3.3 多查询检索（Multi-Query Retrieval）

```
原始查询 → LLM → 生成3-5个不同角度的查询 → 分别检索 → 合并去重 → 去重排序
```

#### 5.3.4 查询分解（Query Decomposition）

- 复杂问题拆解为子问题
- 每个子问题独立检索
- 汇总结果后生成最终答案

### 5.4 检索增强技巧汇总

| 技巧 | 效果 | 成本 |
|------|------|------|
| 混合检索（BM25 + Dense） | 召回率 +10-20% | 中 |
| Reranker 精排 | 精准度 +15-25% | 中 |
| HyDE | 模糊查询显著改善 | 高（额外 LLM 调用） |
| 查询重写 | 多轮对话改善 | 中 |
| 多查询检索 | 召回率 +5-15% | 高 |
| Small-to-Big 分块 | 上下文完整性 | 存储成本略增 |
| 元数据过滤 | 精准度提升 | 低 |
| Self-Query 检索 | 结构化+语义联合 | 中 |

---

## 6. 高级 RAG 模式

### 6.1 Self-RAG（自我反思 RAG）

> 论文：[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

#### 核心思想

Self-RAG 训练 LLM 在生成过程中学会：
1. **判断是否需要检索**（Retrieve Token）
2. **评估检索结果的相关性**（Relevance Token）
3. **评估生成内容的支持度**（Supported Token）
4. **评估答案的有用性**（Usefulness Token）

```
生成流程：
Query → [需要检索? 是] → 检索 → [文档相关? 是] → 生成 → [有文献支持? 是] → 输出
       [需要检索? 否] → 直接用参数知识生成
```

#### 关键 Token

```
<Retrieve>：触发检索
<Relevant> / <Irrelevant>：片段相关/不相关
<Fully Supported> / <Partially Supported> / <No Support>：生成内容的支持程度
<Utility: 1-5>：回答有用性评分
```

#### 实践建议

- Self-RAG 需要专门训练的模型（如 `selfrag_llama2_7b`）
- 可用 LangGraph 或 LlamaIndex 实现类似的反思流程
- 核心价值：**减少不必要检索 + 降低幻觉**

---

### 6.2 CRAG（纠正式 RAG）

> 论文：[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)

#### 核心思想

CRAG 引入**检索评估器**，对检索结果进行质量判断：

```
检索结果 → [评估：高质量] → 直接使用
         → [评估：低质量] → 查询重写/扩展 → 重新检索
         → [评估：有噪声] → 精细过滤 + 补充检索
```

#### CRAG 三大动作

| 动作 | 条件 | 操作 |
|------|------|------|
| **Correct** | 检索质量低 | 查询重写、知识精炼 |
| **Incorrect** | 检索结果不相关 | 扩大搜索范围、外部搜索 |
| **Ambiguous** | 结果部分相关 | 精细化过滤 + 补充检索 |

#### 四阶段流程

1. **检索**：标准检索召回 Top-N
2. **评估**：检索评估器（T5-large 微调）计算置信度
3. **纠偏**：根据评估结果调整检索策略
4. **生成**：基于最终检索结果生成答案

#### 实践要点

- 评估器的质量是关键瓶颈
- 可用 LLM-as-Judge 替代专门的评估模型
- 适合**知识密集**且**检索质量波动大**的场景

---

### 6.3 GraphRAG（图增强 RAG）

> 微软开源：[GraphRAG](https://github.com/microsoft/graphrag)

#### 核心思想

在向量检索基础上，引入**知识图谱**来捕获实体间的关系：

```
传统 RAG：Query → 语义相似 chunks → 拼接 → 生成
GraphRAG：Query → 语义搜索 + 图谱遍历 → 结构化上下文 → 生成
```

#### GraphRAG 工作流

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  文档        │────▶│  实体/关系   │────▶│  知识图谱    │
│             │     │  提取(GLEAN) │     │  构建        │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
┌─────────────┐     ┌──────────────┐     ┌───────▼───────┐
│  最终答案    │◀────│  LLM 汇总    │◀────│  社区检测 +   │
│             │     │              │     │  图遍历检索    │
└─────────────┘     └──────────────┘     └──────────────┘
```

#### 两种检索模式

| 模式 | 适用问题 | 方法 |
|------|---------|------|
| **Local Search** | 特定实体相关问题 | 实体为中心的邻居子图 |
| **Global Search** | 摘要性问题（"主要主题是什么？"） | 社区摘要汇总 |

#### GraphRAG 的优劣势

- ✅ 擅长回答需要**关系推理**和**全局理解**的问题
- ✅ 能发现知识库中隐藏的关联
- ❌ 构建图谱成本高（LLM 提取实体+关系）
- ❌ 对简单事实检索可能过度设计

#### 其他图增强 RAG 方案

- **Knowledge Graph RAG (LlamaIndex)**：支持 Neo4j 等图数据库
- **G-Retriever**：直接用图神经网络做检索
- **LightRAG**：更轻量的图谱 RAG 实现

---

### 6.4 其他高级模式

#### 6.4.1 Agentic RAG（智能体 RAG）

```
不是"一次性检索+生成"
而是"Agent 决定何时检索、检索什么、如何迭代"
```

使用 LangGraph / CrewAI 构建多步骤 RAG Agent：
- 路由（Router）：根据问题类型选择不同检索策略
- 多步推理：检索→分析→再检索→综合
- 工具使用：结合检索 + 搜索 + 计算器等

#### 6.4.2 RAPTOR（递归摘要 RAG）

- 对文档建立**树状摘要结构**
- 检索时可在不同抽象层级匹配
- 适合长文档的多粒度理解

#### 6.4.3 ColBERT 风格的 Token-level 检索

- 不是对整个文档做单向量编码
- 而是为每个 Token 编码，做 Token-level 匹配
- 在**代码搜索**和**精确匹配**场景表现出色

#### 6.4.4 高级模式选择指南

| 场景 | 推荐模式 |
|------|---------|
| 简单 FAQ | 标准 RAG + 混合检索 |
| 知识密集型问答 | Self-RAG / CRAG |
| 多跳推理 | Agentic RAG / GraphRAG |
| 全局摘要 | GraphRAG (Global Search) |
| 长文档 QA | RAPTOR |
| 代码搜索 | ColBERT / Agentic RAG |
| 多轮对话 | 查询重写 + Self-RAG |

---

## 7. 评估方法

### 7.1 评估维度

RAG 评估需要同时覆盖**检索质量**和**生成质量**：

#### 检索评估（Retrieval Metrics）

| 指标 | 公式 / 含义 | 说明 |
|------|-----------|------|
| **Hit Rate (HR@K)** | 正确答案是否在 Top-K 中 | 最基础，二进制 |
| **MRR (Mean Reciprocal Rank)** | 1/正确答案的排名 的平均值 | 考虑排名位置 |
| **NDCG@K** | 归一化折损累计增益 | 考虑排名+相关性等级 |
| **Recall@K** | 检索出的相关文档 / 所有相关文档 | 覆盖率 |
| **Precision@K** | Top-K 中相关比例 | 准确率 |

#### 生成评估（Generation Metrics）

| 指标 | 说明 | 局限 |
|------|------|------|
| **Faithfulness** | 生成内容是否忠于检索上下文 | 需要标注或 LLM-as-Judge |
| **Answer Relevance** | 答案是否贴合问题 | 同上 |
| **Context Relevance** | 检索内容与问题的相关度 | 需评估 |
| **ROUGE** | n-gram 重叠度 | 不衡量语义质量 |
| **BLEU** | 翻译/生成质量 | 对 QA 不够有效 |
| **BERTScore** | 基于 BERT 的语义相似度 | 不评估幻觉 |

### 7.2 评估框架

#### 7.2.1 RAGAS

> [GitHub](https://github.com/explodinggradients/ragas)

**最流行的 RAG 评估框架**，提供：

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

| RAGAS 指标 | 衡量什么 | 需要什么 |
|-----------|---------|---------|
| Faithfulness | 答案是否基于上下文 | query, answer, contexts |
| Answer Relevancy | 答案是否贴合问题 | query, answer |
| Context Precision | 检索到的上下文是否与问题相关 | query, contexts, reference |
| Context Recall | 是否检索到了所有相关上下文 | query, contexts, reference |
| Answer Correctness | 答案准确性 | query, answer, ground_truth |
| Aspect Critique | 自定义评估维度 | query, answer, (rubric) |

#### 7.2.2 TruLens

> [GitHub](https://github.com/truera/trulens)

- 支持 RAG 三合一评估（Answer Relevance, Context Relevance, Groundedness）
- 提供可视化 Dashboard
- 集成 LangChain / LlamaIndex

#### 7.2.3 ARES

> [GitHub](https://github.com/stanford-futuredata/ARES)

- 斯坦福出品，全自动化评估流水线
- 使用合成数据训练评估模型
- 适合大规模评估

#### 7.2.4 DeepEval

> [GitHub](https://github.com/confident-ai/deepeval)

- 单元测试风格的评估框架
- 内置 RAG 评估指标
- 易于集成 CI/CD

### 7.3 评估数据集

#### 7.3.1 通用 QA 数据集

| 数据集 | 语言 | 规模 | 适用场景 |
|--------|------|------|---------|
| Natural Questions | 英文 | ~30万 | 开放域 QA |
| HotpotQA | 英文 | ~11万 | 多跳推理 |
| MS MARCO | 英文 | ~100万 | 段落检索 |
| BEIR | 英文 | 18 个任务 | 零样本检索 |
| DuReader | 中文 | ~30万 | 中文阅读理解 |
| C-MTEB | 中文 | 多任务 | 中文检索/Embedding |

#### 7.3.2 构建自有评估集

**最小可行方案**：

```python
# 构造 30-50 条标注好的 QA 对
eval_data = [
    {
        "question": "RAG的全称是什么？",
        "ground_truth": "Retrieval-Augmented Generation",
        "relevant_docs": ["doc_1", "doc_5"]  # 可选：相关文档标注
    },
    # ... 更多
]
```

关键原则：
- 覆盖**不同难度**的问题（简单事实、推理、对比）
- 包含**对抗样本**（问题中包含不存在的知识）
- 标注**相关文档**（评估检索质量）

### 7.4 评估流水线

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 测试集   │───▶│ RAG 系统 │───▶│ 收集所有  │───▶│ 计算指标  │
│ 执行     │    │ 端到端   │    │ 中间结果  │    │ 生成报告  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                     │
                     ┌───────────────────────────────┘
                     ▼
              ┌──────────────┐
              │ 持续监控 +   │
              │ 人工 Review  │
              └──────────────┘
```

**中间结果记录**（调试必备）：

```python
{
    "query": "...",
    "rewritten_query": "...",
    "retrieved_chunks": [...],
    "reranked_chunks": [...],
    "final_prompt": "...",
    "generated_answer": "...",
    "ground_truth": "...",
    "metrics": {}
}
```

---

## 8. 参考资源

### 论文

| 论文 | 链接 | 核心贡献 |
|------|------|---------|
| RAG 原始论文 | [arXiv 2005.11401](https://arxiv.org/abs/2005.11401) | RAG 概念提出 |
| Self-RAG | [arXiv 2310.11511](https://arxiv.org/abs/2310.11511) | 自我反思 RAG |
| CRAG | [arXiv 2401.15884](https://arxiv.org/abs/2401.15884) | 纠正式 RAG |
| GraphRAG | [arXiv 2404.16130](https://arxiv.org/abs/2404.16130) | 图增强 RAG |
| RAPTOR | [arXiv 2401.18059](https://arxiv.org/abs/2401.18059) | 递归摘要 RAG |
| HyDE | [arXiv 2212.10496](https://arxiv.org/abs/2212.10496) | 假设文档嵌入 |
| ColBERT | [arXiv 2004.12832](https://arxiv.org/abs/2004.12832) | Token 级检索 |
| RAGAS 评估 | [arXiv 2309.15217](https://arxiv.org/abs/2309.15217) | 自动化 RAG 评估 |

### 开源项目

| 项目 | GitHub | 用途 |
|------|--------|------|
| **LangChain** | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | RAG 编排框架 |
| **LlamaIndex** | [run-llama/llama_index](https://github.com/run-llama/llama_index) | 数据索引与 RAG |
| **Dify** | [langgenius/dify](https://github.com/langgenius/dify) | 可视化 RAG 应用平台 |
| **FastGPT** | [labring/FastGPT](https://github.com/labring/FastGPT) | 中文 RAG 应用搭建 |
| **RAGFlow** | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 深度文档理解 RAG |
| **MaxKB** | [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB) | 中文知识库问答 |
| **GraphRAG** | [microsoft/graphrag](https://github.com/microsoft/graphrag) | 微软图增强 RAG |
| **Unstructured** | [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | 文档解析预处理 |
| **BGE** | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | BGE 系列模型 |
| **Qdrant** | [qdrant/qdrant](https://github.com/qdrant/qdrant) | 高性能向量数据库 |
| **Milvus** | [milvus-io/milvus](https://github.com/milvus-io/milvus) | 分布式向量数据库 |
| **Chroma** | [chroma-core/chroma](https://github.com/chroma-core/chroma) | 嵌入式向量数据库 |

### 学习资源

- **LangChain RAG 教程**：https://python.langchain.com/docs/tutorials/rag/
- **LlamaIndex RAG 指南**：https://docs.llamaindex.ai/en/stable/understanding/rag/
- **RAG 技术全栈指南**：https://www.pinecone.io/learn/series/rag/
- **arXiv RAG 论文列表**：https://github.com/Tongji-KGLLM/RAG-Survey
- **NVIDIA RAG 最佳实践**：https://developer.nvidia.com/blog/tag/rag/
- **阿里云 RAG 技术文档**：https://help.aliyun.com/zh/model-studio/rag/

---

> **写在最后**：RAG 不是一个"拿来即用"的标准方案，而是一套需要根据场景精心调校的工具箱。分块策略、Embedding 模型、检索参数、Prompt 模板——每一个选择都会影响最终效果。建议从**最简单的流水线开始**，用 RAGAS 建立评估基线，然后逐步引入混合检索、重排序、查询优化等进阶技术，用数据驱动迭代。

---

*文档版本：v1.0 | 生成日期：2026-05-17 | 面向读者：LLM 应用开发中级开发者*
