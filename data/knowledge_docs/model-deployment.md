# 大模型部署与服务化 — 中级实战指南

> **适用读者**：有 PyTorch/HuggingFace 基础，希望在本地或服务器部署开源大模型（LLaMA、Qwen、DeepSeek 等）并构建生产级推理服务的开发者。  
> **最后更新**：2026-05  
> **参考来源**：vLLM 官方文档、HuggingFace Transformers 文档、QubitTool 技术博客、CSDN、知乎、GitHub 各项目 README。

---

## 目录

1. [模型量化：GPTQ / AWQ / GGUF](#1-模型量化-gptq--awq--gguf)
2. [推理引擎对比：vLLM / TGI / llama.cpp / Ollama](#2-推理引擎对比-vllm--tgi--llamacpp--ollama)
3. [GPU 选型与显存估算](#3-gpu-选型与显存估算)
4. [API 服务化：OpenAI 兼容接口](#4-api-服务化-openai-兼容接口)
5. [性能优化：Continuous Batching / Paged Attention / Flash Attention](#5-性能优化-continuous-batching--paged-attention--flash-attention)
6. [生产级部署架构：负载均衡 / 缓存 / 监控](#6-生产级部署架构-负载均衡--缓存--监控)
7. [成本优化](#7-成本优化)

---

## 1. 模型量化：GPTQ / AWQ / GGUF

### 1.1 为什么需要量化？

一个 7B 参数的模型以 FP16 加载需要约 **14 GB** 显存（7B × 2 bytes）。加上 KV Cache 和推理中间激活，实际显存需求可达 **18–22 GB**，远超消费级 GPU（RTX 4060 8GB / RTX 4090 24GB）。

量化的核心思想：**用更少的比特数表示权重**，在不显著损失精度的情况下大幅降低显存占用。

| 精度 | 每参数 bits | 7B 模型权重大小 | 精度损失 |
|------|------------|----------------|---------|
| FP32 | 32 | ~26 GB | 无 |
| FP16 | 16 | ~13 GB | 极小 |
| BF16 | 16 | ~13 GB | 极小 |
| INT8 | 8 | ~6.5 GB | 小 |
| INT4 | 4 | ~3.3 GB | 中等（可接受）|

---

### 1.2 GPTQ（Post-Training Quantization）

**全称**：Generative Pre-trained Transformer Quantization  
**提出者**：Elias Frantar 等（2023）  
**论文**：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

#### 核心原理

GPTQ 是一种**逐层**的**训练后量化**（PTQ）方法，基于 **OBS（Optimal Brain Surgeon）** 二阶信息理论：

1. 对每一层的权重矩阵，计算 Hessian 矩阵（基于校准数据的二阶导数）
2. 逐列量化权重，并在量化每一列后，用 Hessian 的逆来**补偿剩余未量化列的误差**
3. 最终实现 3-bit / 4-bit 量化，精度损失极小

#### 特点

- ✅ 量化速度快（几小时完成 7B 模型）
- ✅ 生态成熟：HuggingFace 上大量预量化 GPTQ 模型可直接使用
- ✅ 支持 Marlin 内核（4-bit CUDA 优化），在 A100/H100 上推理速度极快
- ❌ 仅支持 GPU 推理（依赖 CUDA 内核）
- ❌ 量化过程需要校准数据集（通常几百条文本即可）

#### 快速使用

```python
# 使用 AutoGPTQ 加载量化模型
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_safetensors=True
)

# 使用 vLLM 加载 GPTQ 模型（推荐）
# vllm serve TheBloke/Qwen2.5-7B-Instruct-GPTQ-Int4
```

---

### 1.3 AWQ（Activation-Aware Weight Quantization）

**全称**：Activation-aware Weight Quantization  
**提出者**：MIT 等（2023）  
**论文**：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)

#### 核心原理

AWQ 的核心理念：**并非所有权重对模型输出同等重要。**

1. 分析激活值分布，找到"显著权重通道"（对应大激活值的权重）
2. 对这些重要通道应用 **per-channel scaling** 进行保护
3. 其余通道正常量化

这样可以在保护关键权重的同时保持低比特量化。

#### 特点

- ✅ 同等比特下精度通常优于 GPTQ
- ✅ 量化速度比 GPTQ 更快（无需复杂的 Hessian 计算）
- ✅ 有高效的量化内核支持（AWQ + TinyChat）
- ❌ 预量化模型生态不如 GPTQ 丰富
- ❌ 同样仅支持 GPU

#### 快速使用

```python
# vLLM 已内置 AWQ 支持
# vllm serve TheBloke/Qwen2.5-7B-Instruct-AWQ
```

---

### 1.4 GGUF（GGML Universal Format）

**来源**：llama.cpp 项目的模型格式  
**定位**：专为 **CPU/边缘设备** 推理设计

#### 核心原理

GGUF 是一种**自包含的模型文件格式**，特点：

1. 将模型权重、tokenizer、元数据打包到一个文件中
2. 支持多种量化级别：`q2_K`、`q3_K_S`、`q4_K_M`、`q5_K_M`、`q6_K`、`q8_0` 等
3. **K-quant** 系列：对 attention 层和 FFN 层使用不同精度的混合量化策略
4. 支持 CPU 推理（avx2/avx512）、Apple Silicon（Metal）、部分 GPU（CUDA/Vulkan）

#### GGUF 量化级别速览

| 量化级别 | 质量 | 7B 大小 | 适用场景 |
|---------|------|---------|---------|
| q2_K | 最低 | ~2.8 GB | 极度显存受限 |
| q3_K_M | 低 | ~3.5 GB | 嵌入式设备 |
| **q4_K_M** | **推荐** | **~4.4 GB** | **通用本地推理 ⭐** |
| q5_K_M | 较高 | ~5.3 GB | 质量优先 |
| q8_0 | 接近 FP16 | ~7 GB | 质量要求较高 |

#### 特点

- ✅ 一个文件即用，无需额外依赖
- ✅ CPU 友好，消费级硬件即可运行
- ✅ **Ollama** 和 **llama.cpp** 原生支持
- ✅ Apple Silicon 优化极好
- ❌ GPU 推理性能不如 GPTQ/AWQ
- ❌ 不支持 tensor parallelism 等高级并行策略

#### 快速使用

```bash
# 下载 GGUF 模型
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir ./models

# 用 llama.cpp 运行
./llama-cli -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -p "你好，请介绍一下量化技术" -n 512

# 或用 Ollama
ollama run qwen2.5:7b
```

---

### 1.5 量化方法选择决策树

```
你的部署环境是？
├── GPU 服务器（A100/H100/A6000）
│   └── 追求吞吐量 → vLLM + GPTQ-Int4 或 AWQ-Int4
│       追求极高质量 → FP16 原版 + vLLM
│       追求极致吞吐 → FP8 (H100+) + vLLM
├── 消费级 GPU（RTX 3090/4090 24GB）
│   └── GPTQ-Int4 或 AWQ-Int4 + vLLM/TGI
├── Apple Silicon (M1/M2/M3/M4)
│   └── GGUF q4_K_M + llama.cpp 或 Ollama
└── CPU Only / 边缘设备
    └── GGUF q4_K_M + llama.cpp
```

---

## 2. 推理引擎对比：vLLM / TGI / llama.cpp / Ollama

### 2.1 总览对比表

| 维度 | **vLLM** | **TGI** | **llama.cpp** | **Ollama** |
|------|----------|---------|---------------|------------|
| 开发者 | UC Berkeley / 社区 | HuggingFace | ggerganov / 社区 | Ollama Inc |
| 语言 | Python / C++/CUDA | Rust / Python | C/C++ | Go |
| 定位 | 高性能 GPU 推理 | 生产级 GPU 推理 | CPU/边缘推理 | 桌面端一键部署 |
| 模型格式 | HF Transformers | HF Transformers | GGUF | GGUF |
| 核心优化 | PagedAttention, Continuous Batching | Flash Attention, Continuous Batching | mmap, Metal, AVX2 | 封装 llama.cpp |
| OpenAI 兼容 | ✅ 完整兼容 | ✅ 完整兼容 | ✅ 通过 server | ✅ 完整兼容 |
| 多 GPU | ✅ Tensor/Pipeline Parallel | ✅ Tensor Parallel | ❌ 有限支持 | ❌ 有限支持 |
| 量化支持 | GPTQ/AWQ/FP8/GGUF | GPTQ/AWQ/bnb | GGUF | GGUF |
| LoRA 热加载 | ✅ | ✅ | ❌ | ❌ |
| 生产就绪度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 适合谁 | 追求极致吞吐的团队 | 需要 HF 生态集成 | 个人开发者/边缘部署 | 个人用户快速上手 |

---

### 2.2 vLLM — 高性能 GPU 推理首选

**GitHub**：[vllm-project/vllm](https://github.com/vllm-project/vllm) ⭐ 45k+

#### 核心亮点

1. **PagedAttention**：将 KV Cache 分页管理，显存利用率提升 2-4x
2. **Continuous Batching**：动态合并请求，吞吐量提升 10-23x
3. **Chunked Prefill**：将长 prefill 切片，避免阻塞 decode
4. **Prefix Caching**：自动缓存共享 prefix（如 system prompt）
5. **OpenAI 兼容 API**：完全兼容 `/v1/chat/completions`、`/v1/completions`
6. 支持 **300+ 模型架构**，覆盖所有主流开源模型

#### 快速启动

```bash
# 安装
pip install vllm

# 启动 OpenAI 兼容服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256
  }'
```

#### 何时选择 vLLM

- 需要**最高吞吐量**的 GPU 推理场景
- 需要 OpenAI 兼容 API
- 需要多 GPU 分布式推理
- 模型在 vLLM 支持列表中（绝大多数主流模型都支持）

---

### 2.3 TGI（Text Generation Inference）

**GitHub**：[huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)  
**由 HuggingFace 官方维护**，是其推理服务的基础设施。

#### 核心亮点

1. **与 HF Hub 深度集成**：直接加载 HF 上的模型
2. **Watermarking**：内置水印检测
3. **Safetensors**：安全的模型加载格式
4. **内置 Prometheus 指标**：完整的监控体系
5. **Token Streaming**：支持 SSE 流式输出

#### 快速启动

```bash
# Docker 方式（推荐）
docker run --gpus all -p 8080:80 \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --max-total-tokens 8192

# 测试
curl http://localhost:8080/generate \
  -X POST -d '{"inputs":"你好","parameters":{"max_new_tokens":256}}' \
  -H 'Content-Type: application/json'
```

#### 何时选择 TGI

- 团队使用 HuggingFace 生态
- 需要直接与 HF Hub 模型无缝集成
- 需要 HF 的付费推理端点（Inference Endpoints 后端即 TGI）

---

### 2.4 llama.cpp — CPU/边缘推理之王

**GitHub**：[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) ⭐ 75k+

#### 核心亮点

1. **纯 C/C++ 实现**，无 Python 依赖
2. **CPU 优化**：AVX2、AVX512、NEON（ARM）
3. **Apple Silicon Metal 加速**：M 系列芯片推理速度极快
4. **GGUF 格式**：一文件分发，量化灵活
5. **轻量 Server 模式**：内置 HTTP API

#### 快速启动

```bash
# 编译
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# 下载 GGUF 模型
wget https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# 启动 HTTP 服务
./llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99  # 层数放到 GPU（Metal/CUDA）
```

#### 何时选择 llama.cpp

- 在 **CPU 环境** 或 **Apple Silicon** 上部署
- 显存极度有限
- 需要最小的依赖和部署体积
- 嵌入式/边缘设备场景

---

### 2.5 Ollama — 桌面端一键部署

**官网**：[ollama.com](https://ollama.com)

#### 核心亮点

1. **一条命令运行模型**：`ollama run qwen2.5:7b`
2. **Modelfile**：自定义模型配置和 prompt
3. **REST API**：内置 OpenAI 兼容接口
4. **跨平台**：macOS / Linux / Windows
5. **后台封装 llama.cpp**，自动选择最佳量化级别

#### 快速启动

```bash
# Linux 安装
curl -fsSL https://ollama.com/install.sh | sh

# 运行模型
ollama run qwen2.5:7b

# API 模式
ollama serve
# 然后在另一个终端：
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"你好"}]}'
```

#### 何时选择 Ollama

- 个人开发者快速体验
- 本地桌面端 AI 助手
- 不需要高并发，只需要单用户场景
- 配合 Open WebUI 等前端使用

---

## 3. GPU 选型与显存估算

### 3.1 显存需求公式

```
总显存 ≈ 模型权重 + KV Cache + 激活内存 + 系统开销

1. 模型权重 = 参数量 × 每参数字节数
   例：7B FP16 = 7B × 2 bytes ≈ 14 GB
       7B INT4 = 7B × 0.5 bytes ≈ 3.5 GB

2. KV Cache ≈ 2 × 层数 × hidden_size × max_seq_len × batch_size × 2 bytes(FP16)
   简化：约 0.5-2 GB per 1k tokens per batch (for 7B)

3. 激活内存 ≈ 总显存的 5%-15%（取决于 batch size）
```

### 3.2 常见模型显存需求速查

| 模型规模 | FP16 权重 | INT4(GPTQ) | GGUF q4_K_M | 推荐 GPU |
|---------|----------|------------|-------------|-----------|
| 1.5B-3B | 3-6 GB | 1.5-3 GB | ~2 GB | RTX 3060/4060 8GB |
| 7B-8B | 14-16 GB | 4-5 GB | ~4.5 GB | RTX 3080/4070/4080 12-16GB |
| 13B-14B | 26-28 GB | 7-8 GB | ~8 GB | RTX 3090/4090 24GB, A5000 24GB |
| 30B-34B | 60-68 GB | 16-18 GB | ~18 GB | A6000 48GB, A100 40GB |
| 65B-72B | 130-144 GB | 34-38 GB | ~38 GB | A100 80GB ×2, H100 80GB ×2 |
| 175B+ | 350+ GB | 88+ GB | ~90 GB | H100 ×4+, A100 ×8 |

### 3.3 主流 GPU 选型指南

| GPU | 显存 | FP16 算力 | 适合部署的模型 | 成本定位 |
|-----|------|----------|---------------|---------|
| **RTX 4060** | 8 GB | 15 TFLOPS | 1.5-3B 量化模型 | 💰 入门 |
| **RTX 4070** | 12 GB | 29 TFLOPS | 7B INT4 量化 | 💰💰 个人 |
| **RTX 4090** | 24 GB | 83 TFLOPS | 7B FP16, 14B INT4, 34B INT4 | 💰💰💰 高端个人/小团队 |
| **A6000 Ada** | 48 GB | 91 TFLOPS | 34B FP16, 72B INT4 | 💰💰💰💰 专业工作站 |
| **A100 80GB** | 80 GB | 312 TFLOPS | 72B FP16, 70B+ MOE | 💰💰💰💰💰 企业级 |
| **H100 80GB** | 80 GB | 989 TFLOPS | 所有规模 + FP8 加速 | 💰💰💰💰💰💰 顶级集群 |
| **Apple M2 Ultra** | 192 GB 统一 | - | 72B FP16 (CPU+GPU) | 💰💰💰💰 特殊场景 |

### 3.4 显存优化技巧

1. **量化**：INT4 减少 75% 权重大小
2. **Flash Attention**：减少 KV Cache 显存占用
3. **PagedAttention**：减少 KV Cache 碎片
4. **vLLM `--gpu-memory-utilization`**：调高到 0.90-0.95
5. **减少 max_model_len**：少分配 KV Cache 空间
6. **CPU Offloading**：将部分层 offload 到 CPU 内存

---

## 4. API 服务化：OpenAI 兼容接口

### 4.1 为什么要 OpenAI 兼容？

- **生态无缝集成**：与 langchain、llamaindex、autogen、dify 等框架直接对接
- **客户端现成可用**：openai-python、openai-node 等无需修改
- **易于切换后端**：从 OpenAI 迁移到自部署零代码改动

### 4.2 vLLM OpenAI 兼容服务

vLLM 天然提供 OpenAI 兼容的 API Server：

```bash
# 启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key sk-my-secret-key

# Python 客户端（与 openai 库完全相同）
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-my-secret-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "什么是PagedAttention？"}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True  # 支持流式输出
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 4.3 支持的 OpenAI API 端点

vLLM 完整支持以下端点：

| 端点 | 说明 |
|------|------|
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | 对话补全（支持流式） |
| `POST /v1/completions` | 文本补全（支持流式） |
| `POST /v1/embeddings` | 文本嵌入 |

### 4.4 Ollama OpenAI 兼容

Ollama 从 v0.1.24 开始也支持 OpenAI 兼容 API：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 随意填写
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 4.5 LiteLLM — 统一 API 代理（进阶）

**GitHub**：[BerriAI/litellm](https://github.com/BerriAI/litellm)

LiteLLM 可以将多个 LLM 后端统一为一个 OpenAI 兼容端点，支持：

- ✅ 多模型路由
- ✅ 负载均衡
- ✅ 成本追踪
- ✅ 速率限制
- ✅ 日志记录

```bash
pip install litellm[proxy]
litellm --model huggingface/Qwen2.5-7B-Instruct --port 4000
```

---

## 5. 性能优化：Continuous Batching / Paged Attention / Flash Attention

### 5.1 Continuous Batching（连续批处理）

#### 问题：传统 Static Batching

传统推理框架（如 HuggingFace TGI 早期版本）使用 static batching：
- 必须等待一个 batch 中**所有请求完成**才能开始下一个 batch
- 提前完成的请求空等，浪费计算资源
- 长短请求混合时长尾延迟严重

```
传统 Static Batching:
┌──────────────┐
│ Req1 ████████│  ← 已结束，空等
│ Req2 ████████████████████│
│ Req3 ██████│  ← 已结束，空等
└──────────────┘
```

#### 解决：Continuous Batching

vLLM 和现代 TGI 使用 continuous batching：
- 每当一个请求完成，**立即**将新请求加入 batch
- 不需要等整个 batch 结束
- 吞吐量可提升 **10-23x**

```
Continuous Batching:
┌──────────────────────────────────┐
│ Req1 ████████│ Req4 ████│ Req6 ██│
│ Req2 ████████████████████████│
│ Req3 ██████│ Req5 ████████████│
└──────────────────────────────────┘
```

#### 关键参数

```bash
vllm serve model \
  --max-num-seqs 256 \        # 最大并发序列数
  --max-num-batched-tokens 8192  # 每 batch 最大 token 数
```

---

### 5.2 PagedAttention（分页注意力）

**论文**：[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023)

#### 核心问题

LLM 推理时，每个 token 需要存储 KV Cache。传统方式为每个请求预分配一块**连续的显存**（类似 OS 的连续内存分配），导致：

1. **内部碎片**：预分配了但用不完的空间浪费
2. **外部碎片**：多个请求之间的小空隙无法利用
3. **无法共享**：不同请求的相同 prefix（如 system prompt）各自存一份

#### PagedAttention 方案

借鉴操作系统**虚拟内存分页**思想：

- 将 KV Cache 划分为固定大小的 **KV 块**（block）
- 请求按需分配块，不需要连续空间
- 多个请求可**共享同一块**（如 system prompt），大幅节省显存

```
传统 KV Cache:               PagedAttention:
Req1: [████████░░░░░░]       Block Table:
Req2: [████████████████]     Req1 → Block 0, Block 1, Block 3
      ↑ 大量碎片              Req2 → Block 0(共享!), Block 2, Block 4
```

**效果**：显存利用率提升 **2-4x**，支持 **4x 更大的 batch size**。

---

### 5.3 Flash Attention

**论文**：
- [FlashAttention](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024)

#### 核心原理

标准 Attention 计算需要将中间矩阵 (`QK^T` / `softmax`) 写入 HBM（高带宽显存），I/O 成为瓶颈。Flash Attention 通过以下方式解决：

1. **Tiling（分块）**：将 Q、K、V 分块，在 SRAM（共享内存）中完成 softmax 计算
2. **Recomputation**：反向传播时重新计算 attention，而不是存储中间结果
3. **IO-aware**：精确控制数据在 HBM ↔ SRAM 之间的移动

```
标准 Attention:
HBM 读取 Q,K,V → 计算 S=QK^T → 写回 HBM（大矩阵!）
→ 读取 S → softmax → 写回 HBM → 读取 → ×V → 写回
（多次 HBM 读写，瓶颈）

Flash Attention:
分块读取 Q,K,V 到 SRAM → 在 SRAM 内完成所有计算
→ 仅将最终结果写回 HBM
（大幅减少 HBM 访问）
```

#### 效果

- 显存占用：O(N) 代替 O(N²)
- 速度：2-4x 加速
- 支持更长上下文（128K+ tokens）

#### 在 vLLM 中使用

vLLM 默认自动使用 Flash Attention 2/3（需要安装 flash-attn）：

```bash
pip install flash-attn --no-build-isolation
# vLLM 会默认检测并使用
```

---

### 5.4 优化总结

| 技术 | 核心收益 | 是否默认开启 |
|------|---------|------------|
| **Continuous Batching** | 吞吐量 10-23x ↑ | vLLM/TGI 默认 |
| **PagedAttention** | 显存利用率 2-4x ↑ | vLLM 默认 |
| **Flash Attention 2/3** | 速度 2-4x ↑，长上下文支持 | 安装后自动 |
| **Prefix Caching** | 共享前缀 50%+ 显存节省 | vLLM: `--enable-prefix-caching` |
| **Chunked Prefill** | 减少 decode 延迟抖动 | vLLM: `--enable-chunked-prefill` |
| **Speculative Decoding** | 延迟减少 2-3x | 需要配置 draft model |

---

## 6. 生产级部署架构：负载均衡 / 缓存 / 监控

### 6.1 总体架构

```
                         ┌─────────────────────┐
                         │    Nginx / Traefik   │
                         │   (负载均衡 + HTTPS)  │
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │ vLLM #1   │ │ vLLM #2   │ │ vLLM #3   │
              │ (GPU 0)   │ │ (GPU 1)   │ │ (GPU 2)   │
              └───────────┘ └───────────┘ └───────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │   Redis (缓存层)    │
                         │  - Semantic Cache  │
                         │  - Rate Limiting   │
                         └────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │Prometheus │ │  Grafana  │ │   Loki    │
              │ (指标收集) │ │ (可视化)   │ │ (日志收集) │
              └───────────┘ └───────────┘ └───────────┘
```

### 6.2 负载均衡

#### Nginx 配置示例

```nginx
upstream vllm_backend {
    least_conn;  # 最少连接策略
    server 10.0.0.1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.3:8000 weight=1 max_fails=3 fail_timeout=30s;
    keepalive 32;  # 保持连接池
}

server {
    listen 443 ssl;
    server_name api.your-llm.com;

    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;  # LLM 推理可能耗时较长
        proxy_buffering off;       # 流式输出需要关闭缓冲
        proxy_cache off;
        chunked_transfer_encoding on;
    }
}
```

#### 负载均衡策略

| 策略 | 适用场景 |
|------|---------|
| `least_conn` | 请求耗时不均（LLM 场景推荐）|
| `round_robin` | 请求耗时均匀 |
| `ip_hash` | 需要会话保持 |
| `random` | 简单场景 |

### 6.3 缓存策略

#### 6.3.1 语义缓存（Semantic Cache）

对于相似问题的重复请求，返回缓存结果：

```python
# 使用 GPTCache
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt

cache = Cache()
cache.init(
    pre_embedding_func=get_prompt,
    data_manager=manager_factory("redis,faiss",
        vector_params={"dimension": 1536},
        redis_params={"host": "localhost"})
)
```

#### 6.3.2 vLLM Prefix Caching

```bash
vllm serve model --enable-prefix-caching
```

自动缓存 system prompt 的 KV Cache，后续请求直接复用，大幅减少 prefill 时间。

### 6.4 监控体系

#### 关键指标

| 指标类别 | 具体指标 | 告警阈值建议 |
|---------|---------|------------|
| **延迟** | TTFT（首 token 时间） | > 2s P95 |
| **延迟** | TPOT（每 token 时间） | > 50ms P95 |
| **吞吐** | tokens/s, requests/s | < 预期 50% |
| **显存** | GPU 显存使用率 | > 95% |
| **队列** | 等待队列长度 | > 100 |
| **错误** | 4xx/5xx 错误率 | > 1% |
| **利用率** | GPU 利用率 | < 70%（浪费）|

#### Prometheus + Grafana

vLLM 内置 Prometheus 指标导出：

```bash
# 启动时开启指标
vllm serve model --host 0.0.0.0 --port 8000

# 指标端点
curl http://localhost:8000/metrics
```

#### Grafana Dashboard

vLLM 官方提供预配置的 Grafana Dashboard，导入即可使用：
- 请求延迟分布
- GPU 利用率曲线
- KV Cache 使用率
- 队列长度历史

---

## 7. 成本优化

### 7.1 成本构成分析

运行一个 LLM 推理服务的成本主要包括：

| 成本项 | 占比 (典型) | 优化空间 |
|--------|------------|---------|
| GPU 算力（云/自建） | 60-80% | ⭐⭐⭐ |
| 带宽 | 5-10% | ⭐ |
| 存储 | 5-10% | ⭐ |
| 人力运维 | 10-20% | ⭐⭐ |

### 7.2 GPU 成本优化策略

#### 策略 1：量化

```
7B FP16 on A100 80GB：可跑 batch_size=128，成本 $2.5/h
7B INT4 on A10 24GB：可跑 batch_size=32，成本 $0.8/h
→ 成本降低约 68%，精度损失 < 3%
```

#### 策略 2：选择合适的 GPU

| GPU | 云成本 ($/h) | 性价比 (tokens/$) |
|-----|-------------|-------------------|
| A100 80GB | ~$2.5 | 基准 |
| A6000 48GB | ~$1.5 | +40% |
| RTX 4090 (自建) | ~$0.5/h 电费 | +200% |
| L40S 48GB | ~$1.2 | +60% |

#### 策略 3：弹性扩缩容

```bash
# Kuberay + vLLM: 基于 GPU 利用率的 HPA
# 低峰期缩容到 1 副本，高峰期扩容到 N 副本
```

#### 策略 4：Spot/Preemptible 实例

云厂商的抢占式实例成本可降低 **60-80%**：

```yaml
# AWS Spot Instance 请求示例
# 配合 vLLM checkpoint/restart 能力使用
```

#### 策略 5：多模型共享 GPU

使用 vLLM 的 LoRA 热加载，单 GPU 同时服务多个微调模型：

```bash
vllm serve base-model \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64
```

### 7.3 缓存降本

- **Prefix Caching**：system prompt KV Cache 复用，节省 30-50% prefill 计算
- **语义缓存**：相似问题直接返回，节省 20-40% 推理调用
- **结果缓存**：高频固定问题缓存

### 7.4 成本估算公式

```
月成本 ≈ GPU实例时价 × 24h × 30天 × GPU数量 × 利用率系数

例：1× A100 80GB 7×24 服务
月成本 ≈ $2.5 × 24 × 30 × 1 × 0.85 ≈ $1,530/月
可处理：约 1,500-2,000 万 tokens/天（7B 模型）
即：约 $0.003 per 1K tokens
vs OpenAI GPT-4：$0.03 per 1K tokens
→ 自建成本约为 API 的 1/10
```

---

## 附录：推荐学习资源

### 论文

- [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AWQ (MLSys 2024)](https://arxiv.org/abs/2306.00978)
- [vLLM / PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [FlashAttention (NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)

### 官方文档

- [vLLM 官方文档](https://docs.vllm.ai/)
- [HuggingFace TGI 文档](https://huggingface.co/docs/text-generation-inference/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Ollama 官方文档](https://github.com/ollama/ollama/tree/main/docs)

### 中文资源

- [QubitTool: 模型量化完全指南](https://qubittool.com/zh/blog/model-quantization-complete-guide)
- [CSDN: GPTQ 原理与实现](https://blog.csdn.net/u012535132/article/details/159416872)
- [Fenrier Lab: GPTQ 量化技术演进](https://seanwangjs.github.io/2024/04/05/gptq.html)

---

> **下一步学习建议**：掌握本文内容后，建议动手实践——用 vLLM 部署一个 Qwen2.5-7B 模型，配置 OpenAI 兼容接口，对接 LangChain 构建一个 RAG 应用。实践是最好的学习方式。
