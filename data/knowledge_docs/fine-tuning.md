# 大模型微调（Fine-tuning）—— 中级指南

> 从零到一掌握大模型微调：原理、工具、实战

---

## 1. 为什么要微调？

| 方式 | 适用场景 | 成本 | 效果 |
|------|---------|------|------|
| **Prompt Engineering** | 任务简单、格式固定 | 低 | 不稳定、token 消耗大 |
| **RAG** | 知识密集型、需要时效性 | 中 | 检索质量决定上限 |
| **Fine-tuning** | 风格/格式/领域专用 | 高 | 稳定、延迟低、质量高 |

**微调的核心价值**：让模型学会「怎么说」（风格/格式）和「说什么」（领域知识），而不仅仅是「查什么」。

---

## 2. 全量微调 vs 参数高效微调

### 2.1 全量微调（Full Fine-tuning）

更新模型所有参数。7B 模型全量微调需要约 60GB 显存（bf16）。

```
显存需求 = 模型参数 × 2 (bf16) + 梯度 × 2 + 优化器状态 × 4
         = 14B × 18 = 252GB（实际因 flash attention 等可压缩到 ~100GB）
```

**适用**：有充足算力（8×A100），追求极致效果。

### 2.2 参数高效微调（PEFT）

只更新模型的一小部分参数，主流方案对比：

| 方法 | 原理 | 可训参数占比 | 显存节省 | 效果 |
|------|------|-------------|---------|------|
| **LoRA** | 低秩矩阵注入 Attention 层 | ~0.1%-1% | ~70% | ⭐⭐⭐⭐⭐ |
| **QLoRA** | LoRA + 4bit 量化 | ~0.1%-1% | ~90% | ⭐⭐⭐⭐ |
| **Adapter** | 插入小型适配层 | ~1%-5% | ~60% | ⭐⭐⭐ |
| **Prefix Tuning** | 可学习 prefix token | ~0.01% | ~80% | ⭐⭐⭐ |
| **IA3** | 缩放向量 | ~0.01% | ~85% | ⭐⭐⭐ |

**推荐**：绝大多数场景用 **QLoRA**，单卡 24GB 就能微调 7B 模型。

---

## 3. LoRA 原理与实现

### 3.1 核心思想

原始权重矩阵 `W` (d×k)，LoRA 将其更新分解为两个低秩矩阵的乘积：

```
ΔW = A × B
其中 A: d×r, B: r×k, r << min(d, k)   (r 通常取 8~64)

前向传播: h = W·x + ΔW·x = W·x + A·B·x
```

**为什么有效**：模型微调时的权重更新矩阵是低秩的（Aghajanyan et al., 2020），大部分信息可以用很少的参数捕捉。

### 3.2 使用 PEFT 库实现

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    torch_dtype="auto",
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank — 越大表达能力越强，但参数越多
    lora_alpha=32,           # 缩放因子，通常设为 2×r
    lora_dropout=0.1,        # dropout 防止过拟合
    target_modules=[         # 注入的目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 四件套
        "gate_proj", "up_proj", "down_proj"       # MLP 层
    ],
    bias="none"              # 不训练 bias
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 8,388,608 || all params: 7,078,559,744 || trainable%: 0.1185%
```

### 3.3 LoRA 关键超参

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `r` | 8~64 | 简单任务 r=8，复杂任务 r=32~64 |
| `lora_alpha` | 2×r | 相当于学习率缩放 |
| `target_modules` | all-linear | Qwen2/Llama3 全 linear 层效果最好 |
| `lora_dropout` | 0.05~0.1 | 小数据集加 dropout |

---

## 4. QLoRA：4bit 量化 + LoRA

### 4.1 原理

QLoRA 在 LoRA 基础上引入三个关键技术：

1. **4-bit NormalFloat (NF4)**：专门为正态分布权重设计的量化格式
2. **双重量化（Double Quantization）**：对量化常数再进行量化，进一步节省 0.4 bit/参数
3. **分页优化器（Paged Optimizers）**：利用 unified memory 处理梯度峰值

### 4.2 实现

```python
import torch
from transformers import BitsAndBytesConfig

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 之后和普通 LoRA 完全一样
model = get_peft_model(model, lora_config)

# 显存对比（Qwen2-7B）：
# 全量微调: ~55GB
# LoRA:     ~18GB
# QLoRA:    ~8GB   ← 单张 RTX 3070(8G) 就能跑！
```

---

## 5. 微调数据准备

### 5.1 数据格式

监督微调（SFT）的标准对话格式：

```jsonl
{"messages": [{"role": "system", "content": "你是一个专业的大模型应用开发助教"}, {"role": "user", "content": "什么是 RAG？"}, {"role": "assistant", "content": "RAG（检索增强生成）是一种将信息检索与文本生成结合的架构..."}]}
```

### 5.2 数据质量黄金法则

| 法则 | 说明 |
|------|------|
| **少而精 > 多而杂** | 1000 条高质量数据效果优于 10000 条噪声数据 |
| **多样性覆盖** | 确保覆盖目标场景的各种边界情况 |
| **格式一致** | 输出风格、结构、术语保持一致 |
| **长度控制** | 训练数据长度分布应贴近实际使用场景 |
| **去重去污染** | 去除重复、与评测集重叠的数据 |

### 5.3 数据增强技巧

```python
# 1. Self-Instruct：用强模型生成训练数据
# 提示 GPT-4/DeepSeek 生成 (instruction, output) 对

# 2. Evol-Instruct：渐进式增加复杂度
# 从简单问题开始，逐步加深到多步推理

# 3. 反向翻译增强
# 中文 → 英文 → 中文，增加表达的多样性
```

### 5.4 推荐数据量

| 任务类型 | 最少数据 | 推荐数据 | 饱和点 |
|---------|---------|---------|--------|
| 指令遵循 | 1000 | 5000-10000 | ~50000 |
| 格式输出 | 500 | 2000-5000 | ~10000 |
| 风格迁移 | 1000 | 3000-8000 | ~20000 |
| 领域知识注入 | 5000 | 20000+ | 持续受益 |

---

## 6. SFT vs RLHF vs DPO

### 6.1 三种范式对比

```
阶段 1: SFT（监督微调）
  输入: (prompt, 理想回答) 对
  目标: 让模型学会输出格式和基本内容
  局限: 没有「好坏」判断，只会模仿

阶段 2: RLHF（人类反馈强化学习）
  Step 1: 训练奖励模型（Reward Model）
  Step 2: PPO 强化学习优化
  优点: 对齐人类偏好
  缺点: 需要大量人类标注 + 训练不稳定

阶段 3: DPO（直接偏好优化）★ 推荐
  输入: (prompt, chosen回答, rejected回答) 三元组
  优势: 不需要奖励模型，直接优化，稳定
```

### 6.2 DPO 训练示例

```python
# DPO 数据格式
{
    "prompt": "解释什么是 Python 装饰器",
    "chosen": "装饰器是一个接受函数并返回新函数的可调用对象...（详细、有代码示例）",
    "rejected": "装饰器就是@符号后面跟函数名（过于简略，无示例）"
}

# 使用 TRL 库训练
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,      # 冻结的参考模型
    beta=0.1,                 # KL 散度惩罚系数
    train_dataset=dpo_dataset,
    args=training_args,
)
trainer.train()
```

---

## 7. LLaMA-Factory：一站式微调工具

### 7.1 为什么选 LLaMA-Factory

- 🚀 支持 100+ 模型（Llama/Qwen/DeepSeek/GLM...）
- 📊 内置 LoRA/QLoRA/全量微调
- 🖥️ Web UI + CLI 双模式
- 📦 集成 SwanLab/W&B 实验追踪
- 🔧 支持 SFT/DPO/PPO/KTO 等训练方式

### 7.2 CLI 快速上手

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# QLoRA 微调 Qwen2-7B
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2-7B \
    --dataset my_custom_data \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target all \
    --quantization_bit 4 \
    --output_dir ./output/qwen2-lora \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 500
```

### 7.3 Web UI 模式

```bash
llamafactory-cli webui
# 浏览器打开 http://localhost:7860
# 1. 选模型 → 2. 选数据集 → 3. 配参数 → 4. 点训练
```

---

## 8. 过拟合防范

### 8.1 过拟合的信号

- 训练 loss 持续下降但验证 loss 上升
- 输出高度重复、缺乏多样性
- 仅能回答训练集中出现过的问题

### 8.2 防范策略

```python
# 1. 合理的 epoch 数（小数据集 1-3 个 epoch）
num_train_epochs = 3  # 不是越多越好

# 2. LoRA dropout
lora_dropout = 0.1

# 3. 权重衰减
weight_decay = 0.01

# 4. 学习率调度（余弦退火）
lr_scheduler_type = "cosine"
warmup_ratio = 0.1

# 5. 梯度裁剪
max_grad_norm = 1.0

# 6. NEFTune：给 embedding 加噪声
# 在 TrainingArguments 中设置
neftune_noise_alpha = 5.0  # 效果显著！防止过拟合
```

### 8.3 早停（Early Stopping）

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 必须有验证集
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

---

## 9. 评估微调效果

### 9.1 自动评估

```python
# 1. Perplexity（困惑度）
# 越低越好，反映模型对数据的拟合程度
trainer.evaluate()

# 2. ROUGE/BLEU（文本相似度）
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(generated, reference)

# 3. 专用评测集
# C-Eval, CMMLU, MMLU 等
from lm_eval import evaluator
results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=./my_finetuned_model",
    tasks=["ceval-valid", "cmmlu"],
)
```

### 9.2 人工评估维度

| 维度 | 评分标准（1-5） |
|------|----------------|
| **准确性** | 回答事实是否正确 |
| **完整性** | 是否覆盖了问题的所有方面 |
| **格式合规** | 输出格式是否符合要求 |
| **风格一致性** | 语气和风格是否一致 |
| **安全性** | 是否有不当/有害内容 |

### 9.3 A/B 测试

用基座模型和微调模型对同一批 prompt 生成回答，盲评哪个更好。

---

## 10. 实战案例：微调中文技术问答助手

### 10.1 目标

微调 Qwen2-1.5B，让它在「大模型应用开发」领域成为专业助教。

### 10.2 完整流程

```python
# Step 1: 准备数据 (data.jsonl)
# 包含 2000 条 QA 对，涵盖 Prompt/RAG/Agent/部署等主题

# Step 2: QLoRA 微调
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 配置
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 数据加载与格式化
def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = load_dataset("json", data_files="data.jsonl")
dataset = dataset.map(format_chat)

# 训练配置
training_args = TrainingArguments(
    output_dir="./qwen2-tech-qa-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch = 16
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    neftune_noise_alpha=5.0,             # 防过拟合神器
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()

# Step 3: 保存 + 合并
model.save_pretrained("./qwen2-tech-qa-lora-final")
# 合并 LoRA 权重到基座模型
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qwen2-tech-qa-merged")
```

### 10.3 推理测试

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./qwen2-tech-qa-merged",
    device_map="auto"
)

response = pipe(
    [{"role": "user", "content": "LoRA 的 rank 参数怎么选？"}],
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
print(response[0]["generated_text"])
```

---

## 11. 常见问题

### Q1: 微调后模型变「笨」了怎么办？

**灾难性遗忘（Catastrophic Forgetting）**：解决方案——
- 训练数据中混入 10%-20% 通用数据
- 使用较小的学习率（1e-5 ~ 5e-5）
- 适当降低 LoRA rank

### Q2: 数据太少怎么办？

- 使用更强的基座模型（7B > 1.5B）
- 调高 LoRA rank，降低学习率
- 数据增强：Self-Instruct、同义改写
- 多任务微调：多个相关任务一起训练

### Q3: 微调和 RAG 可以结合吗？

**可以，而且推荐！** RAFT (Retrieval Augmented Fine-Tuning) 模式：
1. 先用 RAG 扩充知识覆盖面
2. 在包含检索上下文的数据上微调
3. 模型学会「如何利用检索结果」，效果更好

---

## 12. 推荐资源

| 类型 | 资源 |
|------|------|
| 📄 论文 | LoRA (2106.09685), QLoRA (2305.14314), DPO (2305.18290) |
| 🔧 工具 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl) |
| 📊 数据 | [Belle](https://github.com/LianjiaTech/BELLE), [Firefly](https://github.com/yangjianxin1/Firefly), [Alpaca-ZH](https://github.com/mymusise/Alpaca-Chinese-dataset) |
| 📖 教程 | HuggingFace 官方 PEFT 文档, LLaMA-Factory Wiki |
| ☁️ 算力 | AutoDL（便宜）、Google Colab Pro、Kaggle GPU |
