# MMAS —— 多智能体协作医疗诊断系统

> Multi-agent Medical Assistant System · 面向秋招的项目展示仓库

## 项目简介

MMAS 是一个基于**多智能体协作（Multi-Agent）** 与 **混合专家系统（Mixture of Experts, MoE）** 的医疗诊断系统，用于自动完成临床病例选择题的诊断推理。系统通过三个专职智能体的协作，模拟"问诊 → 信息评估 → 澄清补充 → 专家会诊 → 决策"的完整诊断流程，并借助 MoE 专家路由机制，为不同专科的病例动态激活最相关的医学专家知识，从而提升诊断准确率。

## 核心亮点

- **三智能体协作架构**：Patient Agent、Evaluator Agent、Doctor Agent 各司其职，通过消息传递完成闭环诊断。
- **七步诊断流水线**：从病例格式化、信息质量评估、澄清问答、专家激活到最终决策，流程完整可复现。
- **MoE 专家系统**：基于语义相似度、关键词匹配、机制匹配三个维度（权重 α/β/γ 可调）动态激活专家，并通过 Softmax + 温度参数分配权重。
- **自适应澄清机制**：Evaluator Agent 根据信息完整度评分，自动决定是否需要向 Patient Agent 发起澄清提问，循环补充关键信息。
- **全局模型管理器**：单例模式统一管理模型加载与缓存，支持 8bit/4bit 量化，避免重复加载、节省显存。
- **多后端支持**：支持本地开源模型（Qwen/Llama 等）与 API 模型（DeepSeek-V3.2、GLM-5.1）灵活切换。
- **完善的工程化能力**：断点续训（Checkpoint）、日志管理、错误病例追踪、性能统计（准确率/时延/Token 用量）。
- **消融实验支持**：内置多种 ablation 开关，便于验证各模块对最终效果的贡献。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  MMAS 三智能体系统                        │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │ Patient      │──▶│ Evaluator    │──▶│ Doctor       │ │
│  │ Agent        │   │ Agent        │   │ Agent        │ │
│  │ (病人/病例)   │◀──│ (信息评估/澄清)│   │ (诊断推理)    │ │
│  └──────────────┘   └──────────────┘   └──────────────┘ │
│                          │                              │
│                          ▼                              │
│                 ┌──────────────────┐                    │
│                 │  MoE 专家系统     │◀── 专家知识图谱     │
│                 │ (专家激活/权重)   │◀── 语义模型         │
│                 └──────────────────┘                    │
│                          │                              │
│                          ▼                              │
│              GlobalModelManager（模型统一管理）           │
└─────────────────────────────────────────────────────────┘
```

## 诊断流水线（7 步）

1. **Step 1**：Patient Agent 将原始病例格式化为标准医学选择题。
2. **Step 2**：Evaluator Agent 对信息质量进行多维评分（基础/症状/检查/时间线/逻辑）。
3. **Step 3**：信息不足时，Evaluator Agent 生成澄清问题。
4. **Step 4**：Patient Agent 回答澄清问题，补充缺失信息。
5. **Step 5**：MoE 专家系统激活相关专科专家并分配权重。
6. **Step 6**：Doctor Agent 结合专家意见进行诊断推理。
7. **Step 7**：Evaluator Agent 汇总各专家意见，输出最终答案与置信度。

## 目录结构

```
.
├── run_optimized_mmas.py            # 主入口，编排三智能体诊断流程
├── mmas_patient_agent.py            # Patient Agent（病例格式化与澄清应答）
├── mmas_evaluator_agent.py          # Evaluator Agent（信息评估、澄清、最终决策）
├── mmas_doctor_agent.py             # Doctor Agent（诊断推理与专家意见融合）
├── moe_expert_system.py             # MoE 专家激活与权重分配
├── global_model_manager.py          # 全局模型管理器（单例，量化/缓存）
├── unified_expert_template_manager.py # 统一专家模板管理器
├── api_model_client.py              # API 模型客户端（OpenAI 兼容）
├── api_config.py                    # API 密钥与配置管理
├── log_manager.py                   # 日志管理器
├── intent_keywords.json             # 意图关键词库
├── mechanism_keywords.json          # 机制关键词库
├── precompute_embeddings.py         # 预编码专家知识
├── precompute_intent_vectors.py     # 预计算意图原型向量
├── dev_dataset_expert_knowledge_graph_enhanced.json # 专家知识图谱
└── requirements.txt                 # 依赖
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥（使用 API 模型时需要）

所有密钥均通过环境变量传入，**请勿硬编码**：

```bash
export DEEPSEEK_API_KEY="your_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export GLM_API_KEY="your_key"
export GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
```

开源模型（Qwen2.5/Qwen3/Llama 等）则通过 HuggingFace 本地加载，模型名映射见 `global_model_manager.py`。

## 运行方式

**入口：`run_optimized_mmas.py`**

使用本地模型：

```bash
python run_optimized_mmas.py \
    --dataset_path ./data/your_dataset.jsonl \
    --model_path ./models/Qwen3-4B-Instruct-2507 \
    --expert_knowledge_path ./data/dev_dataset_expert_knowledge_graph_enhanced.json
```

使用 API 模型（DeepSeek）：

```bash
python run_optimized_mmas.py \
    --dataset_path ./data/your_dataset.jsonl \
    --use_api_model \
    --api_model_name deepseek-chat \
    --expert_knowledge_path ./data/dev_dataset_expert_knowledge_graph_enhanced.json
```

## 常用参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset_path` | 数据集文件路径（JSONL） | `./data/all_dev_convo.jsonl` |
| `--model_path` | 本地 LLM 模型路径 | `./models/Qwen3-4B-Instruct-2507` |
| `--use_api_model` | 使用 API 模型而非本地模型 | 关闭 |
| `--api_model_name` | API 模型名（deepseek-chat / glm-5.1） | `deepseek-chat` |
| `--quantization` | 量化方式（none / 8bit / 4bit） | `none` |
| `--top_k` | 激活专家数量 | `3` |
| `--alpha` / `--beta` / `--gamma` | MoE 语义/关键词/机制权重 | 0.4 / 0.3 / 0.3 |
| `--temperature` | Softmax 温度参数 τ | `1.0` |
| `--max_clarification_loops` | 最大澄清循环次数 | `1` |
| `--use_adaptive_threshold` | 启用自适应剪枝策略 | 关闭 |
| `--resume` | 断点续训 | 关闭 |

### 消融实验开关

| 参数 | 作用 |
| --- | --- |
| `--ablation_generic_cot` | 使用通用 CoT 模板替代专业化意图图模板 |
| `--ablation_top1` | 仅激活 Top-1 专家 |
| `--ablation_equal_expert_weights` | 所有激活专家使用等权重 |
| `--ablation_no_mechanism` | 专家激活时禁用机制评分（γ=0） |
| `--ablation_no_clarification` | 禁用澄清循环 |

## 性能统计

系统运行结束后会输出完整统计信息，包括：

- **准确率（Accuracy）**：正确病例占比
- **平均每例耗时（Latency）**：单病例平均处理时间
- **平均激活专家数**、**平均澄清轮数**、**平均 Token 用量**
- **各步骤耗时分布**（初始化 / 模型加载 / 数据集加载 / 总处理时间）

## 技术栈

- **语言 / 框架**：Python、PyTorch、Transformers
- **语义模型**：Sentence-Transformers（paraphrase-multilingual-MiniLM-L12-v2）
- **模型后端**：本地开源模型（Qwen/Llama）+ OpenAI 兼容 API（DeepSeek、GLM）
- **工程组件**：单例模型管理、断点续训、量化、并发推理（ThreadPoolExecutor）

## 说明

- 本仓库为项目核心代码补充，聚焦于三智能体协作、MoE 专家路由与诊断流水线的实现。
- 数据集与模型权重未包含在仓库中，请自行准备。
- 所有 API 密钥均通过环境变量管理，确保安全。
