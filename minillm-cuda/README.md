# MiniLLM-CUDA

面向真实大语言模型推理场景的 CUDA C++ 算子开发与性能优化项目。

基于 PyTorch 和 Hugging Face Transformers，将自研 CUDA 算子集成至 Qwen2.5 模型，实现算子级优化及端到端推理性能分析。

## 1. 开发环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Ti |
| GPU 架构 | Blackwell（SM120） |
| 操作系统 | Ubuntu 22.04 |
| CUDA Toolkit | 13.0.3 |
| Python | 3.10 |
| PyTorch | 2.9.1+cu130 |
| Transformers | 4.57.6 |
| Nsight Compute | 2025.2.0 |

## 2. 环境部署

使用 CUDA 总仓库提供的统一环境部署脚本。

```bash
mkdir -p /workspace
cd /workspace

git clone https://github.com/SaladMaker123/CUDA.git
cd CUDA

bash scripts/setup_env.sh --with-ncu --pause-auto-updates
```

激活 Python 虚拟环境：

```bash
cd /workspace/CUDA/minillm-cuda
source .venv/bin/activate
```

部署脚本支持 CUDA Toolkit、Python 依赖及 Nsight Compute 安装，并提供 NVIDIA 驱动保护和自动更新管理功能。

暂停系统自动更新后，需要自行安排必要的系统安全更新。

## 3. 目标模型

模型：Qwen2.5-0.5B-Instruct

| 模型参数 | 数值 |
|---|---:|
| 参数量 | 494,032,768 |
| Transformer 层数 | 24 |
| 词表维度 | 151936 |
| 隐藏维度 | 896 |
| MLP 中间维度 | 4864 |
| Q Head 数量 | 14 |
| KV Head 数量 | 2 |
| 单头维度 | 64 |
| RMSNorm epsilon | 1e-6 |

推理配置：

- FP16 推理
- 自研归约算子采用 FP32 累加
- Batch Size = 1
- Greedy Decoding（贪心解码）
- KV Cache（键值缓存）
- SDPA 注意力后端

## 4. 核心算子

| 算子 | 主要优化方向 |
|---|---|
| RMSNorm | 并行归约、向量化访存 |
| SwiGLU | 算子融合、向量化访存 |
| RoPE | 线程映射、数据布局 |
| Softmax | 并行归约、Online Softmax |
| GEMV | 显存带宽、数据复用 |
| GEMM | 模型形状及权重布局适配 |
| Prefill Attention | 分块计算、FlashAttention |
| Decode Attention | GQA、KV Cache、Split-KV |

## 5. 项目结构

```text
minillm-cuda/
├── minillm/
│   ├── __init__.py
│   ├── model.py
│   ├── generate.py
│   └── inspect_shapes.py
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_generate.py
├── requirements.txt
└── README.md
```

| 文件 | 功能 |
|---|---|
| model.py | 模型加载及配置检查 |
| generate.py | 基础文本生成 |
| inspect_shapes.py | 模型各计算模块的张量形状检查 |
| benchmark_generate.py | 模型推理性能测试 |
| requirements.txt | Python 依赖配置 |

## 6. 开发进度

| 任务 | 内容 | 状态 |
|---|---|---|
| TASK001 | 工程初始化与环境部署 | 已完成 |
| TASK002 | Qwen2.5 基准推理 | 已完成 |
| TASK003 | PyTorch CUDA 扩展接口 | 待开发 |
| TASK004 | RMSNorm | 待开发 |
| TASK005 | SwiGLU | 待开发 |
| TASK006 | RoPE | 待开发 |
| TASK007 | Softmax | 待开发 |
| TASK008 | GEMV 深度优化 | 待开发 |
| TASK009 | GEMM 模型适配 | 待开发 |
| TASK010 | Prefill Attention | 待开发 |
| TASK011 | Decode Attention | 待开发 |
| TASK012 | 综合集成与性能分析 | 待开发 |
| TASK013 | 工程完善与项目总结 | 待开发 |

## 7. TASK001：工程初始化与环境部署

### 完成内容

- 初始化 MiniLLM-CUDA 项目结构。
- 配置 CUDA C++ 开发环境。
- 配置 Python 虚拟环境及模型推理依赖。
- 完善统一环境部署脚本。
- 完善 NVIDIA 驱动保护及自动更新管理。
- 配置 Git 文件管理规则。

### 环境验证

当前开发实例已通过以下测试：

| 测试项目 | 结果 |
|---|---|
| 原生 CUDA 编译及计算正确性 | PASS |
| PyTorch CUDA FP16 矩阵乘法 | PASS |
| PyTorch CUDA Extension 编译及调用 | PASS |
| CUDA 非默认 Stream 调用 | PASS |
| Nsight Compute 性能计数器 | PASS |

统一部署脚本已在当前实例完成验证，完整的新实例环境重建仍需进一步验证。

## 8. TASK002：Qwen2.5 基准推理

### 实验目标

完成原始模型的加载、文本生成、张量形状检查及推理性能测试，为后续自研 CUDA 算子集成建立统一的性能对照基线。

### 测试配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Ti |
| 数据类型 | FP16 |
| Attention Backend | SDPA |
| Batch Size | 1 |
| 输入 Token 数量 | 37 |
| 输出 Token 数量 | 128 |
| KV Cache | 开启 |
| 生成策略 | 贪心解码 |
| 预热次数 | 2 |
| 正式测试次数 | 5 |

测试输入：

```text
What is CUDA? Explain it briefly.
```

### 张量形状

以 Prefill 阶段的 37 个输入 Token 为例：

| 计算模块 | 输入形状 | 输出形状 |
|---|---|---|
| Embedding | [1, 37] | [1, 37, 896] |
| RMSNorm | [1, 37, 896] | [1, 37, 896] |
| Q Projection | [1, 37, 896] | [1, 37, 896] |
| K Projection | [1, 37, 896] | [1, 37, 128] |
| V Projection | [1, 37, 896] | [1, 37, 128] |
| Output Projection | [1, 37, 896] | [1, 37, 896] |
| Gate Projection | [1, 37, 896] | [1, 37, 4864] |
| Up Projection | [1, 37, 896] | [1, 37, 4864] |
| Down Projection | [1, 37, 4864] | [1, 37, 896] |
| Final RMSNorm | [1, 37, 896] | [1, 37, 896] |
| LM Head | [1, 1, 896] | [1, 1, 151936] |

在启用 KV Cache 的单请求 Decode 阶段，每次前向计算仅处理一个新 Token，并复用历史 K、V 数据。

### 原始模型性能基线

测试日期：2026-11-27

| 性能指标 | 实测结果 | 标准差 |
|---|---:|---:|
| TTFT（首词元延迟） | 24.751 ms | 0.154 ms |
| TPOT（每输出词元耗时） | 22.311 ms/token | 0.344 ms/token |
| Total Latency（总生成延迟） | 2.859 s | 0.044 s |
| Overall Throughput（整体吞吐量） | 44.785 tokens/s | 0.673 tokens/s |
| Decode Throughput（解码吞吐量） | 44.829 tokens/s | 0.680 tokens/s |
| Peak GPU Memory（峰值显存分配量） | 0.943 GiB | - |

峰值显存分配量取五次正式测试中的最大值。

### 测试方法

- 使用原始 Transformers `model.generate()` 完成文本生成。
- 使用 CPU 墙钟时间进行计时。
- 使用 Streamer 记录每个输出 Token 的生成时刻。
- 计时包含框架调度、核函数启动及 Token 回传开销。
- 不包含模型加载、分词、文本解码及网络传输时间。
- 后续优化版本采用相同测试方式进行性能对比。

### 性能分析

- 各计时指标的相对标准差均低于 2%，测试重复性较好。
- 当前短输入、长输出场景下，Decode 阶段占据约 99% 的生成时间。
- GEMV 与 Decode Attention 是后续值得重点研究的优化方向。
- 具体算子瓶颈需通过进一步性能分析确认。

### 运行方式

```bash
cd /workspace/CUDA/minillm-cuda

source .venv/bin/activate

# 模型加载及配置检查
python -m minillm.model

# 基础文本生成
python -m minillm.generate

# 张量形状检查
python -m minillm.inspect_shapes

# 模型推理性能测试
python -m benchmarks.benchmark_generate
```

## 9. 工程规范

- 自定义算子采用 CUDA C++ 实现。
- 各正式算子版本独立保存。
- 统一进行正确性验证及性能测试。
- 保留原始模型作为端到端性能对照基线。
- Nsight Compute 性能报告统一存放于对应的 ncu/ 目录。
- 不提交模型权重、虚拟环境、编译产物及运行日志。