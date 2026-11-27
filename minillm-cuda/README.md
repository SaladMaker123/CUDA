# MiniLLM-CUDA

面向真实大语言模型推理场景的 CUDA C++ 算子开发与性能优化项目。

基于 PyTorch 和 Hugging Face Transformers，将自研 CUDA 算子集成至 Qwen2.5 模型，实现算子级优化及端到端推理性能分析。

## 开发环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Ti |
| GPU 架构 | Blackwell（SM120） |
| 操作系统 | Ubuntu 22.04 |
| NVIDIA Driver | 580.95.05 |
| CUDA Toolkit | 13.0.3 |
| Python | 3.10 |
| PyTorch | 2.9.1+cu130 |
| Transformers | 4.57.6 |
| Nsight Compute | 2025.2.0 |

## 环境部署

使用 CUDA 总仓库提供的统一环境部署脚本：

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/SaladMaker123/CUDA.git
cd CUDA
bash scripts/setup_env.sh
```

如需安装指定版本的 Nsight Compute 并检查性能计数器权限：

```bash
bash scripts/setup_env.sh --with-ncu
```

激活 Python 虚拟环境：

```bash
source minillm-cuda/.venv/bin/activate
```

## 目标模型

Qwen2.5-0.5B-Instruct

- FP16 推理
- FP32 归约累加
- Batch Size = 1
- Greedy Decoding

## 核心算子

| 算子 | 主要优化方向 |
|---|---|
| RMSNorm | 并行归约、向量化访存 |
| SwiGLU | 算子融合 |
| RoPE | 线程映射、数据布局 |
| Softmax | 并行归约、Online Softmax |
| GEMV | 显存带宽、数据复用 |
| GEMM | 模型形状及权重布局适配 |
| Prefill Attention | 分块计算、FlashAttention |
| Decode Attention | GQA、KV Cache、Split-KV |

## 开发进度

| 任务 | 内容 | 状态 |
|---|---|---|
| TASK001 | 工程初始化与环境部署 | 已完成 |
| TASK002 | Qwen2.5 基准推理 | 待开发 |
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

## 环境验证

当前开发实例已通过以下测试：

- 原生 CUDA 编译及计算正确性测试。
- PyTorch CUDA FP16 矩阵乘法测试。
- PyTorch CUDA Extension 编译及调用测试。
- Nsight Compute 性能计数器测试。
