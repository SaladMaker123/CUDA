# MiniLLM-CUDA

基于 PyTorch 和 Transformers 的 Qwen2.5 推理算子开发与性能优化项目。

通过自研 CUDA C++ 算子实现算子级性能优化及端到端推理性能分析。

## 1. 开发环境

| 项目 | 配置 |
|---|---|
| GPU | RTX 5070 Ti（SM120） |
| OS | Ubuntu 22.04 |
| CUDA | 13.0.3 |
| Python | 3.10.12 |
| PyTorch | 2.9.1+cu130 |
| Transformers | 4.57.6 |
| Nsight Compute | 2025.2.0 |

## 2. 环境部署

```bash
mkdir -p /workspace
cd /workspace
git clone https://github.com/SaladMaker123/CUDA.git
cd CUDA

bash scripts/setup_env.sh --with-ncu --pause-auto-updates

source /etc/profile.d/cuda-gemm-env.sh
cd minillm-cuda
source .venv/bin/activate

export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2
```

暂停自动更新后需自行安排必要的系统安全更新。

## 3. 模型配置

| 参数 | 数值 |
|---|---|
| 模型 | Qwen2.5-0.5B-Instruct |
| 参数量 | 494M |
| Transformer 层数 | 24 |
| 隐藏维度 | 896 |
| 中间维度 | 4864 |
| Q / KV Head | 14 / 2 |
| Head Dimension | 64 |
| 推理精度 | FP16 |
| Attention Backend | SDPA |
| KV Cache | 开启 |

## 4. 工程结构

```text
CUDA/
├── operators/          # CUDA 算子源码
├── scripts/            # 环境部署脚本
└── minillm-cuda/
    ├── csrc/           # C++ 绑定接口
    ├── minillm/        # 模型及扩展加载
    ├── tests/          # 正确性测试
    ├── benchmarks/     # 性能测试
    └── README.md
```

调用链路：

Python → C++ Binding → CUDA Launcher → CUDA Kernel

## 5. 开发进度

| 任务 | 内容 | 状态 |
|---|---|---|
| TASK001 | 环境部署 | 完成 |
| TASK002 | 原始模型推理基线 | 完成 |
| TASK003 | PyTorch CUDA 扩展 | 完成 |
| TASK004 | RMSNorm | 待开发 |
| TASK005 | SwiGLU | 待开发 |
| TASK006 | RoPE | 待开发 |
| TASK007 | Softmax | 待开发 |
| TASK008 | GEMV | 待开发 |
| TASK009 | GEMM 模型适配 | 待开发 |
| TASK010 | Prefill Attention | 待开发 |
| TASK011 | Decode Attention | 待开发 |
| TASK012 | 综合集成与性能分析 | 待开发 |
| TASK013 | 工程完善与项目总结 | 待开发 |

## 6. 原始模型性能基线

RTX 5070 Ti，FP16，Batch Size = 1。

输入 37 Token，固定输出 128 Token，预热 2 次，测试 5 次。

| 性能指标 | 平均值 | 标准差 |
|---|---:|---:|
| TTFT（首词元延迟） | 24.751 ms | 0.154 ms |
| TPOT（每输出词元耗时） | 22.311 ms/token | 0.344 ms/token |
| 总生成延迟 | 2.859 s | 0.044 s |
| 整体吞吐量 | 44.785 tokens/s | 0.673 tokens/s |
| 解码吞吐量 | 44.829 tokens/s | 0.680 tokens/s |
| 峰值显存分配量 | 0.943 GiB | - |

测试采用 Streamer 记录 Token 生成时刻，包含框架调度、
核函数启动及 Token 回传开销，不包含模型加载及分词时间。

模型权重版本哈希尚未记录。

## 7. CUDA 扩展验证

通过 Vector Add 算子建立统一 CUDA 扩展调用接口。

- PyTorch JIT 编译及加载成功。
- 支持连续存储的 FP32 张量。
- 张量形状测试：11/11 PASS。
- 非法输入测试：4/4 PASS。
- 非默认 CUDA Stream 测试：PASS。
- 最大绝对误差：0.0。

当前仅完成接口验证，尚未替换模型内部算子。

## 8. 运行方式

```bash
# 模型配置检查
python -m minillm.model

# 文本生成
python -m minillm.generate

# 张量形状检查
python -m minillm.inspect_shapes

# 推理性能测试
python -m benchmarks.benchmark_generate

# CUDA 扩展编译及加载
python -m minillm.extension

# 向量加法正确性验证
python -m tests.test_vector_add
```