
# CUDA RMSNorm Optimization

## 1. 项目介绍

面向 LLM 推理场景的 RMSNorm CUDA 算子优化，
支持 FP16 / FP32，采用 FP32 累加，
最终集成至 MiniLLM-CUDA。

## 2. 实验环境

- GPU：RTX 5070 Ti（Blackwell，SM 12.0）
- CUDA：13.0
- PyTorch：2.9.1+cu130
- FlashInfer：0.7.6
- Nsight Compute：2025.2.0

## 3. 优化路线

| 版本 | 优化方法 | 状态 |
|---|---|---|
| V0 | 共享内存树形归约 | 完成 |
| V1 | 归约同步优化 | 待完成 |
| V2 | Warp Shuffle 归约 | 待完成 |
| V3 | 向量化访存 | 待完成 |
| V4 | 混合精度实验 | 待完成 |
| V5 | Residual RMSNorm 融合 | 待完成 |
| V6 | Block Size 参数调优 | 待完成 |

## 4. 性能对比

测试精度：FP16，N = 896。

### Benchmark

CUDA Event 计时，Warmup = 20，Repeat = 100，
取中位数，单位为 us。

| 版本 | M=1 | M=37 | M=512 | M=4096 |
|---|---:|---:|---:|---:|
| PyTorch | 31.888 | 31.296 | 31.120 | 85.216 |
| FlashInfer | 12.448 | 11.936 | 11.840 | 15.872 |
| V0 | 6.880 | 6.688 | 6.592 | 19.968 |

### Kernel 性能

Nsight Compute 计时，单位为 us。

| 版本 | Decode | Prefill | Decode/FI | Prefill/FI |
|---|---:|---:|---:|---:|
| FlashInfer | 2.720 | 12.672 | 100% | 100% |
| V0 | 4.736 | 26.624 | 57.43% | 47.60% |

Decode：M = 1；Prefill：M = 4096。

相对性能 = FlashInfer 耗时 / 自研 Kernel 耗时。

FlashInfer 与 Qwen2RMSNorm 的中间舍入顺序存在差异。
V4、V5 独立进行精度及融合性能对比。

## 5. 版本分析

### V0 - Shared Memory Reduction

**实现方法**

- 每个 Block 处理一行，256 个线程。
- 标量访存，FP32 平方和累加。
- 共享内存树形归约，10 次 Block 级同步。

**正确性**

- Correctness：61/61 PASS
- Memcheck / Racecheck：无错误

**关键 NCU 指标**

| 指标 | V0 Decode | FI Decode | V0 Prefill | FI Prefill |
|---|---:|---:|---:|---:|
| Instructions Executed | 1598 | 690 | 6545408 | 1191936 |
| Shared Memory Wavefronts | 228 | 145 | 934585 | 262431 |
| Long Scoreboard | 21.92 | 3.81 | 9.79 | 13.62 |
| Barrier | 5.23 | 0.15 | 6.74 | 2.29 |
| DRAM Throughput (%) | 0.31 | 0.54 | 31.33 | 66.08 |

Stall 指标单位为每条发射指令对应的 Warp 等待周期数。

**瓶颈判断**

- Decode 场景存在较明显的全局访存依赖等待。
- 两种场景均存在较明显的 Block 同步等待。
- Prefill 指令总数为 FlashInfer 的 5.49 倍。
- Prefill 共享内存访问波前数为 FlashInfer 的 3.56 倍。
- DRAM 读取量接近，但吞吐率明显低于 FlashInfer。

**实验结论**

主要优化方向为减少同步等待、归约开销及访存指令数量。
各项开销的实际影响需要通过后续控制变量实验验证。

V1 优先缩小归约阶段的同步范围。

## 6. 运行方法

在 minillm-cuda 目录下执行：

```bash
python -m tests.test_rmsnorm --version v0
python -m benchmarks.benchmark_rmsnorm
```

NCU 原始报告及 CSV 数据见 [ncu/](ncu/)。
