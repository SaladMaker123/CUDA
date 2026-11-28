
# CUDA RMSNorm Optimization

## 1. 项目介绍

面向 LLM 推理场景的 RMSNorm CUDA 算子优化，
支持 FP16 / FP32，普通版本采用 FP32 累加，
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
| V1 | 归约同步优化 | 完成 |
| V2 | Warp Shuffle 归约 | 完成 |
| V3 | 向量化访存 | 待完成 |
| V4 | 混合精度实验 | 待完成 |
| V5 | Block Size 参数调优 | 待完成 |

算子融合推迟至模型集成后的专项优化阶段。

## 4. 性能对比

测试精度：FP16，N = 896。

### Benchmark

CUDA Event 计时，Warmup = 20，Repeat = 100，
取中位数，单位为 us。

| 版本 | M=1 | M=37 | M=512 | M=2048 | M=4096 |
|---|---:|---:|---:|---:|---:|
| PyTorch | 31.952 | 31.392 | 31.504 | 47.552 | 85.760 |
| FlashInfer | 12.576 | 12.288 | 12.000 | 12.496 | 15.856 |
| V0 | 6.944 | 6.496 | 6.560 | 12.544 | 20.000 |
| V1 | 6.880 | 6.528 | 6.528 | 11.808 | 17.856 |
| V2 | 6.880 | 6.464 | 6.528 | 9.824 | 16.736 |

以上数据均来自 V2 阶段的同轮性能测试。

### Kernel 性能

Nsight Compute 计时，单位为 us。

| 版本 | Decode | Prefill | Decode/FI | Prefill/FI |
|---|---:|---:|---:|---:|
| FlashInfer | 2.720 | 12.672 | 100% | 100% |
| V0 | 4.736 | 26.624 | 57.43% | 47.60% |
| V1 | 4.608 | 25.120 | 59.03% | 50.45% |
| V2 | 4.544 | 23.904 | 59.86% | 53.01% |

Decode：M = 1；Prefill：M = 4096。

相对性能 = FlashInfer 耗时 / 自研 Kernel 耗时。

Benchmark 与 NCU 的计时口径不同。
FlashInfer 与 Qwen2RMSNorm 的浮点舍入顺序存在差异。

## 5. NCU 核心指标

FP16，M = 4096，N = 896。

| 指标 | V0 | V1 | V2 | FlashInfer |
|---|---:|---:|---:|---:|
| Instructions Executed | 6545408 | 5705728 | 5623808 | 1191936 |
| Global Load Requests | 344064 | 344064 | 344064 | 32768 |
| Shared Load/Store Instructions | 221184 | 221184 | 73728 | 32768 |
| Shared Memory Wavefronts | 934585 | 484732 | 299925 | 262431 |
| Long Scoreboard | 9.79 | 11.72 | 11.74 | 13.62 |
| Short Scoreboard | 2.85 | 1.72 | 1.41 | 9.59 |
| Barrier | 6.74 | 8.47 | 7.78 | 2.29 |
| MIO Throttle | 1.471 | 0.467 | 0.074 | 15.938 |
| DRAM Throughput (%) | 31.33 | 33.19 | 34.95 | 66.08 |

Stall 指标为每条发射指令对应的 Warp 等待周期数，
不代表 Kernel 总执行时间占比。

## 6. 版本分析

### V0 - Shared Memory Reduction

实现方法：

- 一个 Block 处理一行，256 个线程。
- 标量访存，FP32 平方和累加。
- 共享内存树形归约，10 次 Block 级同步。

瓶颈判断：

- Prefill 指令总数为 FlashInfer 的 5.49 倍。
- 共享内存访问及同步等待开销较大。
- 全局访存依赖等待明显，尚无显存带宽饱和证据。

### V1 - Synchronization Optimization

实现方法：

- 跨 Warp 归约使用 Block 级同步。
- Warp 内部归约使用 volatile 共享内存访问及 Warp 级同步。
- Block 同步次数由 10 次减少至 5 次。

实验结论：

- Prefill 指令总数减少 12.83%。
- 共享内存访问波前数减少约 48%。
- Barrier 等待周期比值不降反升。
- Decode 和 Prefill Kernel 性能分别提升 2.78% 和 5.99%。

本次优化有效，但同步等待问题仍未充分解决。

### V2 - Warp Shuffle Reduction

实现方法：

- 使用 Warp Shuffle 完成两级归约。
- 共享内存容量由 1024 B 减少至 32 B。
- Block 同步次数由 5 次减少至 2 次。

实验结论：

- Prefill 共享内存读写指令减少 66.67%。
- 共享内存访问波前数减少 38.13%。
- MIO Throttle 等待周期比值下降 84.17%。
- 整体指令数量仅减少 1.44%。
- Decode 和 Prefill Kernel 性能分别提升 1.41% 和 5.09%。
- FP16、M=2048、N=896 的 Benchmark 性能提升约 20.2%。

部分大隐藏层维度测试存在性能下降。
全局访存请求数量不变，Long Scoreboard 仍然较高。

V2 作为 V3 开发基线，后续重点优化访存指令数量。
具体瓶颈贡献及微小性能收益的稳定性仍需进一步验证。

## 7. 正确性验证

| 版本 | Correctness | Memcheck | Racecheck | Synccheck |
|---|---|---|---|---|
| V0 | 61/61 PASS | PASS | PASS | 未测试 |
| V1 | 61/61 PASS | PASS | PASS | PASS |
| V2 | 61/61 PASS | PASS | PASS | PASS |

## 8. 运行方法

在 minillm-cuda 目录下执行：

```bash
python -m tests.test_rmsnorm --version v2
python -m benchmarks.benchmark_rmsnorm
```

NCU 原始报告及 CSV 数据按版本保存在本地 ncu/ 目录，
不纳入 Git 版本控制。
