
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

| 版本 | M=1 | M=37 | M=512 | M=2048 | M=4096 |
|---|---:|---:|---:|---:|---:|
| PyTorch | 32.128 | 31.680 | 31.120 | 47.616 | 84.848 |
| FlashInfer | 12.576 | 12.080 | 12.128 | 12.448 | 15.904 |
| V0 | 6.880 | 6.656 | 6.592 | 12.640 | 20.032 |
| V1 | 6.880 | 6.624 | 6.560 | 10.784 | 17.872 |

以上数据均来自 V1 阶段的同轮性能测试。

### Kernel 性能

Nsight Compute 计时，单位为 us。

| 版本 | Decode | Prefill | Decode/FI | Prefill/FI |
|---|---:|---:|---:|---:|
| FlashInfer | 2.720 | 12.672 | 100% | 100% |
| V0 | 4.736 | 26.624 | 57.43% | 47.60% |
| V1 | 4.608 | 25.120 | 59.03% | 50.45% |

Decode：M = 1；Prefill：M = 4096。

相对性能 = FlashInfer 耗时 / 自研 Kernel 耗时。

Benchmark 与 NCU 的计时口径不同。
FlashInfer 与 Qwen2RMSNorm 的中间舍入顺序存在差异。
V4、V5 独立进行精度及融合性能对比。

## 5. 版本分析

### V0 - Shared Memory Reduction

**实现方法**

- 每个 Block 处理一行，256 个线程。
- 标量访存，FP32 平方和累加。
- 共享内存树形归约，10 次 Block 级同步。

**瓶颈判断**

- Prefill 指令总数为 FlashInfer 的 5.49 倍。
- Prefill 共享内存访问波前数为 FlashInfer 的 3.56 倍。
- Decode 存在明显的全局访存依赖等待。
- 两种场景均存在较明显的同步屏障等待。
- Prefill 显存吞吐率为 31.33%，尚未达到带宽饱和。

主要优化方向为减少同步、归约及访存指令开销。

### V1 - Synchronization Optimization

**实现方法**

- 保持 V0 的线程映射、访存方式及计算精度。
- 跨 Warp 归约使用 Block 级同步。
- Warp 内部归约使用 volatile 共享内存访问及 Warp 级同步。
- Block 同步次数由 10 次减少至 5 次。

**关键 NCU 指标**

以下为 FP16 Prefill 场景。

| 指标 | V0 | V1 |
|---|---:|---:|
| Instructions Executed | 6545408 | 5705728 |
| Shared Memory Wavefronts | 934585 | 484732 |
| Shared Load/Store Instructions | 221184 | 221184 |
| Long Scoreboard | 9.79 | 11.72 |
| Short Scoreboard | 2.85 | 1.72 |
| Barrier | 6.74 | 8.47 |
| MIO Throttle | 1.47 | 0.47 |
| DRAM Throughput (%) | 31.33 | 33.19 |

Stall 指标为每条发射指令对应的 Warp 等待周期数，
不代表 Kernel 总执行时间占比。

**瓶颈判断与实验结论**

- 指令总数减少 12.83%。
- 共享内存访问波前数减少约 48%。
- 共享内存读写指令数量保持不变。
- Barrier 等待周期比值上升，未达到预期目标。
- Long Scoreboard 仍然较高，访存依赖等待仍需优化。

NCU 测得 Decode 和 Prefill 性能分别提升 2.78% 和 5.99%。
Benchmark 中 FP16、M=4096、N=896 的性能提升 12.09%。

本次优化收益有限，尚不能将全部性能变化归因于同步次数减少。

保留 V1 作为后续 V2 的开发基线，
进一步使用 Warp Shuffle 减少共享内存归约开销。

## 6. 正确性验证

| 版本 | Correctness | Memcheck | Racecheck | Synccheck |
|---|---|---|---|---|
| V0 | 61/61 PASS | PASS | PASS | 未测试 |
| V1 | 61/61 PASS | PASS | PASS | PASS |

## 7. 运行方法

在 minillm-cuda 目录下执行：

```bash
python -m tests.test_rmsnorm --version v0
python -m tests.test_rmsnorm --version v1
python -m benchmarks.benchmark_rmsnorm
```

NCU 原始报告及 CSV 数据按版本保存在本地 ncu/ 目录，
不纳入 Git 版本控制。
