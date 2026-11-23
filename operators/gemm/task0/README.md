# Task 0: GPU Device Information

## 1. 实验目标

使用 CUDA Runtime API 查询 RTX 5070 Ti 的硬件参数，计算理论 FP32 算力与显存带宽，为后续 GEMM 性能分析提供硬件基线。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Ti |
| Architecture | Blackwell (SM 120) |
| CUDA Toolkit | 13.0 |
| NVIDIA Driver | 580.95.05 |
| OS | Ubuntu 22.04 |

## 3. 硬件参数

| 参数 | 数值 |
|---|---:|
| SM Count | 70 |
| CUDA Cores | 8,960 |
| Core Clock | 2,452 MHz |
| Memory Clock | 14,001 MHz |
| Global Memory | 16 GB |
| L2 Cache | 48 MiB |
| Memory Bus Width | 256 bit |
| Shared Memory / SM | 100 KiB |
| Shared Memory / Block (Opt-in) | 99 KiB |
| Registers / SM | 65,536 |
| Max Threads / SM | 1,536 |
| Max Blocks / SM | 24 |
| Warp Size | 32 |

## 4. 理论性能

### 4.1 FP32 FMA 峰值

每次 FMA 计为 2 FLOPs。

**FP32 Throughput = CUDA Cores × 2 × Core Clock**

8960 × 2 × 2.452 GHz = **43.94 TFLOPS**

### 4.2 显存带宽

GDDR7 有效传输速率约为 28 Gbps/pin，显存位宽为 256 bit。

**Memory Bandwidth = Transfer Rate × Bus Width / 8**

28.002 × 256 / 8 = **896.06 GB/s**

以上结果为理论估算，并非实际持续性能。

## 5. 实验结论

RTX 5070 Ti 具有 70 个 SM、48 MiB L2 Cache 和每 SM 100 KiB Shared Memory。

后续任务：

- **Task 1：** 测量实际内存带宽与访问延迟。
- **Task 2：** 测量 FP32 FMA 指令吞吐与延迟。
- **Task 3：** 建立 GEMM 性能模型，指导 Tile 参数选择。