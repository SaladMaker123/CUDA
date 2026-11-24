# TASK4 — FP32 GEMM 深度优化

## 1. 实验环境

- GPU：NVIDIA GeForce RTX 5070 Ti
- 架构：Blackwell SM120
- CUDA Toolkit：13.0
- 计算精度：FP32 输入、FP32 累加、FP32 输出
- 性能基准：cuBLAS SGEMM

## 2. 版本规划

| 版本 | 优化方法 | 状态 |
|---|---|---|
| V0 | Naive GEMM | 已完成 |
| V1 | Shared Memory Tiling | 待完成 |
| V2 | Register Blocking | 待完成 |
| V3 | cp.async Double Buffering | 待完成 |

## 3. V0 — Naive GEMM

采用二维线程块，每个线程独立计算输出矩阵中的一个元素，直接从全局内存读取 A、B 矩阵数据并完成乘加运算。

### 分块参数

| 参数 | 数值 |
|---|---:|
| BM | 16 |
| BN | 16 |
| TM | 1 |
| TN | 1 |
| Threads per Block | 256 |

### 性能测试

测试规模：M=N=K=4096。

| 实现 | 执行时间（ms） | TFLOPS | 相对 cuBLAS 性能 |
|---|---:|---:|---:|
| V0 | 49.562 | 2.773 | 8.143% |
| cuBLAS | 4.036 | 34.056 | 100% |

测试采用 CUDA Event 计时，预热 10 次，每轮重复测试 20 次，进行三轮交替测试并取中位数。

### 正确性验证

| 矩阵规模 | 最大绝对误差 | 验证结果 |
|---|---:|---|
| 4096×4096×4096 | 3.585815e-04 | PASS |
| 513×769×257 | 1.621246e-05 | PASS |

### 初步分析

当前实现未使用共享内存或寄存器分块，存在大量重复的全局内存读取。后续将通过共享内存分块提高数据复用率。

具体性能瓶颈将在后续统一进行 Nsight Compute 分析时确认。

## 4. 编译与运行

```bash
make -j$(nproc)
./cublas_benchmark.x 4096
./sgemm-v0.x 4096
```