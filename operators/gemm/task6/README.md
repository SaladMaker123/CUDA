# TASK6：HGEMM 深度优化

## 实验环境

- GPU：NVIDIA RTX 5070 Ti
- CUDA：13.0
- Architecture：sm_120
- 输入精度：FP16
- 累加与输出精度：FP32

## V1：TMA 与多 Warp 分块

### 优化方法

- Block Tile 扩展至 128 × 128 × 32，提高数据复用率。
- 每个 Block 包含 8 个 Warp，划分为 4 个计算组。
- 每个 Warp 负责 32 × 64 的输出区域，维护 8 个累加片段。
- 使用 TMA 完成全局内存到共享内存的数据搬运。
- 采用 4 × 2 Block Swizzle，改善 L2 Cache 数据复用。
- 使用单缓冲结构，通过 Barrier 保证搬运与计算的正确性。

### 性能结果

测试规模：M = N = K = 4096，预热 10 次，计时 20 次。

| 版本 | 执行时间 (ms) | TFLOPS | 正确性 |
|---|---:|---:|---|
| V1 | 2.031906 | 67.640424 | PASS |

cuBLAS 最大绝对误差：0。

### 后续优化方向

当前采用单缓冲结构，TMA 搬运与 Tensor Core 计算未实现重叠。后续通过多级缓冲及流水线调度进一步提高计算效率。

## 编译与运行

```bash
make VERSION=1
make run VERSION=1
```