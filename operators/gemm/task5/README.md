# TASK5：Tensor Core HGEMM

## 1. 实验目标

基于 CUDA WMMA 实现 FP16 输入、FP32 累加与输出的矩阵乘法，建立后续 HGEMM 性能优化的基础版本。

## 2. Kernel 设计

- Block Tile：64 × 64 × 64
- Warp Tile：32 × 32
- WMMA Tile：16 × 16 × 16
- Threads per Block：128
- Shared Memory：16 KiB

每个 Block 包含 4 个 Warp，分别负责输出矩阵的四个 32 × 32 区域。

线程协作加载 A、B 至共享内存，通过 WMMA 执行矩阵乘加，使用 FP32 Fragment 完成跨 K 分块累加，最终直接写回全局内存。

## 3. 实验结果

测试平台：NVIDIA GeForce RTX 5070 Ti

| M | N | K | Time (ms) | TFLOPS |
|---:|---:|---:|---:|---:|
| 128 | 128 | 128 | 0.012480 | 0.336082 |
| 4096 | 4096 | 4096 | 5.028723 | 27.330785 |

128³ 矩阵正确性验证通过，与 cuBLAS 参考结果的最大绝对误差为 0。

4096³ 矩阵实测吞吐量为 27.33 TFLOPS。

## 4. 性能分析

当前版本尚未采用向量化加载、异步数据搬运及多阶段流水线，数据搬运与计算串行执行，存在进一步优化空间。

## 5. 编译与运行

```bash
make
make run
```

本版本作为后续 HGEMM 优化的性能基准。