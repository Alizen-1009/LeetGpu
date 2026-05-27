# Reduction with Nsight Compute

这个目录现在只保留一条最直接的工作流：

1. `python Nsight-learn/profile_reduce_ncu.py`
   负责构造输入、预热、并用 `cudaProfilerStart()/cudaProfilerStop()` 只圈住目标 kernel。
2. `bash Nsight-learn/run_ncu_reduce.sh`
   负责直接调用 `ncu`，生成 `.ncu-rep`，再从 report 导出一份文本摘要。

## 为什么不再用 two-step attach

`two-step attach` 适合这几种情况：

- 目标进程不是你直接启动的
- 目标进程寿命很短，`launch-and-attach` 来不及接管
- 你需要手动连到某个已经在跑的 CUDA 进程

这里的场景并不复杂：我们自己启动 Python，且只想抓一小段 reduce kernel。  
直接用一条 `ncu ... python ...` 命令，再配合 `--profile-from-start off` 和 `cudaProfilerStart()/cudaProfilerStop()`，就已经足够稳定，也更符合 Nsight Compute 的常见用法。

## 最简单的用法

先确认当前 `python` 能导入 `torch`，而且 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果你的 `torch` 在某个 conda/env 里，可以直接指定解释器：

```bash
PYTHON=/path/to/env/bin/python bash Nsight-learn/run_ncu_reduce.sh float4 16777216
```

然后直接跑：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216
```

默认会生成两个文件：

```bash
Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep
Nsight-learn/ncu-reports/reduce_float4_16777216.txt
```

其中：

- `.ncu-rep` 给 `ncu-ui` 打开做可视化
- `.txt` 是从 report 导出的命令行摘要，方便快速看结果
- 默认只抓自定义 reduce kernel：`reduce_sum_kernel` 和 `sum_kernel`。这样可以避开 Python、allocator、随机数生成、`torch::zeros` 等不相关 kernel。

## 常用命令

基础指标：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
```

更完整的分析：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set full
```

只看 `reduce_sum_kernel`：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic --kernel-name regex:reduce_sum_kernel
```

指定架构：

```bash
ARCH=8.9 bash Nsight-learn/run_ncu_reduce.sh float4 16777216
```

增加采样次数：

```bash
PROFILE_ITERS=10 bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
```

学习 NCU 时建议先保持 `PROFILE_ITERS=1`。NCU 更适合看单个 kernel 的硬件指标；多次迭代会生成多个 kernel result，报告会变大，也不等同于普通 benchmark 的平均耗时。后面学 `torch perf` 时再专门做稳定计时。

## 打开 report

命令行查看：

```bash
ncu --import Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep --page details
```

图形界面查看：

```bash
ncu-ui Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep
```

## 如果权限不够

如果看到 `ERR_NVGPUCTRPERM` 或类似报错，说明当前用户没有 GPU performance counters 权限。  
这种情况需要管理员放开 profiling 权限，或者临时用有权限的用户运行 `ncu`。

## 看 reduce kernel 时重点看什么

先从这几块开始，不用一上来盯所有指标：

- `GPU Speed Of Light`：看整体吞吐，尤其是 memory throughput 和 compute throughput 谁更接近上限。
- `Memory Workload Analysis`：reduce 通常是 memory-bound，重点看 global load/store、L2、DRAM throughput。
- `Launch Statistics`：确认 grid/block 配置，例如 block 数、每 block 线程数是否符合预期。
- `Occupancy`：看理论/实际 occupancy、寄存器和 shared memory 是否限制并发。
- `Scheduler Statistics` / `Warp State Statistics`：如果吞吐不高，看 warp 在等内存、同步，还是调度不足。

当前 `float4` 名字有点容易误导：`reduction_float4.cu` 现在走的是两阶段 reduce，但第一阶段仍是单 float load；真正用 `float4` 向量化 load 的是 `float4_atomic`。所以对比时可以跑：

```bash
bash Nsight-learn/run_ncu_reduce.sh naive 16777216 --set basic
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
bash Nsight-learn/run_ncu_reduce.sh float4_atomic 16777216 --set basic
```
