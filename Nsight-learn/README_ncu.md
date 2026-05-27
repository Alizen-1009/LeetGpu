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
