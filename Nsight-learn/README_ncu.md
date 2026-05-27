# Reduction with Nsight Compute

这个目录里的 `profile_reduce_ncu.py` 和 `run_ncu_reduce.sh` 是给 `ncu` 学习和单算子分析准备的。

当前版本默认假设：

- 学习脚本放在 `Nsight-learn/`
- reduce 的 `.cu` 实现还放在 `reduction/`

## 1. 先准备 Python 环境

要在你平时能 `import torch` 的环境里运行，例如：

```bash
conda activate <your-env>
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果这里 `torch.cuda.is_available()` 不是 `True`，先不要往下跑 `ncu`。

## 2. 先做一次普通 benchmark

第一次建议先不用 `ncu`，先把 PyTorch extension 编译好，同时确认算子结果正常：

```bash
python Nsight-learn/profile_reduce_ncu.py --mode bench --impl float4 --size 16777216 --warmup 20 --iters 200 --check
```

如果你的卡需要显式指定架构，比如你之前脚本里常写的 `8.9`，可以加：

```bash
python Nsight-learn/profile_reduce_ncu.py --mode bench --impl float4 --size 16777216 --arch 8.9 --check
```

可选实现：

- `naive`
- `float4`
- `float4_atomic`
- `torch`

这个步骤会输出平均延迟、近似读取带宽，以及和 `torch.sum` 的结果误差。

## 3. 再用 ncu 抓 profile

最简单的跑法：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216
```

这个脚本内部会执行：

- `--profile-from-start off`
- `cudaProfilerStart()/cudaProfilerStop()`

所以 `ncu` 只会抓我们手动圈起来的 reduce 调用，不会把前面的 warmup 一起算进去。

生成的 report 默认在：

```bash
Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep
```

## 4. 常用 ncu 命令

快速看基础指标：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
```

看更全的分析：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set full
```

只看 `reduce_sum_kernel`：

```bash
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic --kernel-name regex:reduce_sum_kernel
```

如果你想对比不同实现：

```bash
bash Nsight-learn/run_ncu_reduce.sh naive 16777216 --set basic
bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
bash Nsight-learn/run_ncu_reduce.sh float4_atomic 16777216 --set basic
```

## 5. 打开 report

命令行看摘要：

```bash
ncu --import Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep --page details
```

图形界面看 report：

```bash
ncu-ui Nsight-learn/ncu-reports/reduce_float4_16777216.ncu-rep
```

## 6. 第一次建议重点看什么

- `LaunchStats`：grid/block 配置是否合理。
- `Occupancy`：活跃 warps 是否被寄存器、shared memory 或 block size 限住。
- `SpeedOfLight`：更偏向 memory bound 还是 compute bound。
- `MemoryWorkloadAnalysis`：全局内存访问是否连续、吞吐有没有吃满。

对这个 reduce 来说，第一次最值得对比的是：

- `naive` 和 `float4` 的吞吐差异
- `float4_atomic` 里 `atomicAdd` 带来的串行化影响
- 两阶段 reduce 和单阶段 atomic reduce 的 kernel 时间构成

## 7. 一点经验

- 第一次运行最慢是正常的，因为 `torch.utils.cpp_extension.load()` 会编译 `.cu` 文件。
- 真正做对比时，先普通跑一次把 extension cache 热起来，再开 `ncu`。
- `--set full` 很重，学习阶段先用 `--set basic` 更合适。
- `profile_iters` 默认是 1。如果单次 kernel 太短、采样不稳定，可以加大：

```bash
PROFILE_ITERS=10 bash Nsight-learn/run_ncu_reduce.sh float4 16777216 --set basic
```
