# vLLM Request with Nsight Systems

`ncu` 适合回答“这个 kernel 为什么慢”。  
`nsys` 更适合回答“这一次 request 的时间花在哪”：CPU 调度、tokenizer、CUDA API、GPU kernel、memcpy、CUDA Graph、NCCL/通信、kernel 之间的空隙。

这个目录里先用一条可控的离线 vLLM request 学习 Nsys：

1. `profile_vllm_nsys.py`
   负责加载本地模型、warmup、用 NVTX 标记 `vllm:init` / `vllm:warmup_*` / `vllm:request_*`，并用 `cudaProfilerStart()/cudaProfilerStop()` 只圈住正式 request。
2. `run_nsys_vllm.sh`
   负责调用 `nsys profile`，生成 `.nsys-rep`，再导出一份文本摘要。

## 最简单的用法

先确认 vLLM 环境可用：

```bash
/home/alizen/miniconda3/envs/vllm/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```

直接抓一次 request：

```bash
bash Nsight-learn/run_nsys_vllm.sh /home/alizen/models/Qwen3.5-0.8B 64
```

脚本会自动把 vLLM 环境的 `bin` 放进 `PATH`。这点很重要，因为 vLLM/FlashInfer/Triton 的 JIT 编译可能需要调用 `ninja`、`nvcc` 这类子进程。

默认输出：

```bash
Nsight-learn/nsys-reports/vllm_qwen_request.nsys-rep
Nsight-learn/nsys-reports/vllm_qwen_request.txt
```

打开图形界面：

```bash
nsys-ui Nsight-learn/nsys-reports/vllm_qwen_request.nsys-rep
```

命令行看摘要：

```bash
less Nsight-learn/nsys-reports/vllm_qwen_request.txt
```

## 常用参数

换 prompt：

```bash
PROMPT="写一个简短的 CUDA shared memory 优化建议。" bash Nsight-learn/run_nsys_vllm.sh
```

模拟小 batch：

```bash
BATCH_SIZE=4 bash Nsight-learn/run_nsys_vllm.sh /home/alizen/models/Qwen3.5-0.8B 64
```

增加被 profile 的 request 次数：

```bash
PROFILE_ITERS=3 bash Nsight-learn/run_nsys_vllm.sh /home/alizen/models/Qwen3.5-0.8B 64
```

让 CUDA Graph 展开成 node，方便看每个 kernel，但开销更高：

```bash
bash Nsight-learn/run_nsys_vllm.sh /home/alizen/models/Qwen3.5-0.8B 64 --cuda-graph-trace=node
```

关闭 vLLM 的 CUDA Graph，更容易看清逐 kernel 时间线，但这不是生产默认性能：

```bash
ENFORCE_EAGER=1 bash Nsight-learn/run_nsys_vllm.sh /home/alizen/models/Qwen3.5-0.8B 64
```

## 一个实际调优流程

1. 先定义目标：单 request latency、TTFT、TPOT、吞吐、显存占用，不能只看一个总时间。
2. 用 vLLM 自己的日志或简单计时拿 baseline，固定模型、prompt 长度、max tokens、batch/concurrency。
3. 用 Nsys 抓一条 warmup 后的 request。第一版不要抓模型加载，否则 timeline 太吵。
4. 在 `nsys-ui` 里先看 NVTX range：`vllm:profile` 下面每个 `vllm:request_*` 的时长。
5. 展开 CUDA timeline：看 request 里 GPU 是否连续忙碌，还是 kernel 之间有明显空洞。
6. 看 CUDA API lane：如果 CPU 线程长时间卡在同步、内存分配、图构建、调度，问题不一定在 kernel 本身。
7. 看 GPU kernel summary：找最耗时的 attention、matmul、sampling、copy kernel。只有当某个 kernel 明确可疑，再切到 NCU 深挖。
8. 改一个变量再重跑：例如 `max_model_len`、batch size、并发、`gpu_memory_utilization`、CUDA Graph/eager、attention backend、量化、tensor parallel。

## 看 timeline 时的几个信号

- GPU 上大片空白：CPU 调度、tokenizer、同步、调度线程或 request batching 可能是瓶颈。
- CUDA API 很密但 GPU kernel 很短：kernel launch overhead 或小 batch 可能明显。
- CUDA Graph launch 很少但内部看不清：用 `--cuda-graph-trace=node` 或 `ENFORCE_EAGER=1` 做诊断。
- memcpy 很多：检查输入/输出搬运、paged KV cache 行为、是否有不必要的 CPU/GPU 同步。
- kernel summary 中某几个 kernel 占大头：记录名字和形状，再用 NCU 单独看这些 kernel。

真实线上服务也可以用 Nsys 抓 `vllm serve`，但第一步建议先用这个离线 request 脚本把时间线读懂。等知道要看什么之后，再把同样的指标迁移到 OpenAI server/client 形态。
