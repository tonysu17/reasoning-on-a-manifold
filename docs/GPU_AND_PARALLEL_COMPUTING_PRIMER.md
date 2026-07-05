# A Primer on Parallel Programming and GPUs for ML Training & Inference

> An exhaustive, self-contained introduction to the systems fundamentals behind running
> machine-learning workloads on GPUs — written for an ML researcher who is fluent in
> Python/PyTorch and strong in ML theory, but is not (yet) a systems or HPC specialist.
> It covers both **training** and **inference**, with inference (the likely day-to-day use
> case) treated in special depth.

*Generated 2026-06-28. Hardware figures (TFLOP/s, bandwidth, VRAM) are representative and
vendor/version-dependent — always confirm against the current datasheet for your exact card.*

---

## How to read this document

You can read it front-to-back, but each section is also self-contained enough to use as
reference. A suggested path by goal:

- **"I just want my inference run to work and be fast."** Read §2 (hardware), §6 (memory —
  *will it fit?*), §9 (inference deep dive), and §10 (the practical checklist). Skim §4 and §7.
- **"I want to understand *why*, from the ground up."** Read in order; §1→§4 build the mental
  model that everything else specializes.
- **"I'm about to fine-tune / train."** Add §5 (precision), §7 (batch size), and §8
  (distributed training) to the inference path above.

Recurring threads to watch for, because they explain almost every performance phenomenon you
will meet: **(a)** moving bytes is far more expensive than doing arithmetic (the *memory wall*);
**(b)** GPUs hide latency with massive parallelism, so they need *lots of work in flight*;
**(c)** most LLM *decoding* is **memory-bandwidth-bound**, which is the single fact that explains
why batching, the KV cache, and quantization matter so much.

---

## Table of contents

- [1. Foundations of parallel and concurrent computing](#1-foundations-of-parallel-and-concurrent-computing)
  - [1.1 Concurrency vs parallelism](#11-concurrency-vs-parallelism)
  - [1.2 Latency, throughput, and bandwidth as distinct goals](#12-latency-throughput-and-bandwidth-as-distinct-goals)
  - [1.3 Flynn's taxonomy and SIMT](#13-flynns-taxonomy-and-simt)
  - [1.4 Forms of parallelism](#14-forms-of-parallelism)
  - [1.5 Amdahl's Law and Gustafson's Law](#15-amdahls-law-and-gustafsons-law)
  - [1.6 The memory wall: why moving data dominates](#16-the-memory-wall-why-moving-data-dominates)
  - [1.7 Hazards of parallelism](#17-hazards-of-parallelism)
  - [1.8 Roofline thinking: compute-bound vs memory-bound](#18-roofline-thinking-compute-bound-vs-memory-bound)
  - [1.9 The host/device split and why offload exists](#19-the-hostdevice-split-and-why-offload-exists)
- [2. GPU hardware architecture](#2-gpu-hardware-architecture)
  - [2.1 Why GPUs differ from CPUs: latency-hiding vs latency-avoiding](#21-why-gpus-differ-from-cpus-latency-hiding-vs-latency-avoiding)
  - [2.2 The compute hierarchy: GPU → SM → warp → ALU](#22-the-compute-hierarchy-gpu-sm-warp-alu)
  - [2.3 Tensor Cores: where the headline TFLOP/s come from](#23-tensor-cores-where-the-headline-tflops-come-from)
  - [2.4 The memory hierarchy in full detail](#24-the-memory-hierarchy-in-full-detail)
  - [2.5 Memory coalescing and bank conflicts](#25-memory-coalescing-and-bank-conflicts)
  - [2.6 Occupancy: necessary but not sufficient](#26-occupancy-necessary-but-not-sufficient)
  - [2.7 HBM vs GDDR, and a datacenter GPU comparison](#27-hbm-vs-gddr-and-a-datacenter-gpu-comparison)
  - [2.8 Inter-GPU interconnect: PCIe vs NVLink/NVSwitch](#28-inter-gpu-interconnect-pcie-vs-nvlinknvswitch)
- [3. The CUDA programming model and the ML software stack](#3-the-cuda-programming-model-and-the-ml-software-stack)
  - [3.1 The CUDA programming model: kernels and the thread hierarchy](#31-the-cuda-programming-model-kernels-and-the-thread-hierarchy)
  - [3.2 The execution model: asynchrony, streams, events](#32-the-execution-model-asynchrony-streams-events)
  - [3.3 The library stack you actually use](#33-the-library-stack-you-actually-use)
  - [3.4 How PyTorch turns Python into kernels](#34-how-pytorch-turns-python-into-kernels)
  - [3.5 Kernel fusion and the tools that do it](#35-kernel-fusion-and-the-tools-that-do-it)
  - [3.6 Memory management: the allocator, OOM, and host transfers](#36-memory-management-the-allocator-oom-and-host-transfers)
  - [3.7 Eager vs graph compilers: JAX/XLA in one breath](#37-eager-vs-graph-compilers-jaxxla-in-one-breath)
  - [3.8 The compilation chain and version compatibility (the real-world gotcha)](#38-the-compilation-chain-and-version-compatibility-the-real-world-gotcha)
- [4. Performance fundamentals: FLOPs, bandwidth, arithmetic intensity, and the roofline](#4-performance-fundamentals-flops-bandwidth-arithmetic-intensity-and-the-roofline)
  - [4.1 FLOP vs FLOP/s: getting the units right](#41-flop-vs-flops-getting-the-units-right)
  - [4.2 The FLOP cost of a matrix multiply (derived)](#42-the-flop-cost-of-a-matrix-multiply-derived)
  - [4.3 The transformer FLOP rules: ~6N per token (training), ~2N per token (inference)](#43-the-transformer-flop-rules-6n-per-token-training-2n-per-token-inference)
  - [4.4 Memory bandwidth: the other wall](#44-memory-bandwidth-the-other-wall)
  - [4.5 Arithmetic intensity and the roofline model](#45-arithmetic-intensity-and-the-roofline-model)
  - [4.6 Cache reuse, data locality, and tiling: why a naive matmul is slow](#46-cache-reuse-data-locality-and-tiling-why-a-naive-matmul-is-slow)
  - [4.7 Putting it together: why prefill is fast and decode is slow](#47-putting-it-together-why-prefill-is-fast-and-decode-is-slow)
  - [4.8 MFU and HFU: measuring how well you're using the hardware](#48-mfu-and-hfu-measuring-how-well-youre-using-the-hardware)
  - [4.9 Batch size and arithmetic intensity (bridge to the batch-size section)](#49-batch-size-and-arithmetic-intensity-bridge-to-the-batch-size-section)
  - [4.10 How to actually measure FLOPs and intensity](#410-how-to-actually-measure-flops-and-intensity)
- [5. Numerical precision and data types](#5-numerical-precision-and-data-types)
  - [5.1 How a floating-point number is built from bits](#51-how-a-floating-point-number-is-built-from-bits)
  - [5.2 The format zoo: bit layouts and tradeoffs](#52-the-format-zoo-bit-layouts-and-tradeoffs)
  - [5.3 Why precision matters I: memory (bytes per parameter)](#53-why-precision-matters-i-memory-bytes-per-parameter)
  - [5.4 Why precision matters II: speed (tensor cores)](#54-why-precision-matters-ii-speed-tensor-cores)
  - [5.5 Mixed-precision training: the recipe](#55-mixed-precision-training-the-recipe)
  - [5.6 Numerical stability: the failure modes](#56-numerical-stability-the-failure-modes)
  - [5.7 Quantization for inference](#57-quantization-for-inference)
  - [5.8 FP8 training and inference (Hopper / Ada and later)](#58-fp8-training-and-inference-hopper-ada-and-later)
  - [5.9 Practical guidance: what precision should *you* use?](#59-practical-guidance-what-precision-should-you-use)
- [6. GPU memory: what consumes it in training and inference](#6-gpu-memory-what-consumes-it-in-training-and-inference)
  - [6.1 The bytes-per-parameter foundation](#61-the-bytes-per-parameter-foundation)
  - [6.2 Training memory: the six consumers](#62-training-memory-the-six-consumers)
  - [6.3 Activation memory: why it scales, why it dominates, and how to crush it](#63-activation-memory-why-it-scales-why-it-dominates-and-how-to-crush-it)
  - [6.4 Inference memory: weights + KV cache + activations + overhead](#64-inference-memory-weights-kv-cache-activations-overhead)
  - [6.5 "Does it fit?" — rules of thumb and worked verdicts](#65-does-it-fit-rules-of-thumb-and-worked-verdicts)
  - [6.6 The caching allocator, fragmentation, and why `nvidia-smi` "lies"](#66-the-caching-allocator-fragmentation-and-why-nvidia-smi-lies)
  - [6.7 Paged KV cache (paged attention) — the inference allocator analogue](#67-paged-kv-cache-paged-attention-the-inference-allocator-analogue)
  - [6.8 The full toolbox for fitting a model that "doesn't fit"](#68-the-full-toolbox-for-fitting-a-model-that-doesnt-fit)
  - [6.9 Reading memory in practice: `nvidia-smi` and `torch.cuda.memory_summary()`](#69-reading-memory-in-practice-nvidia-smi-and-torchcudamemorysummary)
- [7. Batch size: throughput, latency, convergence, and memory](#7-batch-size-throughput-latency-convergence-and-memory)
  - [7.1 Definitions: micro-batch, mini-batch, global/effective batch](#71-definitions-micro-batch-mini-batch-globaleffective-batch)
  - [7.2 The throughput curve: why batch size and utilization are linked](#72-the-throughput-curve-why-batch-size-and-utilization-are-linked)
  - [7.3 Latency versus throughput: the serving tradeoff](#73-latency-versus-throughput-the-serving-tradeoff)
  - [7.4 Training: convergence and the statistics of batch size](#74-training-convergence-and-the-statistics-of-batch-size)
  - [7.5 Inference batching in depth](#75-inference-batching-in-depth)
  - [7.6 Practical: choosing batch size and finding the maximum that fits](#76-practical-choosing-batch-size-and-finding-the-maximum-that-fits)
- [8. Distributed training: multi-GPU and multi-node parallelism](#8-distributed-training-multi-gpu-and-multi-node-parallelism)
  - [8.1 Why distribute at all: two distinct problems](#81-why-distribute-at-all-two-distinct-problems)
  - [8.2 Collective communication primitives](#82-collective-communication-primitives)
  - [8.3 Interconnects and topology](#83-interconnects-and-topology)
  - [8.4 Data parallelism and PyTorch DDP](#84-data-parallelism-and-pytorch-ddp)
  - [8.5 ZeRO and FSDP: sharding the redundant state](#85-zero-and-fsdp-sharding-the-redundant-state)
  - [8.6 Tensor (intra-layer) parallelism](#86-tensor-intra-layer-parallelism)
  - [8.7 Pipeline (inter-layer) parallelism](#87-pipeline-inter-layer-parallelism)
  - [8.8 Sequence, context, and expert parallelism](#88-sequence-context-and-expert-parallelism)
  - [8.9 Composing it all: 3D / N-D parallelism](#89-composing-it-all-3d-n-d-parallelism)
  - [8.10 Frameworks, configs, and operational concerns](#810-frameworks-configs-and-operational-concerns)
- [9. Inference systems deep dive (the reader's primary use case)](#9-inference-systems-deep-dive-the-readers-primary-use-case)
  - [9.1 The two phases: prefill vs decode](#91-the-two-phases-prefill-vs-decode)
  - [9.2 The KV cache: the central resource in LLM serving](#92-the-kv-cache-the-central-resource-in-llm-serving)
  - [9.3 Why batching is the only escape, and what it costs](#93-why-batching-is-the-only-escape-and-what-it-costs)
  - [9.4 Continuous (in-flight) batching](#94-continuous-in-flight-batching)
  - [9.5 Attention and memory kernels: PagedAttention, FlashAttention, FlashDecoding](#95-attention-and-memory-kernels-pagedattention-flashattention-flashdecoding)
  - [9.6 Other decode-time optimisations](#96-other-decode-time-optimisations)
  - [9.7 Speculative decoding](#97-speculative-decoding)
  - [9.8 Serving metrics: latency, throughput, and goodput](#98-serving-metrics-latency-throughput-and-goodput)
  - [9.9 Single-GPU vs multi-GPU, and disaggregated serving](#99-single-gpu-vs-multi-gpu-and-disaggregated-serving)
  - [9.10 The serving-framework landscape](#910-the-serving-framework-landscape)
  - [9.11 Concrete guidance: running a small reasoning model for research inference](#911-concrete-guidance-running-a-small-reasoning-model-for-research-inference)
- [10. Practical workflow: profiling, debugging, and an optimization checklist](#10-practical-workflow-profiling-debugging-and-an-optimization-checklist)
  - [10.1 The diagnostic toolkit](#101-the-diagnostic-toolkit)
  - [10.2 The classic bottlenecks and their fixes](#102-the-classic-bottlenecks-and-their-fixes)
  - [10.3 Debugging out-of-memory (OOM) systematically](#103-debugging-out-of-memory-oom-systematically)
  - [10.4 Precision choices in practice (inference-focused recap)](#104-precision-choices-in-practice-inference-focused-recap)
  - [10.5 Reproducibility and determinism on GPU](#105-reproducibility-and-determinism-on-gpu)
  - [10.6 Environment hygiene — defeating "it works on my machine"](#106-environment-hygiene-defeating-it-works-on-my-machine)
  - [10.7 The optimization checklists](#107-the-optimization-checklists)
  - [10.8 Mini-runbook: a small-model research-inference run](#108-mini-runbook-a-small-model-research-inference-run)

---

## 1. Foundations of parallel and concurrent computing

Before we touch a single CUDA kernel or load a 7-billion-parameter model onto an H100, we need a vocabulary and a mental model for *why* parallel hardware exists, what it is good at, and where it betrays you. Everything later in this primer — occupancy, coalescing, tensor cores, FSDP, KV-cache memory math — is a special case of the ideas in this section. The goal here is to build the conceptual bedrock so that when a profiler later tells you your attention kernel is "memory-bound" or your multi-GPU training run is "communication-bound," you already know what those words mean and roughly what to do about them.

### 1.1 Concurrency vs parallelism

These two words are used interchangeably in casual speech, but they name genuinely different things, and the distinction matters for reasoning about GPUs.

- **Concurrency** is a *structuring* property: a system is concurrent if it is *dealing with* multiple tasks whose lifetimes overlap. Concurrency is about composition — breaking a program into independently-progressing logical activities. It does not require more than one physical execution unit. A single CPU core running an operating system is concurrent: it juggles your browser, your editor, and a background download by rapidly *time-slicing* — switching between them so fast that they all appear to advance together. At any single instant, only one is actually executing.

- **Parallelism** is an *execution* property: a system is parallel if it is *physically doing* multiple things at the same instant, which requires multiple execution units (cores, lanes, GPUs). Parallelism is about *speed through simultaneity*.

Rob Pike's slogan captures it: **"Concurrency is about dealing with lots of things at once. Parallelism is about doing lots of things at once."** You can have concurrency without parallelism (one core, many time-sliced tasks), and parallelism without meaningful concurrency in the structuring sense (a single tight loop whose iterations all run simultaneously on a SIMD unit).

A concrete ML example of the difference: when you call `model.generate()` for a single prompt, the *autoregressive decode loop* is inherently sequential — token $t+1$ depends on token $t$ — so there is little task-level concurrency to exploit across the time dimension. But the matrix multiply that produces *each* token is massively parallel: hundreds of thousands of multiply-accumulate operations happen simultaneously across the GPU's lanes. So a single forward pass is "not very concurrent, but very parallel." Conversely, a web server handling 500 inference requests is highly *concurrent* (500 overlapping request lifetimes) and, on a GPU, you turn that concurrency into parallelism by *batching* the requests so their math executes simultaneously. Holding these two notions apart is what lets you say precisely why batching helps: it converts request-level concurrency into arithmetic-level parallelism the hardware can actually exploit.

### 1.2 Latency, throughput, and bandwidth as distinct goals

Performance is not a single number. Three quantities pull in different directions, and hardware (and your own tuning choices) must pick a balance.

- **Latency** is the time to complete *one* operation, from start to finish. Units: seconds (for GPUs, often microseconds or milliseconds). Example: the time from submitting one prompt to receiving its first token (in serving, the "time to first token," TTFT).

- **Throughput** is the *rate* of completed operations, i.e. work per unit time. Units: operations/second, tokens/second, samples/second. Example: total tokens/second your server emits across all concurrent users.

- **Bandwidth** is a *specific kind of throughput for data movement*: bytes transferred per second across some link (memory bus, NVLink, PCIe, network). Units: GB/s or TB/s. It is the supply rate of the raw material (bytes) that compute consumes.

The crucial insight is that **latency and throughput are often in tension.** Consider inference batching. If a single request takes 20 ms to process alone (latency = 20 ms, throughput = 50 req/s), batching 32 requests together might take 40 ms for the whole batch. Now each individual user waits 40 ms (latency *doubled*, worse) but the server completes 32 requests per 40 ms = **800 req/s** (throughput up 16×, much better). You traded per-request latency for aggregate throughput. This is the central trade-off of GPU serving, and it is why GPUs are described as **throughput-oriented** machines (maximize total work) whereas CPUs are **latency-oriented** (minimize time for any one task, via big caches and branch predictors).

An analogy that does not hand-wave the mechanism: a sports car (CPU) carries 2 people at 200 km/h; a bus (GPU) carries 60 people at 80 km/h. For one passenger, the car has lower *latency*. For moving a stadium crowd, the bus has vastly higher *throughput* — even though it is "slower." Bandwidth is the width of the road feeding the vehicles: if the road is too narrow, neither vehicle reaches its rated speed, no matter how fast its engine. We will see in §1.6 that for modern ML, the road (memory bandwidth) is very often the binding constraint, not the engine (FLOPs).

A useful quantitative relationship is **Little's Law**, from queueing theory:

$$\text{concurrency} = \text{throughput} \times \text{latency}.$$

To keep a high-throughput, high-latency device (a GPU) busy, you need many operations *in flight* simultaneously. If each operation has latency $L = 0.5$ µs and you want throughput $T = 2{,}000$ operations/µs, you need $N = T \times L = 1{,}000$ operations in flight at once. This is *exactly* why GPUs run tens of thousands of threads: not because there is that much truly independent "work" in a wall-clock sense, but because they hide the long latency of memory accesses by always having other threads ready to run. Keep Little's Law in mind — it reappears as the deep justification for GPU "occupancy."

### 1.3 Flynn's taxonomy and SIMT

In 1966 Michael Flynn classified computer architectures by how many independent **instruction streams** and **data streams** they process at once. Two binary axes give four categories.

| Category | Instruction streams | Data streams | Meaning | Example |
|---|---|---|---|---|
| **SISD** | Single | Single | One instruction operates on one datum at a time | Classic scalar CPU core (no vector unit) |
| **SIMD** | Single | Multiple | One instruction operates on *many* data elements simultaneously | CPU vector units (AVX-512, ARM NEON); GPU within a lane group |
| **MISD** | Multiple | Single | Many instructions on the same datum (rare) | Fault-tolerant systems running redundant computations; mostly a theoretical slot |
| **MIMD** | Multiple | Multiple | Independent processors run different instructions on different data | Multi-core CPUs; a cluster of GPUs; distributed training |

Let us unpack the two that matter for GPUs.

**SIMD (Single Instruction, Multiple Data).** A single instruction, e.g. "add," is broadcast to a fixed set of parallel *lanes*, each holding a different data element, and all lanes execute that one instruction in lockstep on the same clock cycle. If you have an AVX-512 unit, one `vaddps` instruction adds sixteen 32-bit floats at once. SIMD is enormously efficient *per transistor* because the expensive parts of a processor — instruction fetch, decode, scheduling — are *amortized* across many lanes: you pay to decode "add" once, but you get 16 adds. The catch is **rigidity**: all lanes must do the *same* operation. If your code says "if x > 0 do A else do B," a pure SIMD unit cannot have some lanes do A while others do B in the same cycle; it must serialize — run A with the B-lanes masked off (idle), then run B with the A-lanes masked off. This is **divergence**, and it wastes lanes.

**MIMD (Multiple Instruction, Multiple Data).** Each execution unit is fully independent, fetching and running its own instruction stream on its own data. A 64-core CPU is MIMD; a rack of 8 GPUs each running its own kernel is MIMD; multi-node distributed training is MIMD at the top level. MIMD is flexible (units can do entirely different work) but expensive (every unit pays for its own fetch/decode/control logic), and coordinating MIMD units requires explicit communication and synchronization (§1.7).

**SIMT (Single Instruction, Multiple Threads) — NVIDIA's variant.** This is the execution model of NVIDIA GPUs and the most important entry for us, even though it is not in Flynn's original list. SIMT is best understood as **SIMD hardware presented to the programmer as many independent scalar threads.**

Here is the mechanism, stated carefully because the distinction is subtle and load-bearing:

- The hardware executes threads in fixed-size groups called **warps** (32 threads on all current NVIDIA GPUs; AMD's analogous "wavefront" is 64 threads on most GCN/CDNA generations and 32 or 64 on RDNA). The 32 threads of a warp share a single instruction-fetch/decode unit and advance *together*, one instruction at a time — that is the "SIMD" part underneath.
- But unlike classic SIMD, where the programmer must explicitly write vector instructions over 16 lanes, in SIMT **you write ordinary-looking scalar code for a single thread** (`c[i] = a[i] + b[i]`), and the hardware runs 32 such threads in lockstep. Each thread has its *own* registers, and on Volta and later, its *own* program counter (Volta introduced independent thread scheduling, giving each thread a per-thread PC), so the programming model is "many threads," not "one vector."
- **The key difference from pure SIMD is how branches are handled.** When the 32 threads of a warp hit a data-dependent branch and disagree (some take the `if`, some the `else`), SIMT does not crash or forbid it — the hardware *automatically* executes the taken path with the non-participating threads **predicated off** (masked, idle), then executes the other path with the complementary mask. The threads **re-converge** afterward. This automatic, per-thread masking is called **warp divergence**. It is handled transparently by hardware, which is why SIMT *feels* like independent threads — but the performance cost is real: a fully divergent 32-way branch can run up to 32× slower because the paths are serialized and lanes sit idle. So SIMT gives you the *programming convenience* of MIMD-style independent threads with the *execution efficiency* (and the divergence penalty) of SIMD.

The practical upshot for ML: the dense linear algebra at the heart of transformers has essentially *no* data-dependent branching inside the hot loops — every element of a matrix multiply does the same multiply-accumulate — so warps stay convergent and SIMT runs at near-peak efficiency. Divergence becomes a concern only in irregular code: custom samplers, mixture-of-experts routing, sparse or ragged-batch attention, beam search bookkeeping. Knowing *why* those are slow (lane masking and serialized paths) is what this taxonomy buys you.

### 1.4 Forms of parallelism

"Parallelism" is not one thing; there are several structurally distinct ways to split work across hardware. In ML you will use *all four at once* in a large training run, so it pays to keep them straight. Each is defined below with its ML instantiation.

#### Data parallelism

**Definition.** Apply the *same* operation/program to *different* pieces of the data, simultaneously. The work is partitioned along the *data* axis; the *program is replicated*.

**ML example.** This is the workhorse of both inference and training. Within a single GPU, a batch of 32 sequences flows through *one copy* of the model, but the 32 examples are processed in parallel across the GPU's lanes — same weights, different inputs. Across multiple GPUs, **distributed data parallelism (DDP)** places a full replica of the model on each of, say, 8 GPUs; each replica processes a different 1/8 of the global batch (a "shard"), computes gradients locally, and then the GPUs **average their gradients** via a collective operation (an *all-reduce*, §1.7) so every replica applies the same weight update and stays in sync. Data parallelism scales throughput almost linearly *as long as the model fits on one device* and the gradient all-reduce does not become the bottleneck. It is the easiest form to reason about and the first you should reach for.

#### Task parallelism

**Definition.** Run *different* operations/programs on (possibly the same or different) data, simultaneously. The work is partitioned along the *function* axis; *different code* runs concurrently.

**ML example.** A serving pipeline where one set of workers tokenizes incoming text, another runs the GPU forward pass, and a third de-tokenizes and streams output — three different tasks running concurrently, each specialized. Or, on a single training step, overlapping the *backward pass compute* of one layer with the *gradient communication* of the layer whose gradients are already ready: two different kinds of work (compute vs. network transfer) proceeding at the same time on different hardware units (SMs vs. NVLink/network engines). Task parallelism is what lets you hide communication behind computation.

#### Pipeline parallelism

**Definition.** A special, important hybrid: split a *sequential* computation into ordered *stages*, place each stage on a different worker, and stream a series of inputs through so that while stage 2 works on item $i$, stage 1 is already working on item $i+1$ — like an assembly line. It is "task parallelism arranged in a dependency chain with throughput recovered by streaming."

**ML example.** A 40-layer model too large for one GPU is split so that GPU 0 holds layers 1–10, GPU 1 holds 11–20, and so on. A "micro-batch" enters GPU 0; when it finishes, its activations are passed to GPU 1 while GPU 0 picks up the next micro-batch. The defining headache is the **pipeline bubble**: at the very start, GPUs 1–3 sit idle waiting for the first micro-batch to reach them (fill), and at the end GPUs 0–2 idle while the last micro-batch drains. For a synchronous schedule (e.g. GPipe) with a pipeline of depth $P$ processing $m$ micro-batches, the idealized bubble fraction is $\frac{P-1}{m + P - 1}$: with $P=4$ stages and $m=4$ micro-batches the pipeline is idle $\frac{3}{7}\approx 43\%$ of the time, but with $m=32$ micro-batches the bubble shrinks to $\frac{3}{35}\approx 9\%$. The lesson — feed pipelines *many* micro-batches to amortize the fill/drain — is a direct quantitative payoff of understanding the structure. (More advanced schedules such as 1F1B and interleaved/zero-bubble pipelines reduce this overhead further, but the same $m \gg P$ intuition holds.)

#### Model parallelism (tensor / operator parallelism)

**Definition.** Split a *single* operation that is too big for one device across multiple devices, partitioning the *parameters and the math itself*. Distinct from pipeline parallelism: pipeline splits the model *between* layers (depth-wise); model/tensor parallelism splits *within* a layer (width-wise).

**ML example.** A single attention or feed-forward weight matrix is sharded column-wise across 4 GPUs; each GPU multiplies the input by its slice of the matrix to produce a slice of the output, and then a collective (an *all-gather* or *all-reduce*) stitches the partial results into the full output before the next layer. This is **tensor parallelism**. It is essential when even one layer's weights or activations exceed a single GPU's memory, but it is *communication-heavy* — there is a collective inside *every* layer, on the critical path — so it is normally confined to GPUs connected by the fastest links (NVLink within a single node) and not stretched across slow inter-node networks.

**How they compose.** A frontier training run uses **3D (or 4D) parallelism**: tensor parallelism within a node (fast NVLink), pipeline parallelism across a few nodes, and data parallelism across many such groups, sometimes with a fourth axis (sharded-data-parallel / FSDP, or sequence/context parallelism) layered on. The art is matching each parallelism form to the interconnect that suits its communication pattern. We will return to the hardware specifics later; for now, hold the four definitions and their ML mappings firmly, because tuning a distributed run is largely the act of choosing the right blend.

| Form | What is split | What is replicated | Communication | Primary ML use |
|---|---|---|---|---|
| Data | The data/batch | The whole model | Gradient all-reduce (once per step) | Scale throughput when model fits |
| Task | The functions/code | — | Hand-off between stages | Overlap compute with comms/IO |
| Pipeline | The model, *between* layers | — | Activations between stages | Fit deep models; bubble is the cost |
| Model/Tensor | The model, *within* a layer | The data | Collective *inside every layer* | Fit huge layers; needs fast links |

### 1.5 Amdahl's Law and Gustafson's Law

Two laws bound how much speedup parallelism can ever deliver. They are not contradictory; they answer different questions. Both are essential for calibrating expectations and for spending money wisely on more GPUs.

#### Amdahl's Law — the serial-fraction ceiling

Suppose a program has a fraction $p$ of its work that *can* be parallelized and a fraction $s = 1 - p$ that is irreducibly **serial** (must run sequentially no matter how many processors you have — think of loading the model, the autoregressive dependency in decoding, or a global synchronization barrier). With $N$ processors, the parallel part speeds up by $N$ but the serial part does not, so the total runtime relative to single-processor time is $s + p/N$, and the **speedup** is

$$S(N) = \frac{1}{s + \dfrac{p}{N}} = \frac{1}{(1-p) + \dfrac{p}{N}}.$$

The decisive feature is the limit as $N \to \infty$: the parallel term vanishes but the serial term remains, so

$$S_{\max} = \lim_{N\to\infty} S(N) = \frac{1}{s} = \frac{1}{1-p}.$$

**The serial fraction sets a hard ceiling on speedup, independent of how many processors you throw at the problem.**

*Worked example.* Suppose 95% of a workload is parallelizable ($p = 0.95$, $s = 0.05$).

- With $N = 8$ GPUs: $S(8) = \dfrac{1}{0.05 + 0.95/8} = \dfrac{1}{0.05 + 0.11875} = \dfrac{1}{0.16875} \approx 5.93\times$. So 8 GPUs deliver under 6× — already a 26% efficiency loss to the serial 5%.
- With $N = 32$: $S(32) = \dfrac{1}{0.05 + 0.95/32} = \dfrac{1}{0.05 + 0.0297} = \dfrac{1}{0.0797} \approx 12.5\times$. Quadrupling GPUs (8→32) barely doubled the speedup.
- With $N \to \infty$: $S_{\max} = 1/0.05 = 20\times$. You can *never* beat 20×, even with infinite hardware, because 5% of the work is stubbornly serial.

The brutal corollary: if even **1%** is serial, your maximum speedup is $1/0.01 = 100\times$, so a 1000-GPU cluster running that workload is at best 10% efficient. This is why shrinking the serial fraction — overlapping communication, removing global barriers, avoiding host–device round-trips — often matters more than adding hardware.

#### Strong vs weak scaling, and Gustafson's Law

Amdahl's Law assumes a **fixed problem size** and asks "how much faster?" — this is **strong scaling** (same total work, more processors, measure the speedup). Strong scaling is pessimistic for large $N$ precisely because of the serial ceiling.

But in practice we rarely keep the problem fixed when we get more hardware: we *grow* the problem. More GPUs means we train a *bigger* model or use a *larger* batch. This is **weak scaling** — keep the work *per processor* fixed and grow the total problem with $N$ — and it is the regime ML actually lives in. Gustafson's Law reframes the question accordingly. If, on the larger problem, a fraction $s$ of the *scaled* runtime is serial and the rest is parallel work done by $N$ processors, the **scaled speedup** is

$$S(N) = s + p\,N = N - s\,(N - 1) = N + (1 - N)\,s.$$

This is *linear* in $N$, not bounded by $1/s$. 

*Worked example.* With the same serial fraction $s = 0.05$ ($p = 0.95$) and $N = 32$:

$$S(32) = 0.05 + 0.95 \times 32 = 0.05 + 30.4 = 30.45\times.$$

Contrast this with Amdahl's $12.5\times$ for the same $s$ and $N$. The difference is *not* a contradiction — the two laws answer different questions:

- **Amdahl (strong scaling):** *fixed* work. "I have this exact job; how much faster with more GPUs?" The serial part is a fixed amount of time, so its *relative* weight grows as the parallel part shrinks, throttling speedup.
- **Gustafson (weak scaling):** *scaled* work. "I have more GPUs; how much more work can I do in the same wall-clock time?" As the problem grows, the serial part stays roughly constant while the parallel part grows with $N$, so the serial *fraction* shrinks and near-linear scaling is recoverable.

This is precisely why training ever-larger models on ever-larger clusters works at all: ML is overwhelmingly a *weak-scaling* enterprise. Nobody buys 1000 GPUs to train a tiny model 1000× faster (Amdahl forbids it); they buy 1000 GPUs to train a 1000× *bigger* model in the same time (Gustafson permits it). When you *do* hit a strong-scaling wall — e.g. trying to cut the latency of one fixed inference request by adding GPUs — remember Amdahl is the law in force, and the autoregressive serial dependency is your $s$.

### 1.6 The memory wall: why moving data dominates

Here is the single most counter-intuitive fact for someone coming from ML theory, where we count FLOPs (floating-point operations) as *the* cost of an algorithm: **on modern hardware, arithmetic is nearly free, and moving the data to feed that arithmetic is the expensive part.** This is the **memory wall**, and internalizing it explains the majority of GPU performance behavior.

#### The growing gap

For decades, the peak compute rate of processors grew far faster than the rate at which memory could supply operands. Compute throughput (FLOP/s) has historically improved much faster per year than DRAM **bandwidth** (bytes/s) and *dramatically* faster than DRAM **latency** (time to first byte), which has barely improved at all. The result is a widening gap: a modern GPU can perform on the order of **hundreds of arithmetic operations in the time it takes to fetch a single operand from main memory.** The expensive resource is no longer the multiplier; it is the wire.

Some representative orders of magnitude (exact figures are vendor- and generation-dependent; treat these as illustrative, not spec-sheet exact):

| Operation | Approximate cost | Relative |
|---|---|---|
| One floating-point multiply-add (operands already in registers) | ~1 unit of time/energy | 1× |
| Read operand from on-chip SRAM (L1/shared memory) | ~5–30× | tens × |
| Read operand from on-chip L2 cache | ~200–300 cycles latency | ~100× |
| Read operand from off-chip DRAM (HBM/GDDR) | ~hundreds of cycles latency, ~100–1000× the energy of the FLOP | hundreds × |
| Move operand over NVLink to another GPU | higher still | thousands × |
| Move operand over PCIe / network to another node | higher again | $10^4$–$10^6$ × |

The hierarchy is the headline: **every step away from the arithmetic unit — registers → on-chip SRAM → L2 → DRAM → other GPU → other node — costs roughly an order of magnitude more in both time and energy.** A multiply-add that costs ~1 might be dwarfed by the ~hundreds it costs just to fetch its inputs from DRAM. So "how many FLOPs does my model do" is frequently the *wrong* question; the right one is "how many *bytes* must I move, and from how far away?"

#### Arithmetic intensity (preview)

This motivates the central diagnostic quantity we will deepen in the roofline section. The **arithmetic intensity** (also "operational intensity") of a computation is

$$I = \frac{\text{FLOPs performed}}{\text{bytes moved (to/from memory)}}\quad[\text{FLOP/byte}].$$

It measures how much useful arithmetic you extract from each byte you pay to move. A computation with *high* arithmetic intensity does lots of math per byte and can keep the arithmetic units busy; a computation with *low* arithmetic intensity starves them, because the units sit idle waiting for the memory system. Two canonical ML cases make this vivid:

- **A large matrix–matrix multiply** ($A_{m\times k} B_{k\times n}$) performs $\sim 2mnk$ FLOPs but moves only $\sim (mk + kn + mn)$ elements. For large square matrices the FLOPs grow as $\Theta(n^3)$ while the bytes grow as $\Theta(n^2)$, so intensity grows like $\Theta(n)$ — *high*, and growing with size. Each element loaded from DRAM is *reused* many times (every entry of $A$ participates in $n$ multiply-adds). This is why GPUs love big matmuls and why training, dominated by large batched matmuls, is typically **compute-bound** and can approach peak FLOP/s.

- **An element-wise operation** like adding a bias, applying a GELU, or a LayerNorm reads each element, does a small constant number of FLOPs on it, and writes it back. Intensity is $O(1)$ — perhaps well under 1 FLOP/byte, *no reuse*. These kernels are hopelessly **memory-bound**: the arithmetic units finish instantly and then wait on DRAM. This is exactly why *kernel fusion* (e.g. fusing bias+GELU+dropout into one pass, or FlashAttention fusing the whole attention computation) is such a big win — it raises arithmetic intensity by doing more math per byte loaded, reading the data once and doing all the work before writing back, instead of making a separate memory round-trip per operation.

The single most important consequence for LLM *inference* specifically: **autoregressive decoding of one token at a time is profoundly memory-bound.** To generate each new token you must read the *entire* model's weights from DRAM (tens of GB) but you only do a tiny amount of arithmetic on them (a matrix–*vector* product, not matrix–matrix, because the batch is effectively one token). The arithmetic intensity is dismal, so decode speed is set almost entirely by **how fast you can stream the weights out of memory**, i.e. by memory bandwidth — *not* by the GPU's peak FLOP/s. This is the deepest reason batching helps inference so much (it amortizes one weight-read across many requests, raising intensity), why memory bandwidth is the spec to watch for a serving GPU, and why quantization (fewer bytes per weight) directly speeds up decoding. We will make all of this quantitative in the roofline and inference sections; for now, carry the slogan: **FLOPs are cheap, bytes are expensive, and reuse is everything.**

### 1.7 Hazards of parallelism

Parallelism is not free correctness-wise. The moment multiple execution units touch shared state, a family of bugs and costs appears that simply does not exist in sequential code. You will meet these both as a *user* of frameworks (which mostly hide them) and as an *author* of custom kernels or distributed code (where they bite). Define each carefully.

#### Race conditions and data races

A **race condition** is any situation where the *outcome* of a program depends on the *relative timing* (the interleaving) of concurrent operations — the answer changes depending on who "wins the race." A **data race** is the most common and dangerous special case: *two or more threads access the same memory location concurrently, at least one access is a write, and there is no synchronization ordering them.* The result is undefined — you may read a half-written value, lose an update, or get different answers on different runs.

The canonical example is the **lost update**. Suppose 1000 threads each want to do `counter += 1`, which is really three steps: read `counter`, add 1, write back. If two threads read the same old value 5 before either writes, both compute 6 and both write 6 — *two* increments produced *one* net change. The final count is silently wrong, and *how* wrong depends on the exact interleaving, so the bug may appear only intermittently. In GPU code this arises constantly: thousands of threads contributing to a shared sum (a reduction), multiple threads scattering into the same output bin (a histogram), or accumulating gradients into a shared parameter.

#### Atomics

The hardware fix for the lost-update race is an **atomic operation**: a read-modify-write that the hardware guarantees executes **indivisibly** — no other thread can observe or interleave with its intermediate state. `atomicAdd(&counter, 1)` performs the read-add-write as one uninterruptible unit, so increments cannot be lost. Atomics are the primitive underneath safe parallel reductions, histograms, and gradient scatter. Their cost: when many threads hit the *same* address, the hardware must **serialize** those atomics (only one can proceed at a time on that location) — this is **contention**, and a hot atomic address can collapse a massively-parallel kernel into an effectively sequential one. The mitigation is to reduce contention: have each thread accumulate privately, then combine a few partial results (hierarchical reduction) so atomics touch any given address rarely.

#### Synchronization, barriers, and locks

To impose order on otherwise-unordered concurrent operations you use **synchronization** primitives:

- A **barrier** is a meeting point: every participating thread must *arrive* at the barrier before *any* thread is allowed to proceed past it. On a GPU, `__syncthreads()` is a barrier across the threads of one block — used, for example, after threads cooperatively load a tile of data into fast shared memory, to guarantee *all* the data is present before *anyone* starts computing on it. In distributed training, a barrier across GPUs ensures all replicas finish a step before the next begins. Barriers are essential but costly: everyone waits for the *slowest* arriver, so a single straggler (an overloaded GPU, a slow network link) stalls the entire group — the "straggler problem."

- A **lock / mutex** (mutual exclusion) protects a **critical section** — a region of code that only one thread may execute at a time — guaranteeing exclusive access to shared state. Locks are common on CPUs; on GPUs they are usually avoided in favor of atomics and lock-free designs, because tens of thousands of threads queuing on a lock destroys parallelism.

#### Deadlock

A **deadlock** is a standstill in which a set of threads are all *blocked forever*, each waiting for a resource (or a synchronization event) that another blocked thread in the set holds and will never release. The classic recipe is two threads each holding one lock and each waiting for the other's lock — neither can proceed, neither will release. In GPU/distributed ML the most common deadlock is a **mismatched collective**: if GPU 0 calls an `all-reduce` but GPU 1 (due to a code-path divergence, e.g. a different `if` branch, or an exception on one rank) never makes the matching call, GPU 0 waits *forever* for data that will never come, and the whole job hangs — typically until a timeout fires. The defense is discipline: every rank must execute the *same* sequence of collectives in the *same* order; lock-ordering conventions prevent the lock-based version.

#### False sharing

A subtler performance hazard (not a correctness bug) that arises because memory moves in fixed-size **cache lines** (commonly 64 bytes on CPUs; on NVIDIA GPUs the L1 cache line is 128 bytes, fetched from DRAM/L2 in 32-byte sectors), not individual bytes. **False sharing** occurs when two threads modify *different* variables that happen to live on the *same* cache line. Although the variables are logically independent and there is no real data race, the cache-coherence hardware sees the *line* being written by two cores and must ferry that line back and forth between their caches on every write, as if they were sharing — hence "false" sharing. Throughput collapses even though the code is correct. (False sharing is most acute on cache-coherent CPUs; on GPUs the analogous penalty shows up as multiple cores contending for the same line and as uncoalesced/partial-sector traffic.) The fix is *padding*: space the per-thread variables apart so each sits on its own cache line. The general principle this teaches — *the granularity of memory movement is the line/sector, not the byte* — recurs centrally in §coalescing for GPUs.

#### Determinism and reproducibility — why GPU reductions are non-deterministic

A consequence that surprises researchers: the *same* training script, *same* data, *same* seed, run twice on the *same* GPU, can produce *bit-for-bit different* results. The principal culprit is **floating-point non-associativity combined with non-deterministic reduction order.**

Floating-point addition is **not associative**: because each intermediate result is rounded to finite precision, $(a + b) + c$ can differ from $a + (b + c)$ in the last bits. For example, adding a tiny number to a huge number can round the tiny one away entirely, so the order in which you sum a set of values changes the rounded result. Now recall that a GPU computes a sum (a reduction) by having thousands of threads add their contributions *in parallel*, and the order in which those partial additions land — which warp finishes first, which atomic arrives first — depends on run-time scheduling that is *not* fixed from run to run. Different orders, different rounding, different last bits. The same effect appears in the gradient all-reduce across GPUs, where contributions arrive in network-dependent order. Each individual rounding difference is minuscule, but in a long training run those tiny perturbations are amplified by the chaotic dynamics of optimization, so two "identical" runs can diverge into measurably different models.

The practical levers: frameworks expose deterministic modes (e.g. `torch.use_deterministic_algorithms(True)`, fixed cuBLAS workspace settings, deterministic algorithm selection) that *force* a fixed reduction order and avoid non-deterministic kernels — at a real **performance cost**, since the fastest kernels are often the non-deterministic ones. The standing trade-off — **speed vs. exact reproducibility** — is one you should make consciously: turn determinism on when debugging or when a result must be exactly reproducible, accept non-determinism for maximum throughput otherwise, and in *all* cases report mean ± standard deviation over seeds rather than trusting a single run's exact number.

### 1.8 Roofline thinking: compute-bound vs memory-bound

We can now assemble §1.2 (bandwidth as a distinct resource) and §1.6 (arithmetic intensity) into the single most useful mental model for GPU performance: the **roofline model**. It answers the first question you should ask about any kernel: *is this limited by how fast the chip can compute, or by how fast it can move data?* The answer dictates which optimizations will help and which are wasted effort.

The model posits two hard ceilings ("roofs") on the achievable performance (in FLOP/s) of any computation:

1. The **compute roof**: the GPU's peak arithmetic rate, $\pi$ FLOP/s (a flat horizontal ceiling — you cannot exceed the hardware's multiply-accumulate throughput).
2. The **memory roof**: the peak rate at which the memory system can *feed* arithmetic. If memory bandwidth is $\beta$ bytes/s and the computation has arithmetic intensity $I$ FLOP/byte, then bytes/s $\times$ FLOP/byte = FLOP/s, so the bandwidth-limited performance is $\beta \times I$ (a sloped line rising with intensity).

The achievable performance is whichever roof is *lower*:

$$P(I) = \min\big(\pi,\; \beta \cdot I\big).$$

Plotted with arithmetic intensity $I$ on the x-axis and attainable FLOP/s on the y-axis, this is a rising diagonal ($\beta I$) that hits a horizontal cap ($\pi$) and stays flat — the shape of a roofline. The two regimes meet at the **ridge point**, the intensity $I^\star = \pi/\beta$ where the sloped memory roof meets the flat compute roof:

- If your kernel's intensity $I < I^\star$, you are on the sloped part: **memory-bound.** Performance is $\beta I$ — limited by bandwidth. Adding compute (a faster chip, tensor cores) does *nothing*; you must move fewer bytes or move them faster (fusion to raise $I$, better caching/reuse, higher-bandwidth memory, lower-precision data to shrink bytes).
- If $I > I^\star$, you are under the flat part: **compute-bound.** Performance is capped at $\pi$ — limited by arithmetic throughput. More bandwidth does nothing; you need faster math (tensor cores, lower-precision arithmetic, fewer FLOPs/better algorithm).

*Worked example.* Take representative datacenter-GPU numbers in the H100 class: peak $\pi \approx 1{,}000$ TFLOP/s (a deliberately round, conservative stand-in — an H100 SXM is rated near 1,979 dense BF16 TFLOP/s with tensor cores, and real kernels rarely hit peak) and memory bandwidth $\beta \approx 2$ TB/s $= 2{,}000$ GB/s (H100 HBM3 is ~3.35 TB/s; we round down for a clean illustration). The ridge point is

$$I^\star = \frac{\pi}{\beta} = \frac{1{,}000 \times 10^{12}\ \text{FLOP/s}}{2 \times 10^{12}\ \text{bytes/s}} = 500\ \text{FLOP/byte}.$$

So you need to do **500 floating-point operations for every byte** you fetch just to *break even* and keep the tensor cores fed. (With true H100 peak numbers, $\approx 1{,}979 / 3.35 \approx 590$ FLOP/byte — the same order of magnitude and the same lesson.) That is a very high bar, and it reframes everything:

- LLM **decode** (one token at a time) has intensity of order ~1–2 FLOP/byte — *far* below 500 — so it sits deep in the memory-bound regime, achieving a tiny fraction of peak FLOP/s. Its speed is set by $\beta$. (This is the quantitative version of the §1.6 claim.)
- A large training **matmul** with intensity in the hundreds-to-thousands FLOP/byte can sit at or above the ridge, approaching the compute roof and actually using the tensor cores you paid for.

The roofline is the discipline that stops you from "optimizing" the wrong thing: there is no point hand-tuning arithmetic in a memory-bound kernel (you will still be waiting on DRAM), and no point chasing memory traffic in a compute-bound one. We will make the roofline quantitative for specific GPUs, derive the arithmetic intensity of attention and matmul precisely, and use it to predict inference throughput in later sections. For now, the reflex to build is: **before optimizing, locate the kernel on the roofline.**

### 1.9 The host/device split and why offload exists

Finally, the structural fact that frames every GPU program: a GPU is not a standalone computer. It is a **co-processor** — an **accelerator** — attached to a conventional CPU-based host. This split, the **host/device model**, runs through the entire software stack (and the entire rest of this primer), so we establish the vocabulary now.

- The **host** is the CPU and *its* memory (system RAM, typically tens to hundreds of GB of DDR). The host runs the operating system, your Python process, and the main program logic. CPUs are **latency-optimized**: few but powerful cores, large caches, sophisticated branch prediction and out-of-order execution — superb at sequential, branchy, irregular control flow (parsing, scheduling, I/O, orchestrating the program).

- The **device** is the GPU and *its* own separate memory (e.g. HBM, high-bandwidth memory, typically tens to ~140 GB on a current datacenter card — e.g. 80 GB on an H100, 141 GB on an H200). The GPU is **throughput-optimized**: thousands of simple lanes, comparatively tiny caches, and very high memory bandwidth — superb at regular, data-parallel arithmetic (the dense linear algebra of neural networks) but poor at branchy sequential logic.

- They have **physically separate memories**, connected by a comparatively narrow link — **PCIe** (peripheral bus; PCIe Gen5 ×16 delivers on the order of ~64 GB/s per direction on current generations) or, on tightly-coupled systems, faster proprietary links (NVLink between GPUs; NVLink-C2C / unified memory on integrated designs such as Grace-Hopper and the DGX Spark, whose GB10 superchip uses NVLink-C2C for a coherent CPU+GPU memory model). Crucially, that host–device link is *far* slower than either device's own memory bandwidth (a discrete GPU's HBM at ~2–5 TB/s vs. PCIe at ~tens of GB/s — roughly a 30–80× gap). This bandwidth cliff is the defining constraint of the host/device model. (Integrated designs like Grace-Hopper or DGX Spark soften it: there the CPU and GPU share one physical memory pool over a much wider C2C link, eliminating the PCIe copy — though that shared pool, e.g. the DGX Spark's 128 GB LPDDR5X at ~273 GB/s, is itself far slower than a discrete card's HBM.)

**Why this split, and why "offload"?** The two processor types are good at *complementary* things, so the standard pattern is **heterogeneous computing**: the CPU runs the control-heavy orchestration and *offloads* the heavy, regular numerical work — the matmuls — to the GPU, which devours it. A canonical training step is therefore a choreography: (1) the host prepares a batch and copies it across the link into device memory; (2) the host *launches* kernels — issues commands telling the GPU what to compute — which the device executes asynchronously on its data; (3) results (a loss scalar, generated tokens) are copied back across the link to the host. The word **offload** names this delegation of compute from host to device; it also names a memory tactic — when a model's weights or optimizer state exceed device memory, frameworks can *offload* the overflow to host RAM (or even NVMe disk) and stream pieces back as needed, trading the painful PCIe bandwidth cliff for the ability to run a model that otherwise would not fit at all.

Two performance lessons fall out of this structure immediately, and both recur throughout the primer:

1. **Minimize host–device traffic.** Because the link is the slowest resource in the system by far, crossing it is expensive. You want to copy data to the GPU *once* and do *as much work as possible* there before copying results back — not ping-pong tensors across PCIe between every operation. Many real-world "GPU is slow" mysteries are actually a program accidentally synchronizing and copying across the link too often (e.g. calling `.item()` or `.cpu()` inside a hot loop, forcing the host to stall for the device and drag a value back).

2. **Overlap, don't stall.** Because kernel launches are *asynchronous* — the host issues a command and immediately continues without waiting for the GPU to finish — a well-built program keeps the host running *ahead* of the device, queueing up the next batch's data transfer and the next kernels while the GPU is still busy on the current one (this is the task parallelism of §1.4 and the latency-hiding of Little's Law, §1.2, applied at the system level). The host should be a tireless dispatcher feeding the device, never an idle waiter blocked on a result it didn't need yet. When you later see CUDA *streams* and asynchronous copies, they are the machinery for exactly this overlap.

With these foundations — concurrency vs. parallelism, the latency/throughput/bandwidth triad, Flynn's taxonomy and SIMT, the four forms of parallelism, the Amdahl/Gustafson scaling laws, the memory wall and arithmetic intensity, the parallel hazards, roofline thinking, and the host/device split — every GPU-specific mechanism in the sections that follow becomes a concrete instance of a principle you now hold. The recurring refrain to carry forward: **the hardware has enormous arithmetic power, and the entire game is keeping it fed.**

---

## 2. GPU hardware architecture

A graphics processing unit (GPU) is, at heart, a machine built for one job: apply the same arithmetic to enormous quantities of data, all at once, as fast as the memory system can feed it. Everything distinctive about GPU hardware — the thousands of arithmetic units, the small caches, the bizarre-looking memory rules, the 32-wide execution lanes — follows from that single design goal. This section builds the picture from the ground up: first *why* a GPU is shaped the way it is, then the compute hierarchy, the tensor cores that produce the headline performance numbers, the memory hierarchy in full detail, the access patterns that make or break bandwidth, and finally the inter-GPU interconnects you will lean on once one card is not enough.

Throughout, I use NVIDIA terminology (CUDA, SM, warp) because that is what you will run on RunPod, on A100/H100-class cards, and on the DGX Spark. I note the AMD equivalents where they matter. All specific GPU numbers are **representative and version-dependent** — vendors ship multiple SKUs (stock-keeping units, i.e. product variants) of the "same" chip with different clocks, memory sizes, and power limits, and the exact figure depends on the silicon revision, the firmware, and the cooling. Treat the numbers as accurate to within roughly ten to twenty percent, good enough for capacity planning and back-of-envelope performance modelling, not as datasheet guarantees.

### 2.1 Why GPUs differ from CPUs: latency-hiding vs latency-avoiding

The cleanest way to understand a GPU is by contrast with a CPU, because the two are optimised for opposite things.

A **CPU (central processing unit)** is a *latency-optimised* machine. Most CPU workloads — running an operating system, a web server, a branchy single-threaded program — are dominated by long chains of dependent operations where each step needs the result of the previous one, and where the next instruction to execute is hard to predict. The enemy is **latency**: the time to complete one operation. CPUs therefore spend most of their transistor budget *avoiding and hiding latency for a single thread*. They have:

- A handful of powerful cores (a typical server CPU has tens of cores; a laptop has 4–16).
- Deep, sophisticated **control logic**: out-of-order execution (the core reorders instructions to keep busy), branch prediction (it guesses which way an `if` goes before knowing), and speculative execution.
- Large **caches** — a fast on-chip copy of recently-used memory. A modern CPU might devote tens of megabytes to L2/L3 cache so that most memory accesses never touch slow main memory.

A **GPU** is a *throughput-optimised* machine. The workloads it targets — shading millions of pixels, multiplying large matrices — consist of huge numbers of *independent* identical operations. The enemy is not the latency of any one operation but the **total time to finish all of them**. The GPU therefore makes the opposite tradeoff: spend the transistor budget on **arithmetic units (ALUs — arithmetic logic units)**, not on cache and control. A datacenter GPU has *thousands* of ALUs versus a CPU's tens of cores.

#### Latency hiding by oversubscription

This is the single most important idea in GPU architecture, so it is worth stating carefully. A GPU does **not** avoid memory latency — a fetch from its main memory still costs hundreds of clock cycles, *longer* in absolute cycles than on a CPU. Instead it **hides** latency by having far more work in flight than it has hardware to execute.

The analogy is a short-order cook versus a fine-dining chef. The fine-dining chef (CPU) works one elaborate dish through to completion, using every trick to shave seconds off that one dish. The short-order cook (GPU) has fifty orders on the rail at once: start the eggs, and while they cook, flip a burger, toast some bread, plate a salad — the cook is never idle waiting on any single order because there is always *another* order ready to advance. The eggs still take three minutes (latency unchanged), but throughput is enormous because idle time is filled with other work.

Concretely: when a group of GPU threads issues a memory load and must wait ~400 cycles for the data, the hardware scheduler instantly switches to another group of threads that *is* ready to compute, and another, and another. As long as you supply enough independent work — enough "orders on the rail" — the expensive arithmetic units stay busy and the memory latency is completely covered. This is why GPUs want you to launch *far* more threads than they have cores: that oversubscription is the fuel for latency hiding, not a wasteful excess.

The cost of this tradeoff: GPUs are bad at exactly what CPUs are good at. A single GPU thread is *slow* (lower clock, no out-of-order tricks). Branchy, dependent, irregular code with little parallelism runs poorly. The GPU only wins when you can express your problem as thousands of independent, mostly-uniform operations — which, fortunately, is exactly what dense neural-network math is.

| Property | CPU (latency-optimised) | GPU (throughput-optimised) |
|---|---|---|
| Core count | Tens of complex cores | Thousands of simple ALUs (grouped into ~100+ SMs) |
| Per-thread speed | Very fast | Slow |
| Clock speed | High (~3–5 GHz) | Moderate (~1–2 GHz boost) |
| Cache per core | Large (MB-scale) | Tiny (KB-scale per SM) |
| Control logic | Heavy (OoO, branch prediction) | Minimal (in-order, shared across lanes) |
| Latency strategy | Avoid/hide for one thread | Hide via massive thread oversubscription |
| Wins at | Serial, branchy, irregular work | Massively parallel, uniform work |

### 2.2 The compute hierarchy: GPU → SM → warp → ALU

A GPU's compute resources are organised in a strict hierarchy. Understanding it is essential because the programming model (Section 3) maps directly onto it, and because nearly every performance limit is explained by some level of this hierarchy running out of a resource.

#### The Streaming Multiprocessor (SM)

The GPU chip is divided into a number of **Streaming Multiprocessors (SMs)** — the fundamental independent processing blocks. (AMD calls the equivalent unit a **Compute Unit, CU**.) An A100 has 108 SMs; an H100 (SXM5 variant) has 132 (the H100 PCIe variant has 114). Think of each SM as a small, self-contained parallel processor with its own arithmetic units, its own register file, its own scratchpad memory, and its own schedulers. The whole GPU is essentially ~100+ of these working in parallel, sharing only the large L2 cache and the main memory.

Each SM contains:

- **CUDA cores** — the scalar ALUs that perform ordinary floating-point and integer arithmetic. An A100 SM has 64 FP32 CUDA cores, so the whole chip has 64 × 108 = 6,912 FP32 cores. ("CUDA core" is a marketing-flavoured name for one FP32 ALU lane; do not over-read it.)
- **Tensor Cores** — specialised matrix-multiply units (Section 2.3); 4 per SM on A100/H100 (the Ampere/Hopper generations use four third- and fourth-generation Tensor Cores per SM, respectively).
- A large **register file** (256 KB per SM on A100 and H100 = 65,536 32-bit registers) — the fastest storage on the chip.
- **Shared memory / L1 cache** — a programmer-controllable scratchpad carved from a combined on-chip SRAM pool (192 KB/SM on A100, 256 KB/SM on H100, of which up to 228 KB can be configured as shared memory on H100).
- **Warp schedulers** (4 per SM on A100/H100) that decide which warps execute each cycle.
- **Special Function Units (SFUs)** for transcendentals (sin, exp, reciprocal, sqrt) and **Load/Store units** for memory traffic.

#### The warp: the true unit of execution

Here is the most important hardware fact a newcomer from the PyTorch world usually does not know: **the GPU does not execute threads individually. It executes them in lockstep groups of 32 called warps.** (AMD uses 64-wide groups called **wavefronts**, though RDNA can also operate in a 32-wide "wave32" mode.) A warp is the atomic unit of scheduling — the hardware always issues an instruction for a whole warp at once, and all 32 threads (called **lanes**) execute that same instruction in the same cycle, each on its own data. This execution model is called **SIMT (Single Instruction, Multiple Threads)**: one instruction stream, driving 32 data lanes.

This is a refinement of the older **SIMD (Single Instruction, Multiple Data)** idea (as in CPU vector units like AVX). The difference is that SIMT lets each lane have its own registers and its own memory address and *appear* to be an independent thread, even though physically they share one instruction stream. (Since Volta, NVIDIA GPUs add independent per-thread program counters, which permits finer-grained interleaving of divergent paths — but execution is still issued per-warp, so the divergence cost below still applies.) The convenience is that you write code as if each thread were independent; the catch — which the next subsection covers — is that when threads in a warp disagree about control flow, the hardware must serialise them, and you pay for it.

When you launch a kernel with, say, 256 threads per block, the hardware silently chops that block into 256 / 32 = 8 warps. Always make your block size a multiple of 32; otherwise the final warp runs partly empty, wasting lanes. A block of 100 threads occupies 4 warps (128 lanes) of hardware, of which 28 lanes do nothing.

#### Warp schedulers and zero-cost context switching

Each SM has multiple **warp schedulers** (4 on A100/H100). Every cycle, each scheduler looks at the pool of warps currently resident on the SM, picks one that is *ready* (its next instruction's inputs are available — not waiting on memory or a dependency), and issues that instruction. This is the latency-hiding mechanism from Section 2.1 made concrete.

The crucial property is that switching between warps is **free** — it costs zero cycles. A CPU context switch is expensive because it must save and restore registers to memory. A GPU avoids this entirely: the registers of *every* resident warp stay live in the giant register file the whole time the warp is resident. The scheduler does not move any state; it just changes which warp's instruction it issues next cycle. This is precisely why a GPU wants many resident warps per SM — they are the ready pool the scheduler draws from to fill stalls. If only one warp is resident and it stalls on memory for 400 cycles, the SM sits idle for those cycles. If many warps are resident, the scheduler almost always finds one ready to run.

#### Warp divergence and its cost

Because all 32 lanes of a warp issue from one instruction stream, a data-dependent branch is a problem. Consider:

```
if (x[i] > 0)
    y[i] = expensive_A();   // taken by 20 lanes
else
    y[i] = expensive_B();   // taken by the other 12 lanes
```

The 32 lanes do not agree on which path to take. The hardware cannot run both branches simultaneously, so it **serialises** them: it runs the `A` branch with the 20 "if" lanes active and the 12 "else" lanes *masked off* (predicated to do nothing), then runs the `B` branch with the 12 "else" lanes active and the 20 "if" lanes masked off. The two paths execute one after another, and the total time is the sum of both paths, not the max. This is **warp divergence**, and in the worst case (every lane takes a different path through a 32-way branch) it costs roughly a 32× slowdown.

Key nuances:

- Divergence is **only** within a warp. If lanes 0–31 all take the `if` and lanes 32–63 (a *different* warp) all take the `else`, there is **no** divergence — each warp is internally uniform. Divergence cost is about disagreement *inside* the 32-lane group.
- A branch that all 32 lanes resolve the same way is essentially free.
- This is why GPU-friendly code avoids data-dependent branching in inner loops, and why neural-network math — which is overwhelmingly branch-free dense arithmetic — maps so beautifully onto the SIMT model. You rarely fight divergence in standard transformer inference; it matters when you write custom kernels (e.g. mixture-of-experts routing, sparse attention).

### 2.3 Tensor Cores: where the headline TFLOP/s come from

The CUDA cores described above do scalar arithmetic — one multiply-add per lane per cycle. That is not where a modern GPU's advertised performance comes from. The headline numbers — "an H100 does ~990 TFLOP/s of dense BF16" — come from **Tensor Cores**, dedicated hardware units that compute small matrix multiplications in a single operation.

#### What a Tensor Core does: the MMA instruction

A Tensor Core executes a **matrix-multiply-accumulate (MMA)** operation. In one instruction it computes

$$ D = A \times B + C $$

where `A`, `B`, `C`, `D` are small matrix *tiles* (e.g. 16×16) rather than scalars. A single MMA instruction therefore performs thousands of multiply-adds at once. Compare:

- A CUDA core: one FP32 multiply-add per cycle → 2 floating-point operations (FLOPs; one multiply + one add).
- A Tensor Core MMA on, say, a 16×8×16 tile (an M×N×K shape: M=16, N=8, K=16): computes a 16×8 output from a 16×16 by 16×8 multiply = 16 × 8 × 16 = 2,048 multiply-adds = 4,096 FLOPs in roughly one instruction.

That density is why Tensor Cores deliver an order of magnitude more throughput than the CUDA cores on the same chip. On an A100, FP32 CUDA-core peak is ~19.5 TFLOP/s, while the Tensor Cores hit ~312 TFLOP/s in BF16 — a ~16× gap. **Essentially all** of the deep-learning performance you care about — every matrix multiply in a transformer's attention and feed-forward layers — runs on Tensor Cores, dispatched automatically by cuBLAS / cuDNN when your data is in a supported precision and your matrix dimensions are friendly (typically multiples of 8 or 16).

The pattern $D = A \times B + C$ is exactly the **accumulate** step a matrix multiply needs: a large matmul is tiled into many small tile-multiplies whose partial products are summed into an accumulator. The Tensor Core accumulates in higher precision (usually FP32) even when the inputs `A`, `B` are in a lower precision like BF16 — this is what makes mixed-precision training numerically stable (covered in the precision section).

#### Supported precisions and the precision/throughput tradeoff

Tensor Cores support a menu of numeric formats, and throughput roughly *doubles each time you halve the bit width*, because narrower operands let more multiply-adds fit in the same silicon and bandwidth. Representative input formats, in rough order of introduction:

| Format | Bits | Typical use | Relative Tensor-Core throughput |
|---|---|---|---|
| FP16 (half) | 16 | Training/inference | 1× (baseline) |
| BF16 (bfloat16) | 16 | Training/inference (preferred) | 1× |
| TF32 (TensorFloat-32) | 19-bit internal format | Drop-in FP32 matmul | ~0.5× of FP16 |
| FP8 (E4M3 / E5M2) | 8 | H100+ inference/training | ~2× of FP16 |
| INT8 | 8 | Quantised inference | ~2× of FP16 |
| FP4 / INT4 | 4 | Blackwell-era inference (FP4 on Blackwell; INT4 from Turing) | ~4× of FP16 |

A few clarifications you will need:

- **BF16 vs FP16.** Both are 16 bits. FP16 has 5 exponent bits and 10 mantissa bits; BF16 has 8 exponent bits and 7 mantissa bits. BF16 keeps the *same dynamic range as FP32* (same exponent width) at the cost of precision, which is why it has largely won for training — it rarely overflows/underflows and needs no loss-scaling. FP16 has more precision but a narrower range. For your inference work, weights are commonly served in BF16.
- **TF32** is a sleight of hand: it takes FP32 inputs but internally uses a format with 8 exponent bits and a 10-bit mantissa (plus sign) — i.e. it rounds the mantissa to 10 bits before multiplying, then accumulates in FP32. It gives you most of FP32's range with Tensor-Core speed, and PyTorch can enable it for `matmul`/`conv` with a single flag. (The "19-bit" label counts 1 sign + 8 exponent + 10 mantissa bits the unit operates on internally; it is a compute mode, not a stored tensor dtype.)
- **FP8 and below** trade accuracy for speed and are the frontier of inference efficiency; they need careful per-tensor (or finer-grained) scaling to stay accurate.

#### Structured sparsity

NVIDIA Tensor Cores from Ampere (A100) onward support **2:4 structured sparsity**: if you prune a weight matrix so that in every contiguous group of 4 values, at least 2 are zero, the hardware can skip the zeros and run the matmul at up to **2× throughput**. The "structured" qualifier matters — the zeros must follow the regular 2-of-4 pattern, not be scattered arbitrarily, because the hardware exploits the fixed pattern to compress storage and routing. In practice this requires a pruning-and-fine-tuning step and is used more in production inference than in research; the advertised "sparse TFLOP/s" figures (e.g. ~624 sparse vs ~312 dense BF16 on A100) assume you have applied it.

### 2.4 The memory hierarchy in full detail

If you remember one thing for performance work, make it this: **modern deep-learning workloads are usually limited by memory, not by arithmetic.** The Tensor Cores can consume data far faster than main memory can supply it, so the art of GPU performance is largely the art of *keeping data close to the ALUs and reusing it*. To reason about that you must know the memory hierarchy — a ladder from tiny-and-instant at the top to huge-and-slow at the bottom.

The numbers below are **order-of-magnitude, A100/H100-representative**, and vary by GPU and clock. Latencies are in clock cycles (at ~1.4 GHz, 1 cycle ≈ 0.7 ns).

| Level | Scope | Approx. size | Approx. latency | Approx. bandwidth | Managed by |
|---|---|---|---|---|---|
| Registers | Per-thread | 256 KB/SM (file) | ~1 cycle | ~tens of TB/s (aggregate) | Compiler |
| Shared memory / L1 | Per-block (per-SM) | up to 164 KB/SM (A100) – 228 KB/SM (H100) | ~20–30 cycles | ~tens of TB/s on-chip | **Programmer** (shared) / HW (L1) |
| L2 cache | Whole GPU | 40 MB (A100) / 50 MB (H100) | ~200 cycles | ~several TB/s | Hardware |
| Global memory (HBM) | Whole GPU | 40–141 GB | ~400–800 cycles | ~1.6–4.8 TB/s | Hardware |
| Host RAM (over PCIe) | CPU side | 100s of GB | ~microseconds | ~32–64 GB/s (PCIe 4.0/5.0 ×16) | You (explicit copies) |

Walk down the ladder:

#### Registers

**Registers** are the fastest storage, private to a single thread, accessed in ~1 cycle. Each thread's local variables live here. The whole SM shares one large **register file** (256 KB = 65,536 32-bit registers on A100 and H100), partitioned among all resident threads. This is critical for occupancy (Section 2.6): if each thread demands many registers, fewer threads fit, and you lose latency-hiding parallelism. The compiler decides register allocation; you influence it indirectly (kernel complexity, compiler flags, launch bounds).

#### Shared memory and L1: the programmer-managed cache

Each SM has a block of fast on-chip SRAM (static RAM) that is split between two roles:

- **L1 cache** — managed automatically by hardware, like a CPU cache.
- **Shared memory** — *managed explicitly by you, the programmer*. This is the defining feature. Shared memory is a scratchpad visible to all threads in a **block** (Section 3), with latency near L1 (~20–30 cycles), an order of magnitude faster than global memory.

On A100 and H100 the L1 and shared memory share one physical SRAM pool, partitioned at kernel launch: A100 has a 192 KB/SM pool (up to 164 KB configurable as shared memory), and H100 has a 256 KB/SM pool (up to 228 KB configurable as shared memory).

Why does programmer-managed memory exist? Because the programmer often knows the reuse pattern better than any automatic cache could infer. The classic example is **tiled matrix multiplication**: to multiply two large matrices, you load a small tile of each into shared memory once, then have all the threads in the block reuse those tiles many times from fast on-chip memory, instead of each thread fetching from slow global memory repeatedly. This is exactly how high-performance matmul and **FlashAttention** work — they are carefully choreographed to stage data through shared memory so the HBM is read as few times as possible.

A **cache hit** means the data you requested was already present in a cache level, so you got it at that level's (fast) latency instead of paying the full trip to global memory. **Locality** is the property that makes hits likely: *temporal* locality (you reuse the same address soon) and *spatial* locality (you access addresses near ones you just touched). Caches and shared-memory tiling both exist to exploit locality — and a workload with poor locality (random scattered access) defeats them and runs at the speed of global memory.

#### L2 cache

The **L2 cache** is a single GPU-wide cache (40 MB on A100, 50 MB on H100) shared by *all* SMs on the GPU. It sits between the SMs and HBM. When an SM misses in its L1, it asks L2; only on an L2 miss does the request go to HBM. L2 is hardware-managed and is the last line of defence against paying full HBM latency. Because it is shared chip-wide, it also serves as a coherence point between SMs. (On the H100, the 50 MB L2 is physically split into two partitions joined by a crossbar, so latency to the "far" half is somewhat higher — a detail you rarely have to reason about explicitly.)

#### Global memory (HBM): the main VRAM

**Global memory** is the GPU's main memory — the "40 GB" or "80 GB" figure on the spec sheet. This is where your model weights, activations, and KV cache live. On datacenter cards it is **HBM (High Bandwidth Memory)** — DRAM dies stacked vertically and connected to the GPU through a very wide interface on a silicon interposer (a shared substrate that carries thousands of wires between the memory stacks and the GPU die). It is:

- **Large**: 40–141 GB depending on the card.
- **High-bandwidth**: ~1.6–4.8 TB/s.
- **High-latency**: ~400–800 cycles. Every access here is what the warp schedulers work to hide.

This is the level you must budget carefully. A 7B-parameter model in BF16 needs 7 × 10⁹ × 2 bytes = **14 GB just for weights** — before any activations, KV cache, or optimizer state. That single calculation tells you a 7B model in BF16 cannot be served on a 12 GB consumer card but fits comfortably on a 40 GB A100. (Memory budgeting is developed fully in the inference section.)

#### Host RAM over PCIe / NVLink

Below the GPU's own memory sits the **host (CPU) RAM**, reachable only by an explicit copy across the system bus. The standard bus is **PCIe (Peripheral Component Interconnect Express)**; PCIe 4.0 ×16 delivers ~32 GB/s and PCIe 5.0 ×16 ~64 GB/s each way — note that this is *one to two orders of magnitude slower than HBM*. Crossing this bus is therefore expensive, and a recurring performance bug is accidentally shuttling data host↔device in a hot loop. This is also why "offloading" weights to CPU RAM (to fit a model that exceeds VRAM) is slow: every layer's weights must crawl across PCIe. Some systems replace PCIe with **NVLink-C2C** for far higher host↔device bandwidth (e.g. the Grace-Hopper and GB10 superchips; Section 2.7).

#### What "memory bandwidth" physically means

**Memory bandwidth** is the rate at which bytes move between the GPU cores and HBM, in bytes per second. Physically it is `(bus width in bits / 8) × (effective data rate per pin)`. HBM achieves its huge bandwidth not by being clocked fast per pin but by being *very wide*: a 5120-bit bus on A100 (versus ~256–384 bits for a desktop GDDR card), moving data on both clock edges. Bandwidth, not capacity, is the number that usually bounds throughput, because of **arithmetic intensity**: the ratio of FLOPs performed per byte loaded. If a kernel does little arithmetic per byte (e.g. an element-wise add, or autoregressive decoding of a single token where you re-read all the weights to produce one token), it is **memory-bound** — the ALUs starve waiting on HBM, and your effective TFLOP/s are a small fraction of peak no matter how fast the Tensor Cores are. If it does much arithmetic per byte (e.g. a big batched matmul), it is **compute-bound** and can approach peak. This dichotomy — captured by the *roofline model* — is the single most useful lens for predicting and diagnosing GPU performance, and it is why batching matters so much for inference (it raises arithmetic intensity by reusing each loaded weight across many tokens).

### 2.5 Memory coalescing and bank conflicts

Knowing the hierarchy is not enough; *how* threads address memory determines whether you get full bandwidth or a small fraction of it. Two access-pattern rules dominate.

#### Coalescing (global memory)

When the 32 threads of a warp each issue a global-memory load in the same instruction, the hardware does **not** perform 32 separate fetches. It inspects the 32 addresses and groups them into the minimum number of aligned memory **transactions** (each transaction moves a fixed chunk, typically 32, 64, or 128 bytes). If the 32 threads access 32 *consecutive* 4-byte words — thread 0 reads address `base+0`, thread 1 reads `base+4`, …, thread 31 reads `base+124` — all 128 bytes fall in one aligned 128-byte transaction. This is a **coalesced** access: one transaction serves the whole warp, and you get full bandwidth.

If instead the threads access *scattered* addresses (e.g. each strided far enough to land in a different transaction), the hardware must issue up to 32 separate transactions to satisfy one instruction. Each transaction still moves a full 32–128 byte chunk but you *use* only 4 bytes of it — so you waste most of the bandwidth you paid for (up to 31/32 ≈ 97% in the worst case, with 32-byte minimum transactions). This is an **uncoalesced** (or scattered) access, and it is the classic reason a kernel runs at a fraction of HBM's rated bandwidth.

The practical rule: **lay your data out so that adjacent threads touch adjacent memory.** For matrices this means caring about row-major vs column-major layout and ensuring the fastest-varying thread index maps to the fastest-varying (contiguous) memory dimension. In PyTorch you mostly inherit good layouts from the library kernels, but the moment you write a custom CUDA/Triton kernel, or do an ill-considered transpose/gather, coalescing becomes your problem. A `.contiguous()` call or a transpose that breaks coalescing can silently cost you several-fold bandwidth.

#### Bank conflicts (shared memory)

Shared memory has its own access hazard. It is physically divided into **banks** — 32 independent 4-byte-wide memory modules — so that, ideally, all 32 lanes of a warp can each read/write a *different* bank simultaneously in one cycle. A **bank conflict** occurs when two or more lanes in the warp access *different addresses that fall in the same bank*. The hardware cannot service them at once, so it serialises: a 2-way conflict halves throughput, an N-way conflict cuts it N-fold.

The mapping is by address: bank = (address / 4 bytes) mod 32. So if 32 threads access shared-memory locations strided by 32 four-byte words, they all hit the same bank — a catastrophic 32-way conflict. The classic fix is **padding**: declare a shared tile as `[32][33]` instead of `[32][32]` so that the column stride that used to align everything to one bank now spreads accesses across all banks. (A special case: if all lanes read the *same* address, the hardware *broadcasts* it for free — no conflict.) As with coalescing, you only confront this when writing custom kernels; the library kernels you call from PyTorch are already padded and tuned.

### 2.6 Occupancy: necessary but not sufficient

**Occupancy** is the ratio of *active warps resident on an SM* to the *maximum number of warps the SM can hold*. If an SM can host 64 warps and your kernel keeps 48 resident, occupancy is 48/64 = 75%. It is a measure of how full the latency-hiding "ready pool" is.

Why does it matter? Section 2.1's whole story — covering memory stalls by switching to another ready warp — only works if there *are* other resident warps to switch to. Low occupancy means few warps in the pool, which means when they stall there is nothing to run, and the SM idles. So you generally want *enough* occupancy to hide latency.

#### What limits occupancy

Each SM has a fixed budget of three resources, and a kernel's per-block demands carve into all three. Whichever runs out *first* caps how many blocks (hence warps) can be resident:

1. **Registers.** The register file (65,536 32-bit registers/SM on A100/H100) is divided among all resident threads. If each thread needs 64 registers, the SM can hold 65,536 / 64 = 1,024 threads = 32 warps. If a complex kernel needs 128 registers/thread, only 512 threads = 16 warps fit — occupancy halves. Register pressure is the most common occupancy limiter for non-trivial kernels.
2. **Shared memory.** The shared-memory pool (up to 164 KB/SM on A100, 228 KB/SM on H100) is divided among resident blocks. If each block declares 48 KB of shared memory, at most 164/48 ≈ 3 blocks fit on an A100, regardless of how few registers they use.
3. **Block/warp count limits.** Hard architectural caps: a maximum number of resident blocks per SM (32 on A100/H100) and a maximum number of resident warps per SM (64 on A100/H100). Tiny blocks (say 32 threads = 1 warp) can hit the block-count cap before filling the warp cap, leaving the SM underpopulated.

These interact: the achievable occupancy is set by the *most constraining* of registers, shared memory, and the block/warp caps. NVIDIA's *occupancy calculator* (now built into Nsight Compute) and `nvcc --ptxas-options=-v`, which prints per-kernel register and shared-memory usage, let you compute this before running.

#### Why "necessary but not sufficient"

It is tempting to chase 100% occupancy, but **high occupancy does not guarantee high performance, and maximum performance often occurs below 100% occupancy.** Two reasons:

- **Latency hiding saturates.** You only need *enough* warps to cover the memory latency; beyond that, more warps buy nothing. If 50% occupancy already keeps the schedulers fed, pushing to 100% is wasted effort.
- **The famous counter-case is register-rich kernels.** A kernel can run *faster* at lower occupancy by using *more* registers per thread to keep data in the fastest storage and reduce memory traffic — even though those extra registers reduce occupancy. Volta-era and later high-performance GEMM (general matrix multiply) kernels deliberately do this: they run at modest occupancy but achieve near-peak throughput because each thread does a lot of work with heavily reused, register-resident data. This is the essence of Vasily Volkov's well-known result that "better performance at lower occupancy" is achievable by raising instruction-level parallelism per thread.

So treat occupancy as a *diagnostic floor*, not a *target ceiling*: very low occupancy (a few warps) almost always means you are leaving latency-hiding on the table and should investigate; but once you are in a healthy range, the real questions are arithmetic intensity, coalescing, and Tensor-Core utilisation, not squeezing the last few percent of occupancy.

### 2.7 HBM vs GDDR, and a datacenter GPU comparison

#### HBM vs GDDR

Two families of GPU main memory exist, and which one a card uses tells you a lot about its intended market:

- **GDDR (Graphics Double Data Rate)** — the memory on consumer/gaming cards (RTX 4090, etc.) and some inference cards. Standard DRAM chips placed *around* the GPU on the board, connected by a relatively narrow bus (256–384 bits). Cheaper, lower bandwidth (~1 TB/s on a high-end GDDR6X card like the RTX 4090). Achieves speed mainly through high per-pin data rates.
- **HBM (High Bandwidth Memory)** — the memory on datacenter cards (A100, H100). DRAM dies *stacked vertically* and placed on the same package as the GPU via a silicon interposer, connected by an enormously wide bus (5120-bit on A100, 6144-bit class on later parts). Far higher bandwidth (~1.6–4.8 TB/s), lower power per byte, but more expensive to manufacture. This is the memory you want for large-model work because bandwidth, as Section 2.4 argued, is usually the binding constraint.

#### Representative datacenter GPUs

The table below is **approximate and version/SKU-dependent** — vendors ship 40 GB and 80 GB variants, SXM (socketed, higher-power) versus PCIe (lower-power) form factors, and revise clocks over time. BF16/FP16 figures are *dense* Tensor-Core peaks; the corresponding sparse (2:4) figures are ~2× higher. Use these for capacity and roofline estimates, not as guarantees.

| GPU | Arch | VRAM | Mem type | Mem BW (approx) | BF16/FP16 dense TFLOP/s (approx) | FP8 dense TFLOP/s | Notes |
|---|---|---|---|---|---|---|---|
| V100 | Volta | 16 / 32 GB | HBM2 | ~0.9 TB/s | ~125 | — | First mainstream Tensor Cores; FP16 only |
| A100 | Ampere | 40 / 80 GB | HBM2 / HBM2e | ~1.55 / ~2.0 TB/s | ~312 | — | TF32, BF16, 2:4 sparsity; workhorse |
| H100 (SXM) | Hopper | 80 GB | HBM3 | ~3.35 TB/s | ~990 | ~1,979 | Adds FP8, Transformer Engine |
| H200 | Hopper | 141 GB | HBM3e | ~4.8 TB/s | ~990 | ~1,979 | Same compute as H100, much more/faster memory |
| B200 (Blackwell) | Blackwell | ~192 GB | HBM3e | ~8 TB/s | ~2,250 per die (~4,500 full chip) | ~4,500/die (~9,000 chip, FP4 dense) | Dual-die package; adds FP4; figures evolving |
| GH200 (Grace+Hopper) | Hopper+Grace CPU | 96/144 GB GPU + LPDDR | HBM3 / HBM3e | ~4–4.9 TB/s | ~990 | ~1,979 | CPU+GPU coherent via NVLink-C2C |
| **DGX Spark (GB10)** | Grace-Blackwell (Blackwell) | 128 GB **unified** LPDDR5X | LPDDR5X (not HBM) | **~273 GB/s** | quoted "1 PFLOP" sparse FP4 | — | Desktop dev box; see below |

A note on each you will actually touch:

- **A100 / H100 on RunPod and lab clusters.** These are your training/large-inference workhorses. The 80 GB A100 and 80 GB H100 are the sweet spot for serving 7B–70B models (with quantisation/parallelism for the larger ones). The H100's headline advantage over the A100 is ~3× the dense Tensor-Core throughput *and* FP8 support; its memory is also ~1.7× faster (3.35 vs ~2.0 TB/s), which matters for the memory-bound decode phase of inference.
- **The DGX Spark (GB10 "Grace Blackwell").** This is a small desktop developer appliance, not a datacenter card, and its memory system is fundamentally different — and you must internalise this. It has **128 GB of *unified* LPDDR5X memory** (low-power DDR, the kind in laptops/phones) shared coherently between the 20-core Grace-class Arm CPU and the Blackwell GPU. The huge *capacity* (128 GB) lets you *load* very large models — a 70B model in 4-bit fits, and NVIDIA markets it for models up to ~200B parameters — that simply will not fit on a 24 GB consumer card. But the memory **bandwidth is only ~273 GB/s**, roughly **12× lower than an H100's HBM3** (3.35 TB/s) and ~17× lower than an H200's HBM3e (4.8 TB/s). Because autoregressive token generation is memory-bandwidth-bound (you re-stream the weights for every token), the Spark will *generate tokens far more slowly per second* than an A100/H100 even when it can hold a bigger model. The right mental model: the Spark is a **capacity-rich, bandwidth-poor** box — excellent for *developing*, debugging, and prototyping pipelines on real large models locally and at low cost, but not for throughput-critical serving or fast training. Run correctness and integration there; run timing-sensitive experiments on rented A100/H100s. Its advertised "1 petaFLOP" figure is a *sparse FP4* number and is not comparable to the dense BF16 columns above.

### 2.8 Inter-GPU interconnect: PCIe vs NVLink/NVSwitch

The moment a model or batch outgrows one GPU, performance hinges on how fast GPUs can talk to *each other* — because multi-GPU execution constantly exchanges data (gradients in training, sharded activations/weights in inference). There are two regimes.

#### PCIe

If GPUs communicate over **PCIe**, they route through the system bus described earlier: PCIe 4.0 ×16 ≈ 32 GB/s, PCIe 5.0 ×16 ≈ 64 GB/s per direction, and often *through host memory or a PCIe switch*, adding latency. Relative to HBM bandwidth (TB/s), PCIe is glacial. For workloads with heavy inter-GPU traffic — tensor parallelism, where every layer's matmul is split across GPUs and results must be summed every layer — PCIe becomes the bottleneck and you lose much of the benefit of extra cards.

#### NVLink and NVSwitch

**NVLink** is NVIDIA's dedicated high-bandwidth GPU-to-GPU interconnect, bypassing PCIe entirely. Each successive generation roughly doubles bandwidth; an H100 exposes **NVLink 4 at 900 GB/s** of aggregate bidirectional bandwidth per GPU (18 links × 50 GB/s) — *more than an order of magnitude* faster than PCIe 5.0 (~14× the PCIe 4.0 ×16 bus), and within a few-fold of HBM. **NVSwitch** is a crossbar switch chip that connects *many* NVLinks so that every GPU in a server (e.g. all 8 GPUs in a DGX/HGX node) can talk to every other at full NVLink bandwidth simultaneously, not just neighbour-to-neighbour. The combination (NVLink + NVSwitch) is what makes an 8-GPU node behave almost like one large GPU for the purposes of **collective operations** — the group communication primitives (all-reduce, all-gather, reduce-scatter) that parallel training and inference rely on, in which all GPUs cooperatively combine or redistribute a tensor. (These collectives and the parallelism strategies that use them are developed fully in the multi-GPU section; here the point is simply that the *interconnect* sets their speed.)

The practical consequence for you: when renting multi-GPU instances, **whether the GPUs are NVLink-connected or merely PCIe-connected can matter more than the GPU model itself** for communication-heavy jobs. A pair of NVLinked A100s can outperform a pair of PCIe-only H100s on a tensor-parallel workload bottlenecked by inter-GPU traffic. Always check the interconnect topology of an instance (`nvidia-smi topo -m` prints the GPU-to-GPU link matrix) before assuming multi-GPU scaling will be linear. AMD's analogue to NVLink is **Infinity Fabric**; the conceptual story — a dedicated fast fabric plus a switch for all-to-all — is the same.

---

## 3. The CUDA programming model and the ML software stack

When you call `model(input_ids)` in PyTorch, an enormous amount of machinery sits between that Python line and the electrons moving through your A100 or H100. This section dissects that machinery layer by layer, from the abstract CUDA execution model down to the compilation chain, and back up to the tools — `torch.compile`, CUDA Graphs, Triton, FlashAttention — that a PyTorch user can actually reach for. The goal is for you to understand not just *what* to type, but *why* a given line of Python turns into fast or slow silicon behaviour, so that when you profile a sluggish inference loop you know which layer to suspect.

### 3.1 The CUDA programming model: kernels and the thread hierarchy

#### What a kernel is

A **kernel** is a function that runs on the GPU (the *device*), launched from code running on the CPU (the *host*). The defining feature of CUDA's model is that a single kernel launch does not run *once* — it runs *many thousands of times in parallel*, once per **thread**, with each thread distinguished only by its index. You write the body of the computation as if for a single thread; the hardware replicates it across a vast grid of threads. This is the **SIMT** model — **Single Instruction, Multiple Thread** — a cousin of the SIMD (Single Instruction, Multiple Data) vector model you may know from CPUs, but with per-thread control flow.

A trivial example, adding two vectors `C = A + B`, written in CUDA C:

```cuda
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) C[i] = A[i] + B[i];                   // do one element
}
```

The `__global__` qualifier marks this as a kernel — host-callable, device-executed. Every thread runs the *same* four lines, but `threadIdx`, `blockIdx`, and `blockDim` differ per thread, so each touches a different array element. You, the programmer, never write a loop over `n`; the parallelism *is* the loop, unrolled across hardware.

#### The grid → block → thread hierarchy

CUDA organises threads in a strict three-level hierarchy. Understanding this hierarchy is the single most important thing in the programming model, because each level maps to a distinct piece of hardware with distinct performance consequences.

- **Thread**: the finest unit. Has its own registers and program counter; executes the kernel body.
- **Block** (or **thread block**): a group of threads — up to 1024 on current hardware — that are guaranteed to run *on the same Streaming Multiprocessor (SM)* and can therefore cooperate. They share a small, fast scratchpad called **shared memory** and can synchronise with a barrier (`__syncthreads()`). Threads in *different* blocks cannot cheaply cooperate. (Hopper adds an optional intermediate level, the *thread-block cluster*, which lets a handful of blocks on neighbouring SMs share memory; you can ignore it for the mental model here.)
- **Grid**: the entire collection of blocks launched by one kernel call. Blocks in a grid are mutually independent and may run in any order, concurrently or sequentially, on any SM. This independence is what lets the same binary scale from a small GPU with a handful of SMs to an H100 with 132 SMs without recompilation — the runtime simply streams blocks onto whatever SMs are free.

The dimensions are configured at launch time with the `<<<grid, block>>>` syntax:

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // ceil(n / 256)
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);
```

For `n = 1,000,000` elements and 256 threads per block, that is `ceil(1000000 / 256) = 3907` blocks, totalling `3907 × 256 = 1,000,192` threads — slightly more than `n`, which is exactly why the `if (i < n)` guard exists: the tail block has 192 idle threads that must not write out of bounds.

Grids and blocks can be 1-, 2-, or 3-dimensional (`dim3`), which is purely a convenience for indexing into multi-dimensional data like images or matrices; a 2-D block of `16×16` is just 256 threads with two index components `threadIdx.x` and `threadIdx.y`.

#### Warps: the real unit of execution

Here is the crucial detail the high-level model hides. The hardware does **not** schedule individual threads. It schedules **warps** — groups of **32 consecutive threads** that execute *in lockstep*, sharing a single instruction-fetch and a single program counter. A block of 256 threads is internally 8 warps. The warp size has been 32 on every NVIDIA GPU to date; design as if it will never change, but read it from the device properties rather than hard-coding it in portable library code. (Since Volta, NVIDIA hardware also supports *independent thread scheduling*, which keeps separate per-thread program counters so threads in a divergent warp can make forward progress independently — but the warp is still issued one path at a time, so the throughput cost of divergence below still applies.)

Two performance consequences fall directly out of the warp abstraction, and both will bite you eventually:

1. **Warp divergence.** Because the 32 threads share instruction issue, if a data-dependent branch (`if (x[i] > 0) ... else ...`) sends some threads down the `if` and others down the `else`, the warp must execute *both* paths serially, masking off the inactive threads on each pass. A fully divergent branch can halve throughput; a 32-way divergent `switch` can cut it by far more. This is why GPU-friendly code prefers branchless arithmetic and data layouts where neighbouring threads take the same path.

2. **Memory coalescing.** When the 32 threads of a warp issue a load, the memory system services them most efficiently if their addresses fall within the same aligned region — then the hardware **coalesces** the requests into a minimal number of wide memory transactions (the L1/L2 cache line is 128 bytes, serviced as four 32-byte sectors). If instead each thread reads a far-flung address (a *strided* or *scattered* access, e.g. column-major reads of a row-major array), the warp issues many separate transactions and effective bandwidth collapses — in the worst case toward 1/32 of peak. "Make thread `i` read element `i`" is the golden rule, and it is precisely why data layout (row- vs column-major, AoS vs SoA) is a first-order performance concern, not a detail.

#### Occupancy: hiding latency with parallelism

Each SM can host many resident warps at once — far more than it has execution units. An H100 SM, for instance, can keep up to 64 warps (2048 threads) resident. The reason is **latency hiding**: when one warp stalls waiting on a memory load (hundreds of cycles to reach HBM — the GPU's main memory), the SM's warp scheduler instantly switches to another *ready* warp, with zero context-switch cost because every warp's registers stay live in the giant register file the whole time. With enough warps in flight, the SM's arithmetic units never idle.

**Occupancy** is the ratio of resident warps to the hardware maximum. It is *bounded* by the scarcest per-SM resource your kernel consumes:

- **Registers.** Each SM has a fixed register file of 65,536 32-bit registers (this has held across Volta, Ampere, and Hopper). If your kernel uses 64 registers per thread, then `65536 / 64 = 1024` threads = 32 warps can be resident — already capping occupancy at 50% of a 64-warp maximum. Register-hungry kernels (lots of live variables, heavy unrolling) "spill" to local memory or simply limit occupancy.
- **Shared memory.** Each SM has a fixed combined shared-memory/L1 budget — 256 KB on H100, of which up to 228 KB can be carved out as shared memory (the rest stays L1). A block requesting 100 KB of shared memory lets at most 2 blocks co-reside, regardless of thread count. (The Ampere A100's combined budget is 192 KB, with up to 164 KB usable as shared memory — a reminder that these figures are architecture-specific.)
- **Block count / thread count** hardware limits (e.g. a maximum of 32 resident blocks per SM on Hopper).

Higher occupancy is *usually* better but not always — a kernel that is already bandwidth-bound, or that deliberately uses extra registers and shared memory to do useful blocking (like FlashAttention), can run faster at lower occupancy. The practical lesson: occupancy is a knob you can *inspect* (the CUDA Occupancy Calculator, or Nsight Compute reports it) and *trade against* register/shared-memory usage, not a number to blindly maximise.

#### How blocks map to SMs — the mental model

To summarise the mapping, which is the heart of why GPUs scale:

| Software concept | Hardware it maps to | Key property |
|---|---|---|
| Thread | A lane within a warp; private registers | Has unique index; cheapest unit |
| Warp (32 threads) | The scheduling/execution unit on an SM | Lockstep issue; divergence & coalescing live here |
| Block | One SM (entirely); shared memory + `__syncthreads()` | Cooperation only *within* a block |
| Grid | The whole GPU; all SMs | Blocks independent → automatic scaling |

The host launches a grid; the **GigaThread** engine hands blocks out to SMs as they become free; each SM runs its blocks' warps, interleaving them to hide latency; when a block finishes, the SM is handed a new one until the grid is exhausted. As a PyTorch user you never write this, but every fast kernel you call — every cuBLAS GEMM, every FlashAttention pass — is organised exactly this way, and the profiler vocabulary (occupancy, divergence, coalescing, achieved bandwidth) is the language you will read its performance in.

### 3.2 The execution model: asynchrony, streams, events

#### Asynchronous launch and CPU/GPU concurrency

A kernel launch is **asynchronous**: when the host thread executes `vecAdd<<<...>>>(...)`, it does *not* wait for the GPU to finish. It enqueues the kernel into a **stream** (a queue of GPU work) and returns almost immediately — typically in a few microseconds — to run the next Python/C++ line. The CPU and GPU then execute *concurrently*. This is by design and is the central performance idea of the whole execution model: while the GPU chews on layer *N*, the CPU is already preparing and enqueuing the kernels for layers *N+1*, *N+2*, … so the GPU is never starved.

The consequence for measuring time is profound and trips up nearly everyone once: a naive

```python
t0 = time.time()
y = model(x)               # returns almost instantly — work is only *queued*
t1 = time.time()           # measures launch overhead, NOT compute!
```

measures essentially nothing, because the GPU work has only been *enqueued*, not completed. To time GPU work you must **synchronise** — block the CPU until the GPU drains its queue — with `torch.cuda.synchronize()` before reading the clock, or better, use CUDA **events** (below) which timestamp on the GPU's own timeline.

#### Streams

A **CUDA stream** is an ordered queue of operations (kernels, memory copies). The contract is:

- Operations **within one stream** execute in strict issue order, one after another.
- Operations in **different streams** have no ordering guarantee and may overlap / run concurrently if resources allow.

By default, all PyTorch work goes onto a single **default stream** per device, so your kernels run in program order — which is what you want for correctness and almost always what you want, full stop. Streams become useful when you want to *overlap independent work*: classically, overlapping a host-to-device data copy of the *next* batch with the compute of the *current* batch, so the PCIe/NVLink transfer hides behind computation. Multiple streams are also how two independent models, or the prefill and decode phases of an inference server, can share one GPU. For the typical single-model inference loop you will write, you rarely create streams by hand; it is enough to know they exist and that the default-stream serialisation is why your code is correct without you thinking about it.

#### Events and synchronization

A **CUDA event** is a lightweight marker you insert into a stream; the GPU records a timestamp (or a "reached here" signal) when it processes that point. Events serve two purposes:

1. **Accurate timing.** Because the timestamp is taken on the GPU timeline, events measure true device elapsed time without conflating CPU launch overhead. The canonical idiom:

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
y = model(x)
end.record()
torch.cuda.synchronize()          # wait until 'end' has actually been reached
ms = start.elapsed_time(end)      # milliseconds of real GPU time
```

2. **Cross-stream synchronization.** One stream can *wait* on an event recorded in another (`stream.wait_event(e)`), letting you express "don't start the compute in stream B until the copy in stream A has finished" without a full device-wide barrier.

The hierarchy of synchronization primitives, from coarsest to finest: `torch.cuda.synchronize()` blocks the CPU until *all* GPU work on the device is done; stream synchronization waits for one stream; event synchronization waits for one point. Reach for the finest one that gives correctness — coarse `synchronize()` calls inside a hot loop are a classic, self-inflicted performance bug because they destroy the CPU/GPU overlap described above.

#### The launch-overhead problem for small kernels

Every kernel launch costs a roughly fixed **launch overhead** — on the order of a few microseconds (commonly quoted around 3–10 µs, with a hard floor near 5 µs for a do-nothing kernel) of CPU-side work to validate arguments, set up the launch, and enqueue it. This is negligible when the kernel then runs for milliseconds, but it is *ruinous* when the kernel itself is tiny. Consider an element-wise bias-add on a small tensor that takes 2 µs to compute but 5 µs to launch: you are spending more time dispatching the work than doing it, and the GPU sits idle between launches waiting for the CPU to feed it. This regime is called being **launch-bound** (or **CPU-bound** / **dispatch-bound**), and it is the dominant failure mode of LLM *decode* — generating one token at a time means a long sequence of small, individually launch-bound kernels.

This single problem motivates three of the most important tools in the modern stack — **kernel fusion**, **CUDA Graphs**, and bigger batches — each attacking it from a different angle, all covered below. Recognising "my GPU utilisation is low and my kernels are small" as the *launch-bound* signature is one of the highest-leverage diagnostic skills you can build.

### 3.3 The library stack you actually use

Between your Python and the silicon sits a tower of libraries. Knowing what each layer is responsible for tells you *where* a bug or a bottleneck lives.

| Layer | Examples | What it does for you |
|---|---|---|
| **Hardware** | A100, H100, DGX Spark (GB10) | SMs, Tensor Cores, HBM |
| **Driver** | NVIDIA kernel driver (`nvidia.ko`) | Talks to the hardware; one per machine, version-sensitive |
| **CUDA Toolkit / runtime** | `libcudart`, `nvcc`, PTX, the CUDA C API | The base programming model: launch kernels, allocate device memory, manage streams |
| **Math / DNN primitive libraries** | **cuBLAS** (dense linear algebra / GEMM), **cuDNN** (conv, attention, normalisation, RNN), **CUTLASS** (template library for writing your own GEMMs), cuFFT, cuSPARSE, **NCCL** (multi-GPU collectives) | Hand-tuned, Tensor-Core-aware implementations of the heavy operations |
| **Framework** | **PyTorch**, **JAX**, TensorFlow | Tensors, autograd, the dispatcher that maps ops → kernels, device/memory management |
| **Compilers / kernel DSLs** | `torch.compile` (TorchInductor), **Triton**, XLA, TensorRT | Generate fused custom kernels; graph-level optimisation |
| **High-level / serving** | HuggingFace `transformers`, **vLLM**, **SGLang**, TensorRT-LLM, DeepSpeed, Accelerate | Model definitions, tokenisation, paged-attention KV-cache, batching, parallelism orchestration |

A few of these deserve a closer look because you will meet them by name in error messages and profiles:

- **cuBLAS** is where your matrix multiplies actually happen. Every linear layer, every attention QKᵀ and the output projection, bottoms out in a cuBLAS **GEMM** (GEneral Matrix Multiply) call — or in cuBLASLt, its more flexible successor that handles mixed-precision and fused epilogues (e.g. GEMM-then-bias-then-GELU in one kernel). When people say a transformer is "just a stack of matmuls," cuBLAS is the thing doing them, and it is among the most heavily optimised code on the box.
- **cuDNN** provides the neural-network-specific primitives: convolutions (with multiple algorithms it auto-tunes between), batch/layer norm, pooling, and increasingly fused attention. Less central for pure transformers than for CNNs, but its fused-attention and normalisation kernels matter.
- **CUTLASS** is not a black-box library but a C++/CUDA *template* library for *building* GEMM and convolution kernels with full control over tiling and Tensor-Core usage. You will not normally write CUTLASS, but many of the custom kernels you *benefit* from (including some FlashAttention and `torch.compile` backends) are built on it.
- **NCCL** (the NVIDIA Collective Communications Library, "nickel") implements the **collectives** — `all-reduce`, `all-gather`, `reduce-scatter`, `broadcast` — that synchronise gradients and shard tensors across multiple GPUs. A *collective* is a communication operation that *all* participating GPUs take part in together (as opposed to a point-to-point send/recv). The moment you go multi-GPU, NCCL's efficiency over NVLink/InfiniBand becomes a first-order determinant of throughput, and "NCCL hangs" is a rite of passage in distributed training.

The mental model: **PyTorch decides *what* to compute and in what order; cuBLAS/cuDNN/CUTLASS decide *how* to compute each heavy op fast; NCCL moves data *between* GPUs; the CUDA runtime and driver actually *launch* it on the hardware.**

### 3.4 How PyTorch turns Python into kernels

#### Eager mode and the dispatcher

By default PyTorch runs in **eager mode**: each tensor operation executes immediately as the Python interpreter reaches it, the way NumPy does. When you write `z = x @ w + b`, three things happen in sequence: `@` triggers a matmul op, `+` an add op, and each is *dispatched* to a concrete kernel.

That **dispatch** is a real, non-trivial mechanism. PyTorch's **dispatcher** looks at the operation (`aten::matmul`), the **device** of the tensors (CUDA vs CPU vs MPS), and the **dtype** (float32, bfloat16, …), and selects the registered kernel for that combination. For a CUDA float tensor, the matmul dispatch ends in a cuBLAS GEMM call; the add dispatches to a small element-wise CUDA kernel. Autograd is layered in here too: in `requires_grad` mode the dispatcher also records the op into the autograd graph so the backward pass can replay it. All of this — Python bytecode interpretation, dispatch key computation, kernel selection, argument marshalling — is **CPU-side work that happens *before* the asynchronous launch**, and it is exactly the overhead that makes small ops launch-bound.

#### Why the GIL does not matter on-device (but Python overhead does)

Python's **Global Interpreter Lock (GIL)** serialises Python bytecode execution, which terrifies people coming from CPU multithreading. On the GPU it is **irrelevant to the actual compute**: once a kernel is launched, it runs on the device entirely outside the interpreter, and the GIL has no bearing on those thousands of GPU threads. The GPU is the parallel machine; Python merely *dispatches* to it.

What the GIL (and Python generally) *does* affect is the **rate at which you can issue work**. The single Python thread, holding the GIL, walks your model's `forward`, and for each op pays interpreter overhead + dispatcher overhead before the few-microsecond launch. If your kernels are large (big batch, big matmul), this per-op CPU cost is hidden behind the asynchronous GPU execution and you are **GPU-bound** — the happy case. If your kernels are small (batch size 1 token-by-token decode, tiny tensors), the CPU cannot enqueue work fast enough, the GPU drains its queue and idles, and you are **CPU-bound / launch-bound** — the Python overhead becomes your wall-clock bottleneck even though the GPU is mostly idle. This is the concrete reason "make the batch bigger" and "compile the model" are the first two pieces of advice for slow inference: both raise the work-per-launch ratio so the fixed CPU/dispatch cost is amortised.

#### What "op" vs "kernel" means

Worth nailing precisely, because the stack blurs them: an **op** (operator) is a *logical* mathematical operation in the framework's vocabulary — `aten::add`, `aten::softmax`, `aten::layer_norm`. A **kernel** is a *concrete compiled GPU function* that implements an op (or part of one, or several fused together). The relationship is many-to-many: one op may dispatch to different kernels depending on dtype/shape/device; one fused kernel may implement several ops at once (the whole point of fusion). When you read a profiler trace, you see *kernels* (what ran on the GPU); when you read your model code, you see *ops*. Connecting the two — "this `softmax` op became these three kernels" — is the essence of performance debugging.

### 3.5 Kernel fusion and the tools that do it

#### What fusion is and why it matters

Recall from the memory-hierarchy discussion that **HBM** (the GPU's multi-gigabyte main memory) is *vast but slow* relative to the on-chip SRAM (registers and shared memory), and that many operations are **memory-bound** — limited by how fast you can move data to and from HBM, not by arithmetic. Element-wise ops (add, GELU, dropout, scaling, normalisation) are the canonical memory-bound operations: they do only a flop or two per element but must read the whole tensor from HBM and write the whole result back.

Now consider an unfused sequence `y = gelu(x @ w + b)` in eager mode. It runs as (at least) three kernels:

1. **matmul** → writes the full intermediate to HBM,
2. **bias add** → reads that intermediate from HBM, writes a new one to HBM,
3. **GELU** → reads *again* from HBM, writes the result to HBM.

Each of steps 2 and 3 makes a full **round-trip to global memory** for data that was *already on chip* a moment earlier. **Kernel fusion** merges these into a single kernel that keeps the intermediate values in registers/shared memory and writes only the final `y` back to HBM — eliminating the redundant reads and writes. For a memory-bound chain, fusing the cheap element-wise ops onto the producer can give close to an *N*× speedup *on that element-wise segment*, because you have cut its HBM traffic by roughly that factor. (In the example above the matmul itself is usually compute-bound and dominates, so the realistic win is eliminating the bias/GELU round-trips, not an N× speedup of the whole expression — fusion shines most on long chains of element-wise and reduction ops.)

A concrete sense of scale: for a tensor of 100M BF16 elements (200 MB), each avoided full round-trip saves reading + writing 400 MB. On a GPU with ~2 TB/s of HBM bandwidth (A100-class; an H100 SXM has ~3.35 TB/s, so the saving there is proportionally larger), that is ~0.2 ms saved *per avoided round-trip* — and a transformer has many such element-wise ops per forward pass.

#### The fusion tools a PyTorch user can reach

You do not hand-write fused CUDA. Four tools, in rough order of how often you will use them, do it for you:

**`torch.compile`** (PyTorch 2.x). The headline tool. Wrapping `model = torch.compile(model)` triggers, on the first call, a trace of your model via **TorchDynamo** (which captures the Python operations into a graph, falling back gracefully — a "graph break" — on bits it can't capture, like data-dependent Python control flow). The graph is handed to a backend, by default **TorchInductor**, which performs fusion and *generates Triton kernels* for the GPU. The net effect: many small eager ops collapse into a few fused kernels, slashing both launch count and HBM traffic. The costs to be aware of: a real **compilation latency** on first run (seconds to minutes), and **recompilation** if input shapes change (mitigate with `dynamic=True` or by padding shapes to a few fixed buckets). For inference this is usually the single biggest one-line win.

**CUDA Graphs.** A complementary attack on launch overhead specifically. A CUDA Graph *records* an entire sequence of kernel launches *once*, then *replays* the whole sequence with a single CPU-side call. Instead of the CPU paying per-kernel launch overhead every iteration, it pays it once at capture time; replay re-issues hundreds of kernels almost for free. This is devastatingly effective for the **decode** loop of LLM inference, where the same small graph of kernels runs once per generated token and would otherwise be hopelessly launch-bound. The constraints: the workload must be *static* (same shapes, same control flow, stable memory addresses every iteration) — which is exactly why fixed-shape decode with a pre-allocated KV-cache is the textbook fit. PyTorch exposes this via `torch.cuda.graph` / `make_graphed_callables`, and `torch.compile(mode="reduce-overhead")` applies CUDA Graphs for you.

**Triton.** OpenAI's Python-embedded DSL for writing GPU kernels. You write Python-looking code that operates on *blocks* of data, and Triton's compiler handles the warp-level details — thread assignment, shared-memory management, coalescing, and much of the tiling — that you would otherwise hand-tune in CUDA C. It is the *backend that `torch.compile` generates by default on NVIDIA GPUs*, so most users benefit from Triton without writing it. You reach for hand-written Triton when you have a specific fused op that the compiler doesn't produce well — it has become the standard way for ML researchers (not just CUDA specialists) to write custom high-performance kernels.

**FlashAttention as the canonical fusion case study.** Naive attention computes `softmax(QKᵀ / √d) V` by *materialising* the full `S × S` attention-score matrix in HBM (for sequence length `S`). At `S = 8192` that intermediate is `8192² ≈ 67M` entries *per head per layer* — tens of megabytes that get written to and read back from HBM purely as a scratchpad, making attention severely memory-bound and quadratic in memory. **FlashAttention** fuses the entire `QKᵀ → scale → softmax → ×V` chain into one kernel that *never writes the full score matrix to HBM*. It tiles `Q`, `K`, `V` into blocks that fit in on-chip SRAM, and uses the **online softmax** trick — computing the softmax incrementally, block by block, while rescaling the running output — so it only ever holds a small tile on chip. The payoff is large: HBM traffic for the scores drops from O(S²) to O(S), the activation memory of attention becomes *linear* in sequence length (which is what makes long contexts feasible at all), and wall-clock speedups of roughly 2–4× over a standard implementation are typical. It is worth internalising FlashAttention not as a magic library call but as the *archetype* of why fusion matters: same math, same FLOPs, dramatically less HBM traffic, dramatically faster. You will usually get it for free by enabling PyTorch's `scaled_dot_product_attention` (SDPA), which dispatches to a FlashAttention backend when shapes and dtypes permit.

### 3.6 Memory management: the allocator, OOM, and host transfers

#### The caching allocator and why OOM "sticks"

Calling the driver to allocate device memory (`cudaMalloc`) is *slow* and *synchronizing* — it would stall the CPU/GPU pipeline if done on every tensor. So PyTorch interposes a **caching allocator**: it grabs large slabs of memory from the driver up front and then services your tensor allocations and frees from its own pools. When a tensor is freed in Python, its memory is *not* returned to the driver — it is kept in PyTorch's cache, ready to satisfy the next allocation of a similar size almost instantly. This is why `nvidia-smi` often shows PyTorch "holding" far more memory than your live tensors need: that is *reserved* cache, not a leak.

This design has direct, practical consequences you must internalise:

- **Reserved vs allocated.** `torch.cuda.memory_allocated()` reports memory backing live tensors; `torch.cuda.memory_reserved()` reports the larger amount held in the cache. The gap is reusable headroom, not waste.
- **`torch.cuda.empty_cache()`** returns the *unused cached* slabs to the driver. It does *not* free live tensors and rarely fixes a genuine OOM (your live tensors are the problem); its real use is letting a *different process* on the same GPU grab that memory.

#### Why "CUDA out of memory" persists across cells

A frequent confusion, especially in notebooks: you hit OOM, you delete a tensor or catch the exception, and the *next* line still OOMs or `nvidia-smi` still shows high usage. Reasons, in order of likelihood:

1. **Python still holds references.** The tensor isn't actually freed until its last reference is dropped. The exception traceback itself holds references to local tensors; a lingering variable in a notebook cell keeps the whole computation graph alive. You must `del` the references (and sometimes `gc.collect()`) before the allocator can reclaim them.
2. **Autograd graph retention.** Keeping a `loss` tensor (or any output with `requires_grad=True`) alive keeps *every activation in its backward graph* alive — potentially gigabytes. Use `torch.no_grad()` (or `inference_mode()`) for inference, and don't accumulate `loss` itself across steps (`total += loss.item()`, never `total += loss`).
3. **The cache is reserved, not leaked.** High `nvidia-smi` usage after freeing is the caching allocator holding slabs — expected, and reused on the next allocation.

#### Fragmentation

Even with enough *total* free memory, an allocation can fail because the free memory is **fragmented** — broken into pieces none of which is individually large enough for a big contiguous request. The classic trigger is a workload with *varying* tensor sizes (variable sequence lengths!): the allocator's pools fill with mismatched holes. The error message is the tell — *"tried to allocate X; Y free; Z reserved"* with `Y` comfortably exceeding `X` yet the allocation still failing. The standard mitigation is the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which lets the allocator grow segments more flexibly and dramatically reduces fragmentation OOMs for variable-shape workloads — worth setting by default for LLM inference/serving. Padding sequences to fixed bucket lengths also helps by making allocations uniform.

#### Pinned (page-locked) host memory and async copies

Data starts life in **host (CPU) memory** and must be copied to **device (GPU) memory** over PCIe (or, on a DGX Spark's unified GB10 design, accessed through shared on-package memory rather than copied over PCIe at all). By default the OS may keep host memory in *pageable* pages it can swap; the GPU's DMA engine cannot copy directly from pageable memory, so the driver first stages it through an internal pinned buffer — an extra copy, and the transfer cannot overlap with compute.

**Pinned (page-locked) memory** — `tensor.pin_memory()`, or `pin_memory=True` in a PyTorch `DataLoader` — is host memory the OS guarantees never to swap. The DMA engine can copy it directly, faster, *and* the copy can be issued **asynchronously** (`tensor.to(device, non_blocking=True)`) so that the transfer overlaps with GPU compute on a separate copy engine. The standard input pipeline — pinned-memory DataLoader + `non_blocking=True` transfers — exists precisely to hide H2D (host-to-device) copy latency behind computation, so the GPU never waits for data. The same applies in reverse for D2H (device-to-host) when you pull results back; note that calling `.item()` or `.cpu()` on a result forces a synchronization and can silently serialise your loop if done mid-pipeline.

#### Unified memory

**Unified (managed) memory** (`cudaMallocManaged`) presents a *single address space* shared by CPU and GPU; the driver migrates pages on demand when either side touches them. It simplifies code (no explicit copies) and lets you *oversubscribe* — address more memory than the GPU physically has, with the driver paging to host — at the cost of page-fault and migration overhead that can be slow if access patterns thrash. On discrete GPUs it is more a convenience/large-model-fitting feature than a performance tool. It is, however, conceptually central to **tightly-coupled CPU+GPU systems like the DGX Spark (GB10)**, where the CPU and GPU share one physical pool of LPDDR5X memory (128 GB at ~273 GB/s) over a coherent NVLink-C2C link, making the "unified" abstraction cheap rather than a paging hack — a genuinely different memory model from the discrete-PCIe A100/H100 case you will also use. (Note that this ~273 GB/s shared bandwidth is far below an H100's multi-TB/s of dedicated HBM, so DGX Spark is positioned for local development and inference of large models, not peak training throughput.)

### 3.7 Eager vs graph compilers: JAX/XLA in one breath

PyTorch's default is **eager/imperative**: ops run as encountered, maximally debuggable (you can `print` any intermediate), at the cost of seeing only one op at a time and thus missing cross-op fusion unless you `torch.compile`. The alternative philosophy is **compile-first**, exemplified by **JAX**, whose `jax.jit` traces your function into a complete graph and hands it to **XLA** (Accelerated Linear Algebra), a whole-program compiler that performs aggressive fusion, layout optimisation, and scheduling over the *entire* computation before running anything. The trade-off is the recurring theme of this section: graph compilation buys fusion and the elimination of per-op Python/launch overhead (often a large win, especially for the small-op, launch-bound regime), at the cost of compilation latency, recompilation on shape changes, and harder debugging (you trace abstract shapes, not concrete values). `torch.compile` is, in effect, PyTorch importing the graph-compilation win into an eager-by-default framework — you stay imperative and debuggable by default, and opt into the compiled graph where it pays.

### 3.8 The compilation chain and version compatibility (the real-world gotcha)

Finally, the chain that turns kernel source into something the GPU runs, because its versioning is a genuine and recurring source of pain.

#### nvcc → PTX → SASS

CUDA C/C++ kernel source is compiled by **`nvcc`**, the NVIDIA CUDA compiler, in two stages:

1. **PTX (Parallel Thread Execution)** — a *virtual* assembly / intermediate representation that is **forward-compatible**: it targets a *virtual* GPU architecture (a `compute_XX` capability) rather than a specific chip. PTX from an older toolkit can be JIT-compiled to run on newer hardware.
2. **SASS** — the *actual* native machine code for a *specific* GPU architecture (an `sm_XX` target, e.g. `sm_80` for A100, `sm_90` for H100, `sm_121` for the GB10's Blackwell GPU). SASS is what the SMs really execute. (SASS is sometimes glossed as "Streaming ASSembler," but NVIDIA has never published an official expansion — treat it simply as the name of the native ISA.)

A built binary can embed SASS for several specific architectures plus a PTX fallback. At load time, if the GPU's architecture matches embedded SASS, it runs directly; if not, the driver **JIT-compiles the embedded PTX** to SASS for the present GPU on the fly (you may notice a one-time startup delay the first time). This is the mechanism that lets a single PyTorch wheel run on A100, H100, and newer cards: it ships SASS for the common ones and PTX for forward compatibility. The flip side: if a library shipped *neither* matching SASS *nor* usable PTX for your card, you get the dreaded *"no kernel image is available for execution on the device"* — the binary simply has nothing runnable for your `sm_XX`.

#### The version-compatibility matrix you will actually fight

Four version numbers must line up, and mismatches produce some of the most common and most confusing setup failures:

| Component | Example | Compatibility rule of thumb |
|---|---|---|
| **NVIDIA driver** | 550.xx or newer | Must be ≥ the CUDA runtime's requirement. *Newer driver runs older CUDA* (backward compatible), not vice-versa. |
| **CUDA toolkit / runtime** | 12.4 | The PyTorch wheel bundles its own CUDA runtime (the `cu124` in `torch==2.x+cu124`); it must be supported by the installed driver. |
| **GPU compute capability** | `sm_90` (H100) | The framework build must ship SASS or PTX for it. |
| **cuDNN / NCCL / framework** | cuDNN 9.x | Must match the CUDA major version the framework was built against. |

The single most useful facts to hold onto:

- **`nvidia-smi`** shows the **driver** version and the *maximum* CUDA version that driver supports — *not* the CUDA your PyTorch is using. The often-lower number from `torch.version.cuda` is the runtime your wheel actually bundles and uses. These two differing is **normal**, not a bug, and confuses nearly everyone the first time.
- Because PyTorch wheels **bundle their own CUDA runtime**, you usually do *not* need a matching system-wide CUDA toolkit installed to *run* PyTorch — only a sufficiently new **driver**. You only need the full system toolkit (with `nvcc`) when *compiling* custom CUDA/C++ extensions, and then *that* `nvcc`'s version should match your PyTorch's CUDA major version or the extension build will mismatch at link/load time.
- On a fresh cloud box (RunPod, etc.), the highest-leverage first commands are `nvidia-smi` (driver + card + current utilisation/memory) and `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"`. If `is_available()` is `False` despite a working `nvidia-smi`, you almost always have a **driver ↔ bundled-CUDA mismatch** — the wheel's CUDA runtime needs a newer driver than the box provides — which is fixed by installing a PyTorch build for an older CUDA, or updating the driver. Recognising that specific failure signature will save you hours.

---

## 4. Performance fundamentals: FLOPs, bandwidth, arithmetic intensity, and the roofline

This section is the quantitative heart of the document. Everything that follows about batch sizes, parallelism, and precision is ultimately a story about two physical resources — **how fast a GPU can do arithmetic** and **how fast it can move bytes** — and the tension between them. If you internalize the handful of formulas and worked examples below, you will be able to look at almost any deep-learning workload and predict, to within a small factor, whether it will be fast or slow, and *why*. That predictive power is what separates someone who merely *runs* models from someone who *understands* them.

We will build up in layers: first define the units precisely, then count the FLOPs of the operations that actually matter, then introduce memory bandwidth as the competing constraint, and finally unify the two into the **roofline model** and the utilization metrics (MFU/HFU) you will use to judge whether a run is healthy.

### 4.1 FLOP vs FLOP/s: getting the units right

A **FLOP** is a single **fl**oating-point **op**eration: one addition, one subtraction, one multiplication, or (depending on how you count) one division of two floating-point numbers. It is a *count* — a pure number, dimensionless. "This matrix multiply costs 4 billion FLOPs" is a statement about *how much arithmetic work* the computation contains, independent of how fast you do it.

A **FLOP/s** (often written FLOPS, FLOPs/s, or "flops" — the overloading is unfortunate) is a *rate*: floating-point operations **per second**. It is a property of *hardware running a workload*, not of the computation itself. An NVIDIA A100 can sustain on the order of hundreds of trillions of FLOP/s; the matrix multiply above contains a fixed number of FLOPs. Dividing the two gives you a time.

To keep these straight, this document uses the convention:

- **FLOP** (no "/s") = a count of operations. We write large counts as, e.g., $4 \times 10^9$ FLOP or "4 GFLOP".
- **FLOP/s** = a rate. Peak hardware rates are quoted in **TFLOP/s** (tera = $10^{12}$) or **PFLOP/s** (peta = $10^{15}$).

> **Why the prefixes are base-10 here.** Unlike memory (where "GB" is ambiguous between $10^9$ and $2^{30}$ bytes — see §4.4), FLOP counts and FLOP/s rates are essentially always quoted in **base-10 SI** prefixes. A "TFLOP/s" means $10^{12}$ floating-point operations per second, full stop. We will be pedantic about base-2 vs base-10 only for *bytes*.

#### 4.1.1 FMA: why a multiply-add counts as 2 FLOPs

The single most important counting convention to nail down is the **fused multiply-add (FMA)**, sometimes called MAC (multiply-accumulate). The operation

$$
d \leftarrow a \times b + c
$$

is *one hardware instruction* on essentially every modern FPU and GPU, but by universal convention it counts as **2 FLOPs**: one multiply and one add. This matters enormously because the workhorse of deep learning — the dot product, and hence the matrix multiply — is built entirely out of FMAs.

This is also the root of a common confusion. Hardware vendors quote peak FLOP/s by counting each FMA as 2 FLOPs (because the chip really does retire a multiply *and* an add every cycle per FMA unit). So when a spec sheet says "X TFLOP/s", the X already has the factor of 2 baked in. As long as you *also* count your workload's FMAs as 2 FLOPs (which the formulas below do), the units are consistent and the division gives the right time. The danger is mixing conventions — counting your workload's FMAs as 1 but dividing by a peak that assumed 2, which makes everything look 2× slower than it is.

#### 4.1.2 Peak vs achieved, and the marketing-number trap

A GPU's **peak FLOP/s** is a *theoretical ceiling*: (number of arithmetic units) × (FLOPs per unit per cycle) × (clock frequency). You essentially never reach it. **Achieved FLOP/s** is what your actual kernel delivers, and the ratio (achieved / peak) is the *efficiency* of that kernel — a number we will formalize as MFU/HFU in §4.8.

Worse, the headline number on a spec sheet is frequently the *most favorable* one the marketing department could find:

| Caveat | What it means | Typical inflation |
|---|---|---|
| **Tensor-core vs CUDA-core** | The huge numbers (e.g. "312 TFLOP/s" dense FP16/BF16 on A100) are for the dedicated matrix-multiply units (tensor cores) at *reduced precision*, not general FP32 arithmetic. | Tensor FP16/BF16 (~312 TFLOP/s) is ~16× the A100's FP32 *CUDA-core* rate (~19.5 TFLOP/s). |
| **Sparsity** | NVIDIA often quotes a "with sparsity" number that *doubles* the dense rate, achievable only on specially-pruned 2:4-sparse weights. (This is why A100 is sometimes listed as "624 TFLOP/s FP16" — that is the *sparse* figure; the dense number is 312.) | 2× |
| **Precision** | Lower precision = more FLOP/s. FP8 ≈ 2× FP16 ≈ ... The headline is usually the lowest precision the chip supports. | 2× per precision step |
| **Boost clock** | Peak assumes the maximum boost frequency, which thermal/power limits may prevent sustaining. | ~5–20% |

**Rule of thumb for sanity:** when someone quotes a GPU's TFLOP/s, immediately ask three questions — *which precision? tensor cores or CUDA cores? dense or sparse?* For training/inference of LLMs you care about the **dense, tensor-core, BF16/FP16 (or FP8)** number, because the matrix multiplies that dominate the workload run on tensor cores. Representative **dense** tensor-core figures (vendor- and clock-dependent; treat as ±10%):

| GPU | Dense BF16/FP16 tensor (TFLOP/s) | FP8 tensor (TFLOP/s, dense) | HBM bandwidth (GB/s, base-10) |
|---|---|---|---|
| A100 80GB (SXM) | ~312 | — (no FP8) | ~2,039 |
| H100 (SXM) | ~990 | ~1,979 | ~3,350 |
| H200 (SXM) | ~990 | ~1,979 | ~4,800 |
| RTX 4090 | ~165 | ~330 | ~1,008 |
| DGX Spark (GB10) | ~125 (BF16 est.; see caveat) | vendor-quoted "1 PFLOP" is FP4 *with sparsity* | ~273 (LPDDR5X) |

> **A note on the RTX 4090 numbers.** The 4090's *dense* FP16/BF16 tensor rate is ~165 TFLOP/s; the often-cited "330 TFLOP/s" is either the FP16 *sparse* figure or the *dense FP8* figure. Its FP8 dense is ~330 (~660 with sparsity). As always, dense-FP16 is the apples-to-apples training number.

> **DGX Spark caveat.** The widely-quoted "1 petaFLOP" for the Spark/GB10 is an **FP4 (NVFP4), with-sparsity** number; the *dense* NVFP4 figure is roughly half that (~500 TFLOP/s), and the realistic **BF16 dense** rate is far lower still (a low-hundreds-of-TFLOP/s figure — we list ~125 as an order-of-magnitude estimate; NVIDIA does not headline a dense BF16 number, so treat this as approximate). Crucially for the inference work in this project, its memory bandwidth (~273 GB/s of LPDDR5X unified memory) is **roughly an order of magnitude below an H100's HBM3** (~3,350 GB/s). Since LLM *decoding* is memory-bound (§4.6–4.7), the Spark will feel much slower than its headline FLOP number suggests for single-stream generation, even though it can *hold* large models thanks to its 128 GB of unified memory. This is the single most important practical consequence of the roofline model for your hardware mix.

### 4.2 The FLOP cost of a matrix multiply (derived)

Almost all the arithmetic in a transformer lives in matrix multiplications (GEMMs — **GE**neral **M**atrix **M**ultiply). So we must be able to count a GEMM exactly.

Consider $C = A \, B$ where $A$ is $M \times K$, $B$ is $K \times N$, and the result $C$ is $M \times N$.

Each output element $C_{ij}$ is a dot product of a length-$K$ row of $A$ with a length-$K$ column of $B$:

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}.
$$

That sum contains $K$ multiplications and $K-1$ additions, i.e. $\approx 2K$ FLOPs (we count the FMA convention: $K$ multiply-adds = $2K$ FLOPs; the off-by-one of "$K-1$ additions" is negligible for large $K$). There are $M \times N$ output elements, so:

$$
\boxed{\text{FLOPs}(C = AB) = 2 \, M \, N \, K.}
$$

**Worked example.** Multiply a $4096 \times 4096$ matrix by a $4096 \times 4096$ matrix ($M=N=K=4096$):

$$
2 \times 4096^3 = 2 \times 6.87 \times 10^{10} = 1.37 \times 10^{11}\ \text{FLOP} = 137\ \text{GFLOP}.
$$

On an A100 at 312 TFLOP/s peak BF16, the *theoretical* floor is

$$
\frac{1.37 \times 10^{11}\ \text{FLOP}}{3.12 \times 10^{14}\ \text{FLOP/s}} = 4.4 \times 10^{-4}\ \text{s} = 0.44\ \text{ms}.
$$

A well-tuned cuBLAS kernel on a problem this large really does get within ~70–90% of peak, so you'd measure perhaps 0.5–0.6 ms. We will see in §4.5 *why* this particular shape is fast (it's compute-bound) while many other operations are not.

### 4.3 The transformer FLOP rules: ~6N per token (training), ~2N per token (inference)

Now the rule every LLM practitioner should have memorized. Let $N$ be the number of parameters in a (dense, decoder-only) transformer — e.g. $N = 1.5 \times 10^9$ for R1-Distill-1.5B, $N = 7 \times 10^9$ for a 7B. Then, ignoring attention's quadratic term (justified below) and other lower-order costs:

- **Inference (forward pass):** $\approx 2N$ FLOPs per token.
- **Training (forward + backward):** $\approx 6N$ FLOPs per token.

#### 4.3.1 Deriving the forward "2N"

Almost every parameter in a transformer is a weight in some matrix (the attention projections $W_Q, W_K, W_V, W_O$ and the MLP matrices, plus the embedding/unembedding). When you push **one token's** activation vector through a weight matrix $W$ of shape $d_\text{in} \times d_\text{out}$, you compute a matrix–vector product. By the GEMM rule with $M=1$, $K = d_\text{in}$, $N = d_\text{out}$:

$$
\text{FLOPs} = 2 \times 1 \times d_\text{in} \times d_\text{out} = 2 \times (\text{number of weights in } W).
$$

So **each weight is touched in exactly one multiply-add per token = 2 FLOPs.** Summing over every weight matrix in the model, the forward pass costs $\approx 2N$ FLOPs per token. That's the whole derivation — it is just "every parameter participates in one FMA per token."

For a **sequence of $T$ tokens** processed together (e.g. prefill, or a training batch), the matrix-multiply weight cost is $\approx 2 N T$, because the same weights are reused for all $T$ tokens (now $M=T$ instead of $M=1$). This reuse is exactly what makes prefill compute-bound and decode memory-bound — the central theme of §4.7.

#### 4.3.2 Deriving the backward "4N" (and hence 6N total)

Training adds a backward pass. For each weight matrix, backpropagation must compute **two** matrix products of the same size as the forward one:

1. the **gradient w.r.t. the input** (to keep propagating the error to earlier layers), and
2. the **gradient w.r.t. the weights** (what the optimizer actually uses).

Each of these costs about as much as the forward matmul ($\approx 2N$ per token each), so the backward pass is $\approx 4N$ per token, and

$$
\text{forward} + \text{backward} \approx 2N + 4N = 6N \ \text{FLOPs per token}.
$$

This "$6N$" is the famous coefficient behind the Kaplan/Chinchilla compute estimate $C \approx 6 N D$, where $D$ is the total number of training tokens. (Gradient *checkpointing*, which recomputes activations to save memory, adds an extra forward pass and pushes the effective cost toward $\approx 8N$ per token — see HFU vs MFU in §4.8.)

#### 4.3.3 Why we can usually ignore attention's quadratic term

The self-attention score computation ($QK^\top$ and the subsequent $\text{softmax}\cdot V$) costs roughly $4 \, T^2 \, d_\text{model}$ FLOPs per layer for a sequence of length $T$ (about $2T^2 d_\text{model}$ for the scores and another $2T^2 d_\text{model}$ for the value aggregation) — note it scales with $T^2$, not with parameters, and this term is *separate* from the $QKV$/output projections, whose weights are already counted in the $2N$ term. The "$2N$" rule deliberately omits this quadratic piece. The omission is fine **when the sequence is short relative to the model's hidden width**, specifically when $T \lesssim d_\text{model}$. For a 1.5B-class model $d_\text{model}$ is ~1536–2048; for a 7B it's ~4096. So at the few-thousand-token CoT lengths typical in this project, attention is a modest correction — and at very long contexts (tens of thousands of tokens) the $T^2$ term takes over and the $2N$ rule undercounts. Flag that mentally whenever $T$ approaches or exceeds $d_\text{model}$.

#### 4.3.4 Worked examples: 1.5B and 7B

**Inference, one forward token, R1-Distill-1.5B ($N = 1.5\times10^9$):**

$$
2N = 2 \times 1.5\times10^9 = 3.0\times10^9\ \text{FLOP} = 3.0\ \text{GFLOP / token}.
$$

Generating a 4,000-token chain-of-thought (decode only, ignoring prefill) is

$$
4000 \times 3.0\times10^9 = 1.2\times10^{13}\ \text{FLOP} = 12\ \text{TFLOP}.
$$

On an A100's 312 TFLOP/s *peak*, that is a compute floor of $12 / 312 \approx 0.038$ s. **But you will never see anything close to this in single-stream decoding** — because decode is memory-bound, the real time is set by bandwidth (§4.7), and you'll measure more like 30–60 s for 4,000 tokens (≈ tens of ms/token). The gap between the 38 ms compute-floor and the tens-of-seconds reality *is* the roofline story.

**Inference, one forward token, 7B ($N = 7\times10^9$):**

$$
2N = 1.4\times10^{10}\ \text{FLOP} = 14\ \text{GFLOP / token}.
$$

**Training, 7B, one token:** $6N = 4.2\times10^{10}$ FLOP = 42 GFLOP/token. Training on $D = 1\times10^{12}$ tokens (1T) would cost

$$
6 N D = 6 \times 7\times10^9 \times 1\times10^{12} = 4.2\times10^{22}\ \text{FLOP}.
$$

At a *realistic* sustained 150 TFLOP/s ($\approx$ 48% MFU on A100-class), that's

$$
\frac{4.2\times10^{22}}{1.5\times10^{14}} = 2.8\times10^{8}\ \text{s} \approx 8.9\ \text{years on a single GPU},
$$

which is exactly why pretraining needs thousands of GPUs in parallel (Section on parallelism). For the *fine-tuning* scales contemplated in this project (LoRA on $\sim10^6$–$10^7$ tokens), the same arithmetic gives single-GPU times of minutes to hours — entirely tractable.

### 4.4 Memory bandwidth: the other wall

FLOPs are only half the story. Before the arithmetic units can multiply two numbers, those numbers must be *delivered* from memory to the compute units. **Memory bandwidth** is the rate at which bytes can be moved between the GPU's main memory (HBM / VRAM) and the on-chip compute units, quoted in **GB/s** or **TB/s** (base-10 for bandwidth, by near-universal convention).

#### 4.4.1 A note on bytes: GB vs GiB

For *capacity and byte counts* we must be careful:

- **GB (gigabyte, base-10)** = $10^9$ bytes.
- **GiB (gibibyte, base-2)** = $2^{30} = 1.073\times10^9$ bytes (≈ 7.4% larger).

GPU **bandwidth** specs are base-10 GB/s. GPU **capacity** ("80GB A100") is usually quoted base-10 too, but allocator/driver overhead means you can address slightly less. PyTorch reports memory in base-2 (GiB/MiB) under the hood. We will write GB for base-10 and GiB for base-2 explicitly whenever the distinction matters; a good habit is to keep a ~7% mental margin and never plan to use 100% of quoted VRAM.

#### 4.4.2 Bytes per number, by precision

How many bytes a tensor occupies depends entirely on its dtype:

| dtype | bytes/element | typical use |
|---|---|---|
| FP32 | 4 | master weights, optimizer state, "full precision" |
| TF32 | 4 (stored), reduced mantissa in compute | A100+ default matmul mode |
| BF16 / FP16 | 2 | training compute, inference weights |
| FP8 (E4M3/E5M2) | 1 | H100+ inference/training |
| INT8 | 1 | quantized inference |
| INT4 / FP4 | 0.5 | aggressive quantized inference |

**Worked weight-size example.** A 7B model's weights:

- in BF16: $7\times10^9 \times 2 = 1.4\times10^{10}$ bytes $= 14$ GB (base-10) $= 13.0$ GiB.
- in FP32: $7\times10^9 \times 4 = 28$ GB.
- in INT4: $7\times10^9 \times 0.5 = 3.5$ GB.

For R1-Distill-1.5B in BF16: $1.5\times10^9 \times 2 = 3.0$ GB. These weight footprints are why the dtype you choose directly determines what fits in a given card, and — as we'll see — how fast decode runs.

#### 4.4.3 Worked example: a memory-bound elementwise op

Consider a trivial elementwise operation — say, a GeLU or a scalar multiply — applied to a BF16 tensor of $n = 10^8$ elements (200 MB). It reads each element once and writes it once: $2 \times 2$ bytes $= 4$ bytes of memory traffic per element, but only ~1–2 FLOPs of arithmetic per element.

- **Bytes moved:** $10^8 \times 4 = 4\times10^8$ bytes = 0.4 GB.
- **FLOPs:** $\sim 1$–$2 \times 10^8$ FLOP.

On an A100 (2,039 GB/s, 312 TFLOP/s):

- **Time if memory-bound:** $0.4\times10^9 / 2.039\times10^{12} \approx 196\ \mu s$.
- **Time if compute-bound:** $2\times10^8 / 3.12\times10^{14} \approx 0.6\ \mu s$.

The memory time is ~**300× larger**, so this kernel spends 99.7% of its life waiting for memory and the arithmetic units sit nearly idle. *No amount of extra FLOP/s would speed it up* — only more bandwidth, or fusing the op with neighbors so the data is read/written fewer times (kernel fusion, §4.6, §4.9). This is the prototypical **memory-bound** kernel, and it explains why activation functions, layernorms, residual adds, and dropout are essentially "free of compute but expensive of bandwidth."

### 4.5 Arithmetic intensity and the roofline model

We now unify compute and bandwidth into a single picture. The key quantity is **arithmetic intensity** (AI), also called **operational intensity**:

$$
\boxed{\text{AI} = \frac{\text{FLOPs performed}}{\text{bytes moved to/from memory}}\quad\left[\frac{\text{FLOP}}{\text{byte}}\right].}
$$

It measures how much arithmetic you extract per byte you fetch — i.e., how well you *reuse* data once it's on chip. High AI = lots of compute per byte = you can keep the arithmetic units busy. Low AI = you starve the arithmetic units waiting on memory.

#### 4.5.1 The roofline derivation and the ridge point

A kernel's runtime is bounded below by *both* limits simultaneously, and the binding one is whichever is slower:

$$
T \ge \max\!\left(\underbrace{\frac{\text{FLOPs}}{\pi}}_{\text{compute time}},\ \underbrace{\frac{\text{bytes}}{\beta}}_{\text{memory time}}\right),
$$

where $\pi$ = peak compute (FLOP/s) and $\beta$ = peak bandwidth (byte/s). Dividing achievable throughput by these and substituting AI = FLOPs/bytes gives the **roofline**: the maximum attainable performance $P$ (in FLOP/s) for a kernel of intensity AI is

$$
\boxed{P_\text{attainable}(\text{AI}) = \min\big(\pi,\ \ \beta \times \text{AI}\big).}
$$

Read this as two straight lines on a log–log plot of *performance (y, FLOP/s)* vs *arithmetic intensity (x, FLOP/byte)*:

- A **sloped roof** $P = \beta \cdot \text{AI}$ on the left (the *memory-bound* region): performance rises linearly with intensity, with slope equal to the memory bandwidth $\beta$. If you're under this roof, you're limited by bandwidth.
- A **flat roof** $P = \pi$ on the right (the *compute-bound* region): once you have enough intensity, you saturate the arithmetic units and adding more intensity buys nothing.

The two roofs meet at the **ridge point** (a.k.a. the *machine balance*):

$$
\boxed{\text{AI}^\star = \frac{\pi}{\beta}\quad\left[\frac{\text{FLOP}}{\text{byte}}\right].}
$$

A kernel with $\text{AI} < \text{AI}^\star$ is **memory-bound** (left of the ridge); one with $\text{AI} > \text{AI}^\star$ is **compute-bound** (right of the ridge). The ridge point is the single number that tells you "how much arithmetic per byte this machine demands before compute becomes the bottleneck."

**Worked ridge points** (dense BF16 tensor peak ÷ HBM bandwidth):

| GPU | $\pi$ (TFLOP/s) | $\beta$ (GB/s) | $\text{AI}^\star = \pi/\beta$ (FLOP/byte) |
|---|---|---|---|
| A100 80GB | 312 | 2,039 | **≈ 153** |
| H100 SXM | 990 | 3,350 | **≈ 296** |
| H200 SXM | 990 | 4,800 | **≈ 206** |
| RTX 4090 | 165 | 1,008 | **≈ 164** |
| DGX Spark (GB10) | ~125 (BF16 est.) | 273 | **≈ 450+** |

Two striking lessons jump out. First, the ridge points are *large* — you need **~150–300 FLOPs of work per byte fetched** just to keep an A100/H100 compute-bound. Second, the **DGX Spark's ridge point is enormous** because its bandwidth is so low relative to its compute (and note the H200, despite matching the H100's compute, has a *lower* ridge point than the H100 precisely because its bandwidth is higher — exactly what makes it better for memory-bound inference). Almost everything memory-touching will be memory-bound on the Spark. That is the roofline making the Spark's character precise.

#### 4.5.2 Classifying the operations you actually run

Now we can classify the workload. Compute the AI of each operation and compare to $\text{AI}^\star \approx 150$–$300$.

**Large square GEMM (compute-bound).** For $M=N=K=n$ in BF16: FLOPs $=2n^3$; bytes (read $A$, $B$, write $C$, ignoring reuse) $= 3 n^2 \times 2$. So

$$
\text{AI} = \frac{2 n^3}{6 n^2} = \frac{n}{3}.
$$

For $n = 4096$, AI $\approx 1365$ FLOP/byte — *far* to the right of any ridge point above. This is **why big GEMMs are compute-bound**: intensity grows linearly with the dimension, so for large matrices you do enormous arithmetic per byte and saturate the tensor cores. (Caveat: this counts each input read once; the *real* on-chip reuse is achieved by tiling — §4.6.)

**LLM decode — the matrix–vector product (memory-bound).** During single-stream generation you process **one token at a time** ($M=1$). Each weight matrix is read from HBM (2 bytes/weight in BF16) and used for exactly **2 FLOPs** per weight (one FMA). So:

$$
\text{AI}_\text{decode} = \frac{2 \times (\text{\#weights})}{2\ \text{bytes} \times (\text{\#weights})} = \frac{2}{2} = 1\ \text{FLOP/byte}.
$$

An arithmetic intensity of **~1**, against a ridge point of ~150–300, means decode is **catastrophically memory-bound** — it runs at well under 1% of peak FLOP/s. The bottleneck is reading the *entire weight matrix from HBM for every single token*. This single fact explains the bulk of LLM inference performance:

> **Single-stream decode time per token $\approx$ (model bytes) / (memory bandwidth).** For a 1.5B BF16 model (3.0 GB) on an A100 (2,039 GB/s): $3.0\times10^9 / 2.039\times10^{12} \approx 1.5$ ms/token *just to stream the weights* — a hard floor, before any overhead. On the DGX Spark (273 GB/s): $3.0\times10^9/273\times10^9 \approx 11$ ms/token. The ~7× ratio is exactly the bandwidth ratio ($2039/273 \approx 7.5$), and it is why the Spark feels slow at generation despite its big memory.

This is also the deep reason **batching helps decode** (§4.10) and why **quantization** (fewer bytes/weight) directly speeds up decode: at AI ≈ 1, halving the bytes nearly halves the time.

**Attention during decode (memory-bound, KV-cache traffic).** Computing attention for the newest token requires reading the entire **KV cache** (the stored keys and values for all previous tokens) from HBM, while doing only $O(T \cdot d)$ FLOPs. The KV cache is read once per generated token and reused for very little arithmetic, so attention-decode is also firmly memory-bound; its cost grows with sequence length because the KV cache grows. This is why long contexts slow generation and why FlashAttention (which avoids materializing the $T\times T$ score matrix in HBM) is a bandwidth optimization, not primarily a FLOP one.

**Elementwise ops (memory-bound).** As shown in §4.4.3, AI ≈ 0.25–0.5 FLOP/byte. Always memory-bound; the remedy is fusion.

**Prefill / training step (compute-bound).** Processing a long prompt or a training batch passes $T$ tokens (or batch×seq) through each weight matrix *at once* ($M = T \gg 1$), so each weight fetched from HBM does $2T$ FLOPs instead of 2. AI $\approx T$ (up to the ridge), pushing the operation across the ridge into compute-bound territory once $T \gtrsim \text{AI}^\star$. **This is the single most important practical lever:** the *same* weights, the *same* model, are memory-bound at batch-1 decode and compute-bound at batch-$T$ prefill — purely because of reuse.

#### 4.5.3 A textual picture of the roofline plot

Imagine log–log axes: x = arithmetic intensity (FLOP/byte), y = attainable performance (FLOP/s).

```
 performance (FLOP/s, log)
   π  ┤                 ______________________  <- flat "compute roof" (peak FLOP/s)
      │                /
      │               /  <- compute-bound region (right of ridge)
      │              /
      │             ◆ ridge point AI* = π/β
      │            /
      │   slope=β /
      │          /  <- memory-bound region (left of ridge)
      │         /
      │   ·decode(AI≈1)      ·elementwise(AI≈0.3)        ·big-GEMM(AI≈1300) ▲(up on flat roof)
      └────────────────────────────────────────────────► arithmetic intensity (FLOP/byte, log)
                AI*≈150–300
```

Each kernel is a point at its AI; its *attainable* performance is where a vertical line from that point hits the roof. Decode and elementwise sit far left under the sloped roof (memory-bound, low attainable performance); large GEMMs sit far right under the flat roof (compute-bound, near peak). Your job as an optimizer is to *move kernels rightward* (raise AI via batching, fusion, tiling) until they reach the flat roof — past that, only more FLOP/s helps.

### 4.6 Cache reuse, data locality, and tiling: why a naive matmul is slow

The roofline's "bytes moved" term is not fixed by the math — it depends on *how cleverly you reuse data once it's on chip*. This is where Section 2's memory hierarchy (registers → L1/shared memory → L2 → HBM) becomes performance-critical.

#### 4.6.1 The naive matmul reads memory far too many times

Recall the matrix multiply $C = AB$ with $M=N=K=n$. The naive triple loop

```
for i in 0..n:        # row of C
  for j in 0..n:      # col of C
    acc = 0
    for k in 0..n:
      acc += A[i,k] * B[k,j]
    C[i,j] = acc
```

does the correct $2n^3$ FLOPs, but look at the memory traffic. To compute each of the $n^2$ output elements it streams an entire row of $A$ and column of $B$ ($2n$ elements). If nothing is cached, that's $2n^3$ element-reads — meaning **each element of $A$ and $B$ is re-read from slow memory $\sim n$ times.** The effective arithmetic intensity collapses toward $O(1)$ and the kernel runs at memory-bound speeds even though the *math* is compute-bound. The arithmetic is there; we're just feeding it through a straw.

#### 4.6.2 Tiling/blocking restores intensity

**Tiling** (a.k.a. **blocking**) fixes this by computing the output in small **tiles** that fit in fast on-chip memory (shared memory / registers). Partition $C$ into $b\times b$ tiles. To compute one output tile you load a $b\times K$ strip of $A$ and a $K\times b$ strip of $B$, but — crucially — once a $b\times b$ block of $A$ and of $B$ are resident in shared memory, you perform $2b^3$ FLOPs on them before evicting. Each loaded element is now reused $\sim b$ times instead of once. The HBM traffic drops by a factor of $\sim b$, and the arithmetic intensity rises from $O(1)$ toward

$$
\text{AI}_\text{tiled} \approx \frac{2 n^3}{(2 n^3 / b)\ \text{bytes-ish}} \sim O(b)\ \text{FLOP/byte},
$$

i.e. proportional to the tile size $b$. Choose $b$ large enough that AI crosses the ridge point and the GEMM becomes genuinely compute-bound. This is precisely what cuBLAS, CUTLASS, and tensor-core kernels do, with multiple levels of tiling matched to the hierarchy:

| Level of tiling | Lives in | Reuse it captures |
|---|---|---|
| Thread-block / "CTA" tile | Shared memory (L1) | Block of A,B reused across the block's threads |
| Warp tile | Registers | Reuse within a warp's output sub-tile |
| Thread/register tile | Registers | Each thread computes a small patch, accumulating in registers |
| (across blocks) | L2 cache | Strips of A,B reused by neighboring output tiles |

The same idea — *load once into fast memory, do as much arithmetic as possible before evicting* — is the universal recipe for turning a memory-bound layout into a compute-bound one. (On NVIDIA GPUs the unit of thread scheduling is the **warp** of 32 threads, and a thread block is partitioned into warps; the "warp tile" above is the slice of the output a single 32-thread warp accumulates.) **FlashAttention** is exactly this insight applied to attention: tile the $Q,K,V$ matrices so the $T\times T$ score block is formed, softmaxed, and consumed entirely in fast SRAM, never written to HBM. It doesn't reduce FLOPs; it slashes *bytes moved*, raising AI and moving attention rightward on the roofline.

> **Takeaway.** "Bytes moved" in the roofline is a property of your *kernel's memory schedule*, not just the problem. Good kernels (tiled GEMM, FlashAttention, fused elementwise) maximize on-chip reuse to raise AI. You almost never write these yourself — you get them from cuBLAS/CUTLASS/cuDNN, PyTorch's fused kernels, FlashAttention, or `torch.compile` — but you must recognize *when* a workload is leaving performance on the table because reuse is low (small batches, unfused ops, tiny matmuls).

### 4.7 Putting it together: why prefill is fast and decode is slow

We can now state the central performance fact of LLM inference crisply, because it falls straight out of arithmetic intensity.

| Phase | Shape ($M$) | What's reused | AI | Bound | Speed character |
|---|---|---|---|---|---|
| **Prefill** (process the prompt) | $M = T_\text{prompt}$ tokens at once | each weight reused across $T$ tokens | $\sim T$ (large) | **compute** | fast; near tensor-core peak; cost $\approx 2NT$ |
| **Decode** (generate token-by-token) | $M = 1$ (or = batch $B$) | weight reused across only 1 (or $B$) tokens | $\sim 1$ (or $B$) | **memory** | slow; floor = (model bytes)/bandwidth per token |

The *same matrices and the same FLOP-per-token formula* govern both phases. The only thing that changes is **how many tokens share each weight fetch** — i.e., arithmetic intensity. Prefill amortizes each weight read over the whole prompt; decode pays a full weight-streaming pass for every single token. That is the complete explanation for why "time to first token" (prefill, compute-bound) and "time per output token" (decode, memory-bound) have such different cost structures, and why throughput-oriented serving systems work so hard to *batch* many decode requests together (raising $M$ from 1 toward $B$, §4.10).

For this project's concrete numbers — generating ~4,000-token CoT chains from R1-Distill-1.5B on a RunPod A100 — the binding constraint is decode bandwidth: expect a hard floor of ~1.5 ms/token (weights only) plus KV-cache and overhead, landing in the low-tens-of-ms/token range, i.e. a minute or two per chain at batch 1. Batching multiple chains (or using vLLM's continuous batching) is the primary lever to push that toward compute-bound efficiency.

### 4.8 MFU and HFU: measuring how well you're using the hardware

To judge whether a run is healthy you need a single normalized efficiency number. There are two standard ones.

#### 4.8.1 MFU — Model FLOPs Utilization

**MFU** is the fraction of the GPU's peak FLOP/s that your run actually devotes to the *useful* model arithmetic (the $6N$-per-token training FLOPs, or $2N$ for inference), **excluding** any recomputation:

$$
\boxed{\text{MFU} = \frac{\text{model FLOPs per second actually achieved}}{\text{peak FLOP/s of the hardware}} = \frac{(6N)\times(\text{tokens/s})}{\pi}\quad(\text{training}).}
$$

**Worked MFU example (training).** Suppose you fine-tune the 7B ($N=7\times10^9$, so $6N = 4.2\times10^{10}$ FLOP/token) and measure a throughput of **2,500 tokens/s** on one A100 ($\pi = 312$ TFLOP/s). Then

$$
\text{achieved} = 4.2\times10^{10} \times 2500 = 1.05\times10^{14}\ \text{FLOP/s} = 105\ \text{TFLOP/s},
$$
$$
\text{MFU} = \frac{105}{312} = 0.337 \approx \textbf{34\%}.
$$

**What's a "good" MFU?** For large-model training on well-tuned stacks, **35–55% MFU** is typical and considered healthy; the original PaLM work reported ~46%, and many production runs sit in the 40s. Below ~30% suggests a bottleneck worth chasing (too-small batch, communication overhead, data loading, unfused ops, bad parallelism mapping). For *inference*, MFU is usually defined with $2N$ and is much lower in latency-bound single-stream decode (often **a few percent**, because decode is memory-bound — §4.7) but can reach tens of percent in heavily-batched throughput serving.

**You never hit 100%, for structural reasons:**
- **Memory-bound phases** (every elementwise op, layernorm, attention's softmax, and *all* of decode) run far below peak FLOP/s by definition — they're bandwidth-limited, so they drag the FLOP-average down.
- **Non-matmul FLOPs** (softmax exp, normalization, activation functions) use the general CUDA cores, not the tensor cores whose enormous rate defines $\pi$.
- **Communication** (all-reduce of gradients, all-gather of weights across GPUs — §parallelism) stalls compute.
- **Pipeline bubbles, kernel launch overhead, warm-up, suboptimal tile shapes** at the boundaries.
- **Data loading / host–device transfer** stalls if the input pipeline can't keep the GPU fed.

#### 4.8.2 HFU — Hardware FLOPs Utilization

**HFU** counts *all* FLOPs the hardware actually executed, **including** redundant recomputation — most importantly the extra forward pass introduced by **gradient (activation) checkpointing**. Checkpointing trades compute for memory: it discards activations on the forward pass and recomputes them during backward, adding roughly one extra forward ($\approx 2N$/token) and pushing total work from $6N$ toward $\approx 8N$/token.

- **MFU** uses the *ideal* model FLOPs ($6N$) — it measures *useful work delivered*.
- **HFU** uses the *actually-executed* FLOPs ($8N$ with checkpointing) — it measures *raw hardware busy-ness*.

Hence **HFU ≥ MFU always**, and the gap is the recomputation overhead. A run can show high HFU (the GPU is genuinely busy) but lower MFU (some of that busy-ness is recompute, not progress). When you read a number, check which is meant: MFU is the honest "how fast am I really training?" metric; HFU is "how hard is the silicon working?". For *judging model-training efficiency*, prefer MFU; for *judging kernel/hardware efficiency in isolation*, HFU is informative.

> **Sanity check you can run.** Measure tokens/s, multiply by $6N$ (training) or $2N$ (inference), divide by your GPU's *dense BF16 tensor* peak. If you get >60% something is probably mis-counted (wrong $N$, wrong peak, counting sparsity); if you get <10% on a training run, you're likely memory- or communication-bound and should profile.

### 4.9 Batch size and arithmetic intensity (bridge to the batch-size section)

We close with the lever that ties this whole section to the next one. Almost every technique for going faster on a GPU is, underneath, a technique for **raising arithmetic intensity so the kernel moves rightward on the roofline toward the compute roof.** Batch size is the biggest such lever.

Recall decode at batch 1 has AI ≈ 1 because each weight fetched from HBM is used for just 2 FLOPs. Now process **$B$ tokens together** (a batch of $B$ independent sequences, all decoding their next token). The weight matrix is fetched from HBM **once** but now multiplies a $d_\text{in}\times B$ block of activations, doing $2 B$ FLOPs per weight instead of 2. The bytes moved (the weights) are unchanged; the FLOPs scale with $B$:

$$
\text{AI}_\text{decode}(B) \approx B\ \text{FLOP/byte}.
$$

So batching is *the* mechanism that walks a memory-bound matmul up the sloped roof:

- At $B=1$: AI ≈ 1 → deeply memory-bound → ~1% MFU, time set by weight streaming.
- As $B$ grows: AI ≈ $B$ rises linearly; once $B \gtrsim \text{AI}^\star$ (≈150 on A100, ≈300 on H100) the operation **crosses the ridge** and becomes compute-bound. Beyond that, you're at the flat roof: throughput (tokens/s) saturates near peak and stops improving with $B$ (you're now FLOP-limited, and further $B$ only adds latency and memory). (In practice the KV cache, not the weight-matmul, becomes the dominant memory traffic well before this idealized $B$, so real serving systems hit the compute roof at somewhat different batch sizes — but the mechanism is exactly this.)

This yields the characteristic LLM-serving throughput curve: **tokens/s rises roughly linearly with batch size in the memory-bound regime, then plateaus** once you hit the compute roof. The same logic explains why larger training batches (up to a point) improve MFU — they amortize each weight fetch over more tokens. The countervailing costs (latency per request, and the memory to *hold* a large batch — activations and KV cache grow with $B$) are exactly what the batch-size section takes up next.

The unifying mental model to carry forward: **a GPU is a machine with a fixed FLOP rate and a fixed byte rate; performance is whichever runs out first; and almost all optimization is the art of doing more FLOPs per byte you move.** Batch size, tiling, fusion, quantization, and FlashAttention are all instances of that one idea.

### 4.10 How to actually measure FLOPs and intensity

Finally, how do you *get* these numbers for a real workload rather than deriving them by hand?

**Analytical counting (first resort — do this always).** For transformers the $2N$ / $6N$ rules and the $2MNK$ GEMM rule let you estimate FLOPs from architecture alone, with no run required. This is how you compute a *target* (the compute floor) to compare a measurement against. Libraries that automate it: `fvcore.nn.FlopCountAnalysis`, `ptflops`, `thop`, `calflops`, `torch.utils.flop_counter.FlopCounterMode` (PyTorch's built-in, dispatch-based and quite accurate for matmuls/attention). These hook the operations and tally $2MNK$ per GEMM for you.

**Empirical profiling (to find where time actually goes).**
- **`torch.profiler`** (with `record_shapes=True` and the `kineto`/CUPTI backend) gives per-kernel time, and recent versions can report achieved FLOP/s and even roofline-style analysis. It's the first-line tool to see *which kernels dominate wall-clock*.
- **NVIDIA Nsight Compute (`ncu`)** is the authoritative per-kernel profiler: it reports **achieved occupancy, achieved FLOP/s, DRAM throughput (bytes/s), and arithmetic intensity**, and will even draw the *measured roofline* placing each kernel as a point — exactly the plot of §4.5.3. Use it when you need to know *why* a specific kernel is slow (is it memory-bound? low occupancy?).
- **NVIDIA Nsight Systems (`nsys`)** gives the *timeline* view: kernel overlap, gaps (GPU idle = you're host- or data-bound), CPU↔GPU transfers, and NCCL communication — use it to spot stalls, bubbles, and launch overhead across the whole program rather than within one kernel.
- **`nvidia-smi dmon` / DCGM** give coarse live utilization, power, memory, and bandwidth counters — good for a quick "is the GPU even busy?" check, but too coarse to diagnose a kernel.

**The measurement loop in practice:** (1) compute the analytical FLOP target and the bandwidth floor (model-bytes/β for decode); (2) measure tokens/s and convert to achieved FLOP/s → MFU; (3) if MFU is low, open `torch.profiler`/`nsys` to see whether you're memory-bound (expected for decode), communication-bound (parallelism), or input-bound (data loader); (4) for a specific hot kernel, drop to `ncu` and read its arithmetic intensity straight off the roofline. The recurring questions are always the same two from the start of this section: *how many FLOPs, and how many bytes?* — everything else is bookkeeping around those two numbers.

---

## 5. Numerical precision and data types

Every number that flows through a neural network — a weight, an activation, a gradient — is stored as a finite pattern of bits. The choice of *how many* bits, and *how those bits are apportioned* between magnitude and precision, is one of the highest-leverage decisions in modern GPU computing. It simultaneously determines (a) how much memory the model occupies, (b) how fast the arithmetic runs on the GPU's specialized matrix hardware, and (c) whether the training or inference is numerically stable at all. This section builds the topic from the bit level upward.

### 5.1 How a floating-point number is built from bits

A floating-point number stores a value in three fields, in direct analogy to scientific notation `± d.ddd × 2^e`:

- **Sign bit (S):** 1 bit. 0 for positive, 1 for negative.
- **Exponent (E):** controls the **dynamic range** — how large or small a number can be. The exponent is stored with a *bias*; the actual scale is `2^(E - bias)`. More exponent bits → wider range.
- **Mantissa / significand (M), also called the fraction:** controls the **precision** — how many distinct values you can represent *between* consecutive powers of two. More mantissa bits → finer resolution. There is an implicit leading `1.` for normalized numbers, so `k` mantissa bits give `k+1` bits of significand.

The value of a normal (normalized) number is:

```
value = (-1)^S × 1.M(binary) × 2^(E - bias)
```

Two properties matter for ML and are worth internalizing because they explain almost everything that follows:

1. **Dynamic range is set by the exponent field alone.** It tells you the largest and smallest magnitudes you can express before you *overflow* (value too big → becomes `inf`) or *underflow* (value too small → becomes `0`, losing it entirely).
2. **Relative precision is set by the mantissa field alone** and is *roughly constant across the whole range* (this is the whole point of floating point versus fixed point). A useful figure of merit is **machine epsilon**, the gap between `1.0` and the next representable number, equal to `2^(-k)` for `k` mantissa bits. With 10 mantissa bits, epsilon ≈ `2^-10 ≈ 0.001`, i.e. about 3 decimal digits of precision.

#### A concrete decode

Take BF16 (8 exponent bits, bias 127, 7 mantissa bits) and decode the bit pattern for `1.5`. Sign `S=0`. We want `1.5 = 1.1(binary) × 2^0`, so the unbiased exponent is `0`, stored as `0 + 127 = 127`, and `127 = 01111111` in 8 bits. The mantissa is `1000000` (the `.1` after the implicit `1.`, padded to 7 bits). So `1.5` in BF16 is `0 01111111 1000000`. The next representable number above `1.5` is one mantissa tick away: `2^0 × 2^-7 = 1/128 ≈ 0.0078` larger, i.e. `1.5078125`. That gap *is* the precision limit of BF16 near 1.5.

### 5.2 The format zoo: bit layouts and tradeoffs

The table below gives the standard layouts. Bias and exact special-value encodings follow IEEE-754 conventions except where the format is a vendor/consortium extension (TF32, the FP8 variants, and the integer formats), which I flag explicitly.

| Format | Total bits | Sign | Exponent | Mantissa | Approx max | Approx min normal | Decimal digits (≈) | Notes |
|---|---|---|---|---|---|---|---|---|
| FP32 (single) | 32 | 1 | 8 | 23 | 3.4e38 | 1.2e-38 | ~7 | IEEE-754; the "master" reference precision |
| TF32 (TensorFloat-32) | 19* | 1 | 8 | 10 | 3.4e38 | 1.2e-38 | ~3 | NVIDIA tensor-core *compute* format; FP32 range, FP16 precision. *Inputs are read from 32-bit FP32 operands; only 19 bits (1+8+10) feed the multiplier. |
| FP16 (half) | 16 | 1 | 5 | 10 | 65504 | 6.1e-5 | ~3 | IEEE-754; narrow range → needs loss scaling for training |
| BF16 (bfloat16) | 16 | 1 | 8 | 7 | 3.4e38 | 1.2e-38 | ~2 | Google Brain format; FP32 range, low precision; training-friendly |
| FP8 E4M3 | 8 | 1 | 4 | 3 | 448 | 1.6e-2* | ~1 | Hopper/Ada; for weights & activations (precision-leaning) |
| FP8 E5M2 | 8 | 1 | 5 | 2 | 57344 | 6.1e-5* | <1 | Hopper/Ada; for gradients (range-leaning) |
| INT8 | 8 | — | — | — | +127 | −128 | — | Integer + external scale factor; not floating point |
| INT4 | 4 | — | — | — | +7 | −8 | — | Integer + external scale; aggressive inference quantization |

\*FP8 follows the OCP/NVIDIA FP8 spec (OFP8), which deviates from IEEE special-value handling. E4M3 has **no infinities** (it uses only a single NaN bit-pattern and reclaims the top of the range for finite values, giving its max of 448), while E5M2 keeps IEEE-style inf/NaN (hence its lower max of 57344). The min-*normal* values are `2^-6 ≈ 1.6e-2` for E4M3 and `2^-14 ≈ 6.1e-5` for E5M2; subnormals extend a few exponents lower (down to `2^-9` for E4M3 and `2^-16` for E5M2). Treat these as representative.

#### The central tradeoff, made visual

Within a fixed bit budget you are *spending* bits on either range (exponent) or precision (mantissa). FP16 and BF16 both cost 16 bits but split them very differently:

```
FP16:  S | EEEEE | MMMMMMMMMM     5 exp, 10 mant  → small range, decent precision
BF16:  S | EEEEEEEE | MMMMMMM     8 exp, 7 mant   → FP32 range, coarse precision
```

This single picture is the key to the most important practical fact in mixed-precision training, which we develop next.

#### Why BF16 is friendlier than FP16 for training

BF16 has the **exact same 8-bit exponent as FP32**. That means BF16 has the *same dynamic range* as FP32 — it can represent the same largest and smallest magnitudes (≈3.4e38 down to ≈1.2e-38 for normals). It just has fewer mantissa bits (7 vs 23), so each number is coarser.

FP16, by contrast, has only a 5-bit exponent. Its maximum representable value is **65504**, and its smallest normal is **6.1e-5** (`2^-14`). In deep networks, gradients routinely have magnitudes well below `1e-5` — and in FP16 those simply **underflow** (toward subnormals and then to zero) **and vanish**. Activations in large models, conversely, can spike above 65504 and **overflow to `inf`**, which then poisons everything downstream as `NaN`.

The practical consequence:

- **BF16 is nearly a drop-in replacement for FP32 ranges.** You can usually train in BF16 without any special tricks, because no value that fit in FP32's range will overflow or underflow in BF16. You only lose precision, and deep nets tolerate that surprisingly well (gradients are noisy anyway).
- **FP16 requires loss scaling** (Section 5.4) precisely to drag the tiny gradients back up into FP16's representable window before they underflow.

The cost of BF16's friendliness is precision: with 7 mantissa bits, epsilon ≈ `2^-7 ≈ 0.008`, under 2 decimal digits. This matters most for **accumulation** (Section 5.5), which is why even BF16 pipelines accumulate in FP32.

### 5.3 Why precision matters I: memory (bytes per parameter)

This ties directly back to the memory section. The bytes-per-element is simply `total_bits / 8`:

| Format | Bytes/element |
|---|---|
| FP32 | 4 |
| TF32 | 4 (read from FP32 operands) |
| FP16 / BF16 | 2 |
| FP8 | 1 |
| INT8 | 1 |
| INT4 | 0.5 |

**Worked example — weights of a 7B-parameter model.** "7B" means `7×10^9` parameters. Multiply by bytes/param:

| Precision | Weight memory |
|---|---|
| FP32 | `7e9 × 4 = 28e9 B = 28 GB` |
| BF16/FP16 | `7e9 × 2 = 14e9 B = 14 GB` |
| FP8/INT8 | `7e9 × 1 = 7e9 B = 7 GB` |
| INT4 | `7e9 × 0.5 = 3.5e9 B = 3.5 GB` |

(These are GB in the base-10 sense, `10^9` bytes. GPU vendors usually quote memory in GB = `10^9` too, while the OS may report GiB = `2^30 ≈ 1.074e9`. A "24 GB" card has ≈ 22.4 GiB. Keep the distinction in mind when a model "just barely" fits.)

This is why precision is the first lever you reach for when a model won't fit. A 7B model in FP32 (28 GB) does not fit on a 24 GB card; in BF16 (14 GB) it fits with room for activations and KV-cache; in INT4 (3.5 GB) it fits on a laptop GPU. Note that **weights are only part of the story** — during inference you also need the KV-cache (which you may keep in FP16/BF16 or quantize to INT8), and during training you additionally need gradients, optimizer states, and activations, each with its own precision (see Section 5.4 and the memory section).

### 5.4 Why precision matters II: speed (tensor cores)

Modern NVIDIA GPUs contain two kinds of math units:

- **CUDA cores:** general-purpose scalar/vector ALUs (arithmetic logic units) that do ordinary FP32/FP64 work.
- **Tensor Cores:** specialized units that compute a small **matrix-multiply-accumulate (MMA)** — each core handles a `4×4×4` tile (`D = A×B + C`), and a full 32-thread warp cooperates to do a larger `16×16×16` operation — in a small number of hardware steps. They are dramatically faster than CUDA cores *but only for the reduced-precision formats they support*, and they accumulate the products into a higher-precision register (typically FP32).

Because almost all the FLOPs in a transformer are matrix multiplies (the `QKV` projections, attention scores, MLP layers), tensor-core throughput essentially *is* your training/inference speed. And tensor-core throughput climbs steeply as precision drops, because a narrower format lets the hardware pack more multiply-accumulate lanes into the same silicon and move operands in fewer bytes.

#### Representative relative throughput

The following are **representative** peak *dense* (non-sparse) matrix throughputs; exact numbers are GPU-generation- and SKU-specific (Ampere A100 vs Hopper H100 vs Blackwell differ, and SXM vs PCIe variants differ within a generation), and "with structured sparsity" figures double them again. I give H100-SXM-class figures (TFLOP/s = tera-FLOPs per second = `10^12` FLOP/s); note that NVIDIA's marketing tables usually lead with the *sparsity-enabled* numbers, so halve those to get the dense figures below.

| Compute path | Representative peak (H100 SXM, dense) | Relative to FP32 |
|---|---|---|
| FP32 (CUDA cores) | ~67 TFLOP/s | 1× |
| TF32 (tensor core) | ~989 TFLOP/s | ~15× |
| FP16/BF16 (tensor core) | ~1979 TFLOP/s | ~30× |
| FP8 (tensor core) | ~3958 TFLOP/s | ~59× |
| INT8 (tensor core) | ~3958 TOP/s | ~59× |

(For scale, the older A100 SXM is roughly: FP32 ~19.5, TF32 ~156, BF16/FP16 ~312 TFLOP/s, INT8 ~624 TOP/s dense — about 6× lower than H100 at the low-precision end, and with no native FP8.)

The headline: **just switching matmuls from FP32 to TF32 or BF16 buys more than an order of magnitude in math throughput, for free, before any other optimization.** This is why "use mixed precision" is the very first performance advice anyone gives. (Whether you actually *realize* this speedup depends on whether the kernel is compute-bound or memory/bandwidth-bound — see the profiling section — but for large matmuls you typically do.)

A subtle but important corollary: lower precision also *halves or quarters the bytes moved* per operand. Many transformer operations (especially at small batch size during inference, and especially attention) are **memory-bandwidth-bound**, not compute-bound. There, the win from BF16 vs FP32 comes not from the tensor cores at all but from moving 2 bytes instead of 4 across the memory bus. Precision pays off on *both* axes.

### 5.5 Mixed-precision training: the recipe

"Mixed precision" does **not** mean "do everything in 16 bits." It means: do the expensive, error-tolerant operations (the matmuls) in a fast low-precision format, while keeping the few precision-sensitive bookkeeping operations in FP32. The canonical recipe (Micikevicius et al., 2018) has three ingredients.

#### (1) Master weights in FP32

The optimizer keeps a **master copy of the weights in FP32**. Each step:

1. Cast the FP32 master weights down to BF16/FP16 for the forward and backward passes (the matmuls run on tensor cores).
2. Compute the BF16/FP16 gradients.
3. Apply the optimizer update to the **FP32** master weights using those gradients.

**Why the FP32 master is essential — a worked example of "swamping."** Suppose a weight is `1.0` and its update this step is `learning_rate × gradient = 1e-4`. In FP16, epsilon near `1.0` is `2^-10 ≈ 9.8e-4`. The update `1e-4` is *smaller than the gap to the next representable FP16 number*, so `1.0 + 1e-4` rounds right back to `1.0` — the update is silently lost, every step, forever. In BF16 it is even worse: epsilon near 1.0 is `2^-7 ≈ 7.8e-3`. Keeping the accumulator (the master weight) in FP32, whose epsilon near 1.0 is `2^-23 ≈ 1.2e-7`, lets these tiny updates actually accumulate. This is the single most important reason mixed-precision training keeps an FP32 copy.

#### (2) Loss scaling (FP16) — and why BF16 mostly skips it

As noted, FP16 gradients can underflow to zero (anything below ≈`6.1e-5` for normals, ≈`6e-8` for the smallest subnormals). **Loss scaling** fixes this with a beautifully simple trick that exploits the chain rule's linearity:

1. Multiply the loss by a large constant `S` (e.g. `S = 1024` or dynamically chosen) *before* calling backward.
2. By linearity, **every gradient is now scaled up by `S`**, shifting the whole gradient distribution up into FP16's representable range, away from the underflow cliff.
3. Before the optimizer step, **unscale** by dividing the gradients by `S` (this division happens in FP32).

**Dynamic loss scaling** automates the choice of `S`: start high, and if any gradient becomes `inf`/`NaN` (overflow), skip that step and halve `S`; if many steps pass cleanly, double `S`. This keeps `S` as large as possible without overflowing.

**BF16 mostly does not need loss scaling** because its FP32-matching exponent already covers the gradient range — there is no underflow cliff to climb away from. This operational simplicity (no scaler to tune, no skipped steps) is a major reason BF16 has become the default for training on Ampere and later hardware.

#### (3) Which ops stay in FP32

Some operations are run in FP32 even inside an autocast region because they are numerically fragile:

- **Reductions / accumulations** — summing many terms (the accumulate in matmul, batch-norm statistics, the `sum` in a softmax denominator, gradient all-reduce across GPUs). Adding many small numbers into a low-precision accumulator loses the small ones to rounding (see Section 5.6). Tensor cores already accumulate in FP32 internally; framework-level reductions are also kept in FP32.
- **Softmax** — involves `exp()` (which can overflow) and a sum (a reduction). Computed in FP32 (with the standard max-subtraction trick for stability).
- **LayerNorm / RMSNorm / BatchNorm** — compute a mean and variance (reductions) and divide by a small number; precision-sensitive, kept in FP32.
- **Loss computation** — often FP32, especially cross-entropy with its `log` and `sum`.

#### Doing it in PyTorch: `torch.amp` / `autocast`

PyTorch's Automatic Mixed Precision (AMP) implements exactly this recipe with two objects:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()            # the dynamic loss scaler (FP16); a no-op-ish path for BF16

for x, y in loader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):   # or torch.float16
        out  = model(x)          # matmuls run in BF16/FP16 on tensor cores...
        loss = loss_fn(out, y)   # ...while softmax/layernorm/loss auto-run in FP32
    scaler.scale(loss).backward()  # scale loss → gradients scaled up (FP16 path)
    scaler.step(optimizer)         # unscale, skip-if-inf, then optimizer.step()
    scaler.update()                # adjust S for next iteration
```

`autocast` maintains an internal **op allow-list**: matmul/conv/linear run in the low-precision dtype; softmax, layernorm, and reductions are automatically promoted to FP32. You do not annotate ops by hand. With `dtype=torch.bfloat16` the `GradScaler` is effectively unnecessary (and is often omitted), reflecting point (2).

#### The full training memory bill, by precision

To connect back to the memory section, here is where the bytes go for a `P`-parameter model trained with Adam in mixed precision (per parameter):

| Component | Precision | Bytes/param |
|---|---|---|
| BF16 weights (compute copy) | BF16 | 2 |
| FP32 master weights | FP32 | 4 |
| Adam 1st moment (m) | FP32 | 4 |
| Adam 2nd moment (v) | FP32 | 4 |
| **Subtotal (states above)** | | **14** |

Add a gradient buffer and you reach the well-known **"16 bytes/param"** rule of thumb for Adam mixed-precision training: the canonical accounting is FP32 master (4) + Adam `m` (4) + Adam `v` (4) + 2-byte weights + 2-byte gradients = 16. (The exact figure shifts by a couple of bytes depending on whether the gradient and the compute-copy weights are kept in BF16 or FP32.) Using the 16 bytes/param figure, a 7B model needs `7e9 × 16 ≈ 112 GB` *just for weights, gradients, and optimizer states*, before activations — which is why a 7B model trains on an 80 GB A100/H100 only with sharding (ZeRO/FSDP) or offloading. Inference, needing only the 14 GB of BF16 weights plus KV-cache, is far cheaper — the asymmetry that makes your reasoning-model inference workload tractable on a single card.

### 5.6 Numerical stability: the failure modes

Low precision does not *usually* break models, but when it does, it is one of a small number of classic pathologies. Knowing them by name makes debugging a `NaN` far less mysterious.

#### Overflow → `inf` → `NaN`

A value exceeds the format's max (FP16: 65504; FP8 E4M3: 448) and becomes `inf` (or, in E4M3's no-infinity encoding, saturates/NaNs). Arithmetic on `inf` (e.g. `inf - inf`, `inf/inf`, `0×inf`) yields `NaN`, which then propagates through every subsequent operation and contaminates the whole tensor. Common sources: an un-stabilized `exp()` in softmax, a too-large loss scale, or exploding activations. The fix is range-aware: stabilize the op (max-subtraction in softmax), clip, or use a wider-range format (BF16 over FP16).

#### Underflow → 0

A value falls below the smallest representable magnitude and becomes exactly `0`, silently. The danger is *silence*: no error, just a gradient that quietly vanishes and a weight that stops learning. This is the FP16-gradient problem that loss scaling exists to solve.

#### Catastrophic cancellation

When you subtract two nearly-equal numbers, the leading significant digits cancel and you are left with only the noisy trailing bits — the *relative* error explodes. Example in a 4-significant-digit format: `1.235 − 1.234 = 0.001`, but each input was only known to ~4 digits, so the result `0.001` has essentially one (noisy) significant figure left. This is why variance is computed as `E[x²] − E[x]²` *only* in a stabilized form (or via Welford's algorithm), and why naive implementations of `log(1+x)` are replaced by `log1p(x)` for small `x`.

#### Accumulation precision and why reductions stay high-precision

Summing `N` numbers in a low-precision accumulator is the most common silent precision killer. Once the running sum grows large, each newly added small term may be **smaller than the accumulator's epsilon at that magnitude**, and gets rounded away entirely.

**Worked example.** Add `1.0 + 0.0001 + 0.0001 + …` in FP16. After the sum reaches `1.0`, FP16 epsilon there is ≈`9.8e-4`. Each `0.0001` increment is below that gap, so *every subsequent add rounds back to the current value* — the sum sticks near `1.0` no matter how many terms you add. In FP32 (epsilon `1.2e-7` near 1.0) the same sum accumulates correctly for a long time. This is exactly why:

- Tensor cores accumulate their products in **FP32** even when inputs are BF16/FP16/FP8.
- Framework reductions (softmax denominators, norm statistics, loss sums, multi-GPU gradient all-reduce) are kept in **FP32**.
- Very long reductions sometimes use **Kahan/compensated summation** or tree/pairwise reduction to limit error growth from `O(N)` toward `O(log N)`.

The general principle: **it is fine to multiply in low precision, but accumulate the result of many such operations in high precision.** Almost every stable mixed-precision design obeys this rule.

### 5.7 Quantization for inference

Mixed-precision *training* still keeps an FP32 backbone. **Quantization** goes further: it represents the model — at least its weights, sometimes its activations too — in very low-bit *integer* formats (INT8, INT4) for **inference**, trading a small amount of accuracy for large gains in memory and (sometimes) speed. The core operation is mapping a real-valued tensor to integers via a **scale** (and optionally a **zero-point** offset):

```
x_int  = round(x_real / scale) + zero_point        # quantize
x_real ≈ (x_int − zero_point) × scale              # dequantize
```

The integers are stored compactly; the `scale`/`zero_point` are kept in higher precision and reapplied at compute time. The art of quantization is choosing those scales so that the rounding error costs as little accuracy as possible.

#### Two orthogonal axes

**Axis 1 — when you quantize:**

- **Post-Training Quantization (PTQ):** take an already-trained FP/BF16 model and quantize it directly, using a small **calibration** set (a few hundred unlabeled samples) to estimate the activation ranges/scales. No retraining. Cheap, fast, and the overwhelmingly common path for LLM inference. Quality is excellent at INT8 and usually good at INT4 with modern methods.
- **Quantization-Aware Training (QAT):** simulate quantization (insert "fake-quant" rounding nodes) *during* training or fine-tuning, so the network *learns* weights robust to the rounding. Recovers more accuracy, especially at very low bit-widths (≤4-bit), but costs a training run. Used when PTQ accuracy is insufficient.

**Axis 2 — what you quantize:**

- **Weight-only quantization:** quantize the weights (the big memory consumer) to INT4/INT8, but **dequantize back to FP16/BF16 at compute time** and do the matmul in floating point. This shrinks the model in memory and — crucially for LLM inference, which is usually **memory-bandwidth-bound** at low batch — speeds up the weight load from VRAM. This is what GPTQ and AWQ do, and it is the dominant approach for serving LLMs on a single GPU. Activations stay in FP16/BF16, so accuracy is well preserved.
- **Weight + activation quantization (e.g. W8A8):** quantize *both* operands so the matmul itself runs on **integer tensor cores** (INT8 MMA), giving a genuine compute speedup, not just a memory one. Harder, because **activations are much harder to quantize than weights** (next subsection).

#### Why activations are the hard part: outliers

Empirically, in large transformers a *small number* of activation channels (feature dimensions) carry values 10–100× larger than the rest — **outlier features** (Dettmers et al., LLM.int8()). A single per-tensor scale chosen to span those outliers wastes almost all the INT8 codes on the rare giants and crushes the ordinary values into a handful of levels, destroying accuracy. The major weight+activation methods are essentially different answers to this outlier problem:

- **LLM.int8() (bitsandbytes):** decompose the matmul — keep the few outlier dimensions in FP16 and do only the well-behaved majority in INT8, then recombine. Mixed INT8/FP16.
- **SmoothQuant:** mathematically migrate the "difficulty" from activations into weights by a per-channel scaling that leaves the product unchanged but makes activations smoother and easier to quantize.

#### Granularity of scales: per-tensor vs per-channel vs per-group

The finer the granularity of the scale factor, the better outliers are contained, at a small storage/compute cost for the extra scales:

- **Per-tensor:** one `scale` for the whole tensor. Cheapest, least accurate.
- **Per-channel (per-row/per-column):** a separate `scale` per output channel. Standard for weights; cheaply absorbs channel-to-channel magnitude variation.
- **Per-group / block-wise:** a separate `scale` for every contiguous group of (e.g.) 64 or 128 weights. This is what 4-bit methods (GPTQ, AWQ, GGUF's k-quants, NF4) use to keep INT4 accurate — the group size (e.g. 128) is a memory-vs-accuracy knob.

#### The method families you will actually encounter

| Family | Bits (typical) | Type | One-line mechanism | Where you meet it |
|---|---|---|---|---|
| **bitsandbytes** (LLM.int8(), NF4) | 8 / 4 | Weight-only (+mixed) | On-the-fly PTQ; NF4 = "normal-float 4" group-wise; basis of **QLoRA** | Hugging Face `load_in_8bit`/`load_in_4bit` |
| **GPTQ** | 4 (3/8) | Weight-only PTQ | Second-order (Hessian-aware) per-layer error-correcting rounding | `auto-gptq`, many HF checkpoints |
| **AWQ** | 4 | Weight-only PTQ | Protect the *salient* weight channels (those that see large activations) via per-channel scaling | `autoawq`, vLLM serving |
| **GGUF / llama.cpp k-quants** | 2–8 | Weight-only PTQ | Block-wise integer quants (Q4_K_M, Q5_K, …) tuned for **CPU/Metal/edge** | local `llama.cpp`/Ollama |
| **SmoothQuant** | 8 (W8A8) | Weight+activation PTQ | Shift activation outliers into weights for true INT8 matmul | server inference, TensorRT-LLM |
| **QAT** | ≤4 | Train-time | Learn quant-robust weights with fake-quant nodes | when PTQ accuracy is insufficient |

#### The accuracy / speed / memory tradeoff, concretely

For a 7B model, the practical operating points look roughly like:

| Precision | Weight memory | Typical quality loss | Notes |
|---|---|---|---|
| BF16 | 14 GB | reference | baseline |
| INT8 (weight-only) | 7 GB | negligible (<~1% on most benchmarks) | safe default for shrinking |
| INT4 (GPTQ/AWQ, group=128) | ~3.5–4 GB | small but measurable (often 1–3% on hard tasks) | dominant single-GPU serving point |
| INT4 with naive per-tensor | ~3.5 GB | large / can break | why group-wise + GPTQ/AWQ exist |

Two caveats worth flagging for a reasoning model specifically: (1) chain-of-thought generation is *long*, so small per-token quality losses can **compound over a long reasoning trace** more than they would on a single-token classification — evaluate quantized reasoning models on end-to-end task accuracy, not just perplexity. (2) Weight-only INT4 mainly buys *memory and bandwidth*, not necessarily lower latency at very small batch where you may already be bandwidth-bound; measure.

### 5.8 FP8 training and inference (Hopper / Ada and later)

FP8 is the newest rung on the ladder and is a genuine *floating-point* format (unlike INT8), so it keeps an exponent and thus a notion of dynamic range — which is what lets it be used for *training*, not just inference. Hopper (H100) and Ada (L4/L40) tensor cores execute FP8 MMAs natively, at roughly **2× the BF16 rate** (e.g. ~3958 vs ~1979 dense TFLOP/s on H100 SXM; see Section 5.4).

The two variants are used asymmetrically, mirroring the range-vs-precision split:

- **E4M3** (4 exponent, 3 mantissa, max 448): more precision, less range. Used for **weights and activations** in the forward pass, where values are bounded and you want resolution.
- **E5M2** (5 exponent, 2 mantissa, max 57344): more range, less precision. Used for **gradients** in the backward pass, where the range is wide and you mainly need to avoid underflow.

Because FP8's range is small, FP8 training relies on **per-tensor (or finer) scaling factors** that are tracked and updated as training proceeds (NVIDIA's Transformer Engine automates this with delayed scaling based on recent amax history). The accumulation, as always, is in FP32. FP8 training is now used for frontier-scale pretraining; FP8 *inference* (often E4M3 weights+activations) is increasingly standard on H100-class serving stacks (TensorRT-LLM, vLLM) for another step down in memory and up in throughput beyond INT8 — with the advantage over INT8 that the retained exponent handles activation outliers more gracefully. (Blackwell pushes this further still with native FP4/NVFP4 microscaling formats, but FP8 remains the practical training/inference workhorse on Hopper/Ada.)

### 5.9 Practical guidance: what precision should *you* use?

For the concrete workload here — **inference of a small reasoning model (≈1.5–8B) on an A100 or H100** — the decision tree is short:

1. **Default to BF16.** On A100/H100 it runs on tensor cores at full speed, has FP32's dynamic range (no overflow/underflow drama, no loss-scaling machinery), and a 1.5B model needs only ≈3 GB of weights, a 7–8B model ≈14–16 GB — both fit comfortably with room for a large KV-cache and batch. For a reasoning model emitting long chains-of-thought, the headroom matters because the **KV-cache grows with sequence length** and will dominate memory at long context; keeping weights in BF16 (rather than FP32) frees that room. This is the right default and you should reach for a lower precision only if a specific constraint forces it.

2. **Use FP16 only if** you are on hardware where it is faster or better-supported than BF16 (older cards) — but prefer BF16 whenever available to avoid the loss-scaling/overflow headaches. For *inference* the stability gap is smaller than in training, but BF16 remains the cleaner choice on Ampere+.

3. **Reach for INT8/INT4 weight-only quantization (GPTQ/AWQ) only when memory-constrained** — fitting a larger model on a smaller card, maximizing batch/context, or serving many models. Expect negligible loss at INT8 and small-but-real loss at INT4; **validate on your actual end-to-end reasoning task**, since errors compound over long generations. INT4 is the right tool to squeeze an 8B-class model onto a 24 GB consumer card or a DGX Spark, less necessary on an 80 GB A100/H100 where BF16 already fits.

4. **Consider FP8 on H100** if your serving stack (TensorRT-LLM, vLLM) supports it for your model: it gives an INT8-like memory step-down with better outlier behavior and a real throughput bump, while staying floating-point. It is overkill for a 1.5B model on an 80 GB card but valuable when squeezing throughput at scale.

5. **Keep the KV-cache precision in mind as a separate knob.** You can serve weights in BF16 but store the KV-cache in FP16 or even INT8 to extend context length; this is an independent decision from weight precision and is often the highest-leverage memory lever for long-reasoning workloads (see the memory and inference sections).

The summary heuristic: **BF16 for almost everything by default; quantize to INT4/INT8 only to fit or to scale; FP8 when your H100 stack supports it and you want throughput. Always accumulate in FP32, and always validate a quantized reasoning model end-to-end rather than trusting perplexity.**

---

## 6. GPU memory: what consumes it in training and inference

If a single resource decides whether your job runs at all, it is GPU memory — the on-card DRAM (called **VRAM**, video RAM; on modern data-center cards this is **HBM**, High-Bandwidth Memory). Compute (FLOP/s) determines *how fast* a run goes; memory determines *whether it runs*. The overwhelming majority of "CUDA out of memory" (OOM) failures are not mysterious — they are an arithmetic accounting problem you can do on paper before you ever launch. This section teaches that accounting exhaustively: every consumer of VRAM in both training and inference, the formulas, worked numbers, the allocator behavior that makes `nvidia-smi` lie to you, and the levers to fit a model that "doesn't fit."

Throughout, I will be explicit about units. I use **GB** = 10⁹ bytes (decimal) and **GiB** = 2³⁰ = 1,073,741,824 bytes (binary). GPU vendors label cards in GB (an "80 GB" A100/H100 is ~80 × 10⁹ bytes ≈ 74.5 GiB), but `nvidia-smi` and PyTorch report in MiB/GiB. This ~7% gap matters when you are 2 GB from the edge, so I will flag it where it bites. (One nuance: HBM cards are typically marketed at a round figure that is already close to a binary boundary — e.g. an "80 GB" A100 reports `81920 MiB` = exactly 80 GiB to `nvidia-smi`, because the marketing "80 GB" is really 80 GiB of HBM. Always trust the `nvidia-smi` MiB total over the marketing label.)

### 6.1 The bytes-per-parameter foundation

Every memory estimate starts from one fact: a parameter is a number, and a number occupies a fixed number of bytes determined by its **dtype** (data type / numerical precision).

| dtype | bytes/element | notes |
|---|---|---|
| FP32 (float32) | 4 | full precision; "master" copy in mixed-precision training |
| TF32 | (stored as 4) | a compute mode on tensor cores, not a storage format; operands still occupy 4 bytes in memory |
| FP16 (float16) | 2 | half precision; narrow exponent range |
| BF16 (bfloat16) | 2 | half precision; FP32-like exponent range, preferred for stability |
| FP8 (e4m3 / e5m2) | 1 | emerging on H100/H200+; mostly compute/quantized weights |
| INT8 | 1 | quantized inference |
| INT4 / NF4 | 0.5 | 4-bit quantized weights (e.g. QLoRA, GPTQ, AWQ) |

The single most useful formula in this entire document:

```
weight_memory_bytes = num_parameters × bytes_per_parameter
```

**Worked example — weights only, in BF16:**
- 1.5B params: 1.5 × 10⁹ × 2 = 3.0 × 10⁹ bytes = **3.0 GB** (≈ 2.79 GiB)
- 7B params: 7 × 10⁹ × 2 = 1.4 × 10¹⁰ = **14 GB** (≈ 13.0 GiB)
- 70B params: 70 × 10⁹ × 2 = 1.4 × 10¹¹ = **140 GB** (≈ 130 GiB)

Note immediately that a 70B model in BF16 (140 GB) does **not** fit on a single 80 GB GPU even with zero room for anything else. That single arithmetic fact dictates multi-GPU strategy, quantization, or both — and we have not yet added a single byte of activations, gradients, or KV cache. (It *does* fit, just barely, in the 128 GB unified memory of a single NVIDIA DGX Spark / GB10 box — though there the ~273 GB/s LPDDR5x bandwidth, not capacity, becomes the binding constraint on tokens/s.)

This is also why your `R1-Distill-1.5B` model is so comfortable: 3 GB of weights leaves enormous headroom on a 24 GB card, which is exactly why a 1.5B model is the right size for iterating on steering/annotation experiments on modest hardware.

### 6.2 Training memory: the six consumers

When you *train* (or fine-tune with full backprop), GPU memory is consumed by six distinct things. Let me enumerate them, then give the per-parameter accounting, then a worked total.

#### (1) Model parameters
The weights themselves, as above. In mixed-precision training you typically hold them in the low-precision compute dtype (BF16, 2 bytes) **and** keep an FP32 "master copy" (4 bytes) that the optimizer updates — more on this below.

#### (2) Gradients
Backpropagation computes a gradient for **every** trainable parameter. The gradient tensor has the same shape as the parameter, so it costs the same bytes/param as one copy of the weights — typically 2 bytes/param in BF16 (some setups accumulate gradients in FP32 → 4 bytes/param).

#### (3) Optimizer states
This is the consumer people forget, and it is usually the largest of the parameter-proportional costs. **Adam** (and AdamW, the default for transformers) maintains **two** running statistics per parameter:
- the first moment `m` (exponential moving average of the gradient),
- the second moment `v` (EMA of the squared gradient).

Both are stored in **FP32** for numerical stability, regardless of the compute dtype → 4 + 4 = **8 bytes/param** just for the optimizer. Plain SGD with momentum keeps one buffer (4 bytes/param); SGD without momentum keeps none. This is a concrete reason to prefer memory-light optimizers (e.g. 8-bit Adam via bitsandbytes, which stores `m`/`v` quantized at ~2 bytes/param total, or Adafactor) when you are tight.

#### (4) Activations stored for the backward pass
The intermediate tensors (layer outputs, attention scores, normalization statistics, etc.) computed during the **forward** pass must be **kept in memory** because the backward pass needs them to compute gradients via the chain rule. Unlike (1)–(3), this cost does **not** scale with parameter count — it scales with **batch size × sequence length × hidden size × number of layers**. For large batches and long sequences it frequently **dominates everything else**. We treat it in depth in §6.3.

#### (5) Framework / CUDA context overhead
Loading CUDA itself costs memory before you allocate a single tensor: the CUDA context, the cuBLAS/cuDNN kernels, NCCL communication buffers (for multi-GPU), and PyTorch's own bookkeeping. Budget **~0.5–2 GB per GPU** as fixed overhead. You can watch it appear: `import torch; torch.cuda.init()` followed by a tiny `.cuda()` tensor already shows several hundred MB resident in `nvidia-smi`.

#### (6) Fragmentation
The caching allocator (§6.6) reserves memory in blocks. Over a run with varying tensor sizes, free memory can become **fragmented** — there is enough *total* free memory for an allocation, but no single *contiguous* block large enough, so the allocation fails with OOM despite "free" memory existing. This is real and can cost you 10–20% of usable VRAM in pathological cases.

#### The mixed-precision per-parameter accounting

Putting (1)–(3) together for the standard **Adam mixed-precision** recipe, per trainable parameter:

| component | dtype | bytes/param |
|---|---|---|
| BF16 weights (compute copy) | BF16 | 2 |
| FP32 master weights | FP32 | 4 |
| BF16 gradients | BF16 | 2 |
| Adam first moment `m` | FP32 | 4 |
| Adam second moment `v` | FP32 | 4 |
| **total (params+grads+optimizer)** | | **16 bytes/param** |

This is the famous **"16 bytes per parameter"** figure (some references quote 18–20 when gradients are also kept in FP32, or when there are extra master/checkpoint buffers). Memorize 16 B/param for Adam mixed-precision as your back-of-envelope constant, then add activations and overhead.

**Worked total — full fine-tuning a 7B model with Adam mixed-precision:**
- params + grads + optimizer: 7 × 10⁹ × 16 = 1.12 × 10¹¹ = **112 GB**
- activations: depends on batch/seq — say **10–40 GB** (see §6.3)
- CUDA/overhead: **~2 GB**
- **Total ≈ 124–154 GB**

A 7B model whose *weights* are a mere 14 GB needs **~9–11× that** to train with full backprop and Adam. This is why full fine-tuning of even "small" 7B models does not fit on a single 80 GB card, and why **parameter-efficient fine-tuning (PEFT)** like **LoRA/QLoRA** exists: if you freeze the base weights and train only small low-rank adapters, consumers (2) and (3) shrink to the adapter's parameter count (often <1% of the model), and only (1) — the frozen weights, possibly 4-bit quantized — remains large. A QLoRA 7B fine-tune (4-bit frozen base ≈ 3.5 GB + tiny adapter states + activations) fits comfortably on a 24 GB card, an order-of-magnitude reduction versus full fine-tuning. Worth keeping in mind for the LoRA/GRPO spillover experiments you have scoped.

### 6.3 Activation memory: why it scales, why it dominates, and how to crush it

#### Why activations scale the way they do

During the forward pass, each transformer layer produces intermediate tensors that must be retained for backprop. The dominant term is roughly proportional to:

```
activation_memory ≈ batch_size × seq_len × hidden_size × num_layers × (bytes) × c
```

where `c` is a constant (often quoted on the order of 10–34× the bare hidden-state size, depending on whether attention score matrices, MLP intermediates, dropout masks, and layernorm stats are counted, and whether attention is materialized or fused via FlashAttention). The canonical Korthikanti et al. ("Reducing Activation Recomputation in Large Transformer Models") accounting gives ≈ 34 per layer when the attention-score matrices are materialized and ≈ 12 with selective recomputation / fused attention — that's the source of the 10–34 range.

The key qualitative facts:
- **Linear in batch size.** Double the batch, double the activations. This is your most direct memory lever and why "just lower the batch size" is the first OOM remedy.
- **Linear in sequence length** for the per-layer feed-forward/hidden activations — but the *attention score matrix* is `batch × heads × seq_len²`, i.e. **quadratic in sequence length** if materialized. FlashAttention avoids storing the full `seq²` score matrix (it recomputes blocks on the fly), turning that quadratic memory term into a near-linear one — a major reason it is the default.
- **Linear in number of layers**, because each layer's activations are stored independently.
- **Independent of optimizer choice** — Adam vs SGD changes (3), not (4).

**Worked example — activations for a 7B-scale model:**
Take hidden_size `h` = 4096, num_layers `L` = 32, BF16 (2 bytes). A common engineering estimate for activations *with* an attention-score term, per token, is on the order of `L × h × (constant ~30) × 2 bytes`. For a single token that is roughly 32 × 4096 × 30 × 2 ≈ 7,864,320 bytes ≈ **7.5 MB/token**. For batch 8 × seq 2048 = 16,384 tokens: 16,384 × 7.5 MB ≈ **123 GB** if nothing is recomputed. This is why, *without* mitigation, activations alone can exceed weights+optimizer combined. (The exact constant is implementation-dependent — and with FlashAttention plus selective recomputation the effective constant is much smaller — so treat this as an order-of-magnitude tool, not a precise predictor; measure with `torch.cuda.memory_summary()` for your stack.)

#### Activation / gradient checkpointing (recomputation)

The classic time-for-memory trade. Normally you store *all* layer activations for backward. With **activation checkpointing** (also called **gradient checkpointing** or **recomputation**) you store only a subset — typically the inputs at certain "checkpoint" boundaries (e.g. one per transformer layer) — and **recompute** the discarded intermediates on the fly during the backward pass.

- **Memory saving:** the saving depends on the checkpointing scheme. The widely cited **O(√L)** result (Chen et al. 2016) comes from placing checkpoints every √L layers — it reduces the *number of stored checkpoints* to O(√L) while recomputing within each segment. The common, simpler **"checkpoint every transformer layer"** scheme instead stores just one (small) activation tensor per layer and recomputes each layer's heavy internal intermediates during backward; in practice this yields a large reduction in the *intermediate* activation memory — often a **3–5× reduction** in total activation memory for transformer stacks. (Note these are two distinct schemes; don't conflate the √L checkpoint *count* with the per-layer scheme's reduction factor.)
- **Cost:** one extra forward pass over the recomputed regions during backward → typically **~20–33% more compute time** (one extra forward ≈ ⅓ of a forward+backward step).

In PyTorch: `torch.utils.checkpoint.checkpoint(...)`, or `model.gradient_checkpointing_enable()` in HuggingFace Transformers. This is almost always the first thing to enable when activations OOM you but you cannot lower the batch (e.g. you need a large effective batch for stable training — though note you can also get that with **gradient accumulation**, running several small micro-batches and summing their gradients before stepping, which trades wall-clock for memory without recomputation).

### 6.4 Inference memory: weights + KV cache + activations + overhead

Inference is much lighter than training because **consumers (2), (3), and most of (4) vanish**: no gradients, no optimizer states, and activations need not be retained for a backward pass (each layer's activations can be freed once the next layer consumes them). What *replaces* them, and what becomes the new dominant variable cost for autoregressive generation, is the **KV cache**.

#### The four inference consumers
1. **Weights** — `params × bytes_per_param`, exactly as §6.1. Often the largest single static cost.
2. **KV cache** — stored keys and values for every past token, every layer (derived below). The dominant *variable* cost; grows with context length and batch.
3. **Activations** — only the transient working set for the current forward step (one token, or one prefill chunk). Small relative to training because nothing is retained for backward. (Prefill over a long prompt is the exception: a large prompt processed in one shot momentarily materializes activations for all prompt tokens at once, which can spike memory — chunked prefill mitigates this.)
4. **Overhead** — CUDA context, kernels, and for serving engines like vLLM, large pre-reserved pools (vLLM grabs ~90% of VRAM by default for its paged KV allocator; see §6.7).

#### Why a KV cache exists

In autoregressive decoding, generating token *t* requires attention over **all** previous tokens 1…*t*−1. The keys (K) and values (V) for those previous tokens were already computed when they were generated. Rather than recompute them at every step (O(t²) total recompute), you **cache** them. The cache grows by one K and one V vector per layer with **every** generated token — so its size grows linearly with the sequence length you have produced.

#### Deriving the KV-cache size formula

For one token, at one layer, you store one K vector and one V vector. Each has dimension `n_kv_heads × head_dim` (with **Grouped-Query Attention / GQA** or **Multi-Query Attention / MQA**, `n_kv_heads` can be much smaller than the number of query heads — a deliberate design choice precisely to shrink the KV cache). Stacking over all layers, all tokens, the whole batch, and accounting for the factor of 2 (one K **and** one V):

```
kv_cache_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_elem
```

The `2` is K-and-V; `bytes_per_elem` is the cache dtype (commonly 2 for FP16/BF16, or 1 with FP8/INT8 KV-cache quantization).

**Worked example A — a 7B-class model (Llama-2-7B shape):**
`n_layers` = 32, `n_kv_heads` = 32 (no GQA), `head_dim` = 128, FP16 (2 bytes). Per token, the cache costs:
```
2 × 32 × 32 × 128 × 2 = 524,288 bytes = 0.5 MiB ≈ 0.5 MB per token
```
- At seq_len = 4,096, batch = 1: 0.5 MB × 4,096 = **~2 GB**
- At seq_len = 4,096, batch = 16: × 16 = **~32 GB** — now *larger than the 14 GB of weights.*
- At seq_len = 32,768 (long context), batch = 1: 0.5 MB × 32,768 = **~16 GB** — again exceeding the weights.

**Worked example B — effect of GQA.** Llama-3-8B uses `n_kv_heads` = 8 (not 32), `head_dim` = 128, `n_layers` = 32. Per token:
```
2 × 32 × 8 × 128 × 2 = 131,072 bytes = 0.125 MiB ≈ 0.125 MB per token
```
— a **4× smaller** KV cache than example A purely from GQA. At seq 32,768, batch 1 that is ~4 GB instead of ~16 GB. This is why every modern long-context model uses GQA/MQA.

**Worked example C — your R1-Distill-1.5B.** Its Qwen2-1.5B backbone has `n_layers` = 28, `n_kv_heads` = 2 (aggressive GQA), `head_dim` = 128, BF16. Per token:
```
2 × 28 × 2 × 128 × 2 = 28,672 bytes ≈ 0.028 MB per token
```
At seq 8,192, batch 1: ~0.23 GB. Even at batch 32 it is ~7 GB — which, on top of 3 GB weights, comfortably fits a 24 GB card. This is precisely why you can push reasonable batch sizes for your CoT-generation runs on modest GPUs: the small model and aggressive GQA keep the KV cache tiny.

#### Why the KV cache, not the weights, limits inference batch size

Weights are a **fixed** cost paid once, shared across the whole batch. The KV cache is paid **per sequence in the batch and per token of context**. So as you raise batch size or context length to improve throughput, the KV cache grows linearly while the weights stay constant — and it is the KV cache that eventually hits the VRAM ceiling. The practical consequence: **maximum serving batch size (and thus throughput) is usually KV-cache-bound, not weight-bound**, especially for long contexts. This is the entire motivation for **paged attention** (§6.7).

### 6.5 "Does it fit?" — rules of thumb and worked verdicts

A reliable napkin procedure:

```
required ≈ weights + (training ? grads+optimizer+activations : KV_cache + activations) + overhead
```

**Inference rule of thumb:** `weights + KV_cache + ~1–2 GB overhead`, with a ~10–20% safety margin for fragmentation and transient activations.

**Training rule of thumb (Adam mixed precision):** `~16 × params bytes + activations + ~2 GB`.

Let me run the verdicts you actually care about — inference, FP16/BF16 weights, modest context — against 24/40/80 GB cards. (Remember the GB-vs-GiB haircut: an "80 GB" card reports ~81920 MiB = 80 GiB usable, of which a few GB go to overhead.)

| Model | Weights (BF16) | + KV (seq 4k, batch 1) | + overhead | Fits 24 GB? | Fits 40 GB? | Fits 80 GB? |
|---|---|---|---|---|---|---|
| 1.5B | 3 GB | +~0.1 GB | ~1.5 GB | ✅ huge headroom | ✅ | ✅ |
| 7B | 14 GB | +~2 GB | ~1.5 GB ≈ **18 GB** | ✅ (tight; little batch room) | ✅ | ✅ |
| 70B | 140 GB | +~10 GB | — | ❌ | ❌ | ❌ (needs 2× 80 GB min, realistically 4×, or 4-bit) |

**Notes on the verdicts:**
- **7B on 24 GB**: fits in BF16 for single-stream inference, but with only ~6 GB of slack you have little room to grow batch size or context. Drop to INT8 (~7 GB weights) or INT4 (~3.5 GB weights) to open up batch/context headroom.
- **70B**: 140 GB BF16 → does not fit any single current 80 GB card. Options: **INT4 quantization** (~35 GB → fits one 40/80 GB card with care), **tensor/pipeline parallelism** across 2–4 GPUs (split the weights; see §6.8), or a large-unified-memory box (e.g. a 128 GB DGX Spark / GB10 holds the FP16 weights but is bandwidth-bound).
- **70B training**: 70B × 16 B ≈ 1.12 **TB** just for params+grads+optimizer — unambiguously a multi-node, ZeRO/FSDP-sharded undertaking. Not a single-card conversation.

**Quantized-weight quick table** (weights only; add KV + overhead):

| Model | BF16 (2 B) | INT8 (1 B) | INT4 (0.5 B) |
|---|---|---|---|
| 1.5B | 3 GB | 1.5 GB | 0.75 GB |
| 7B | 14 GB | 7 GB | 3.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |

### 6.6 The caching allocator, fragmentation, and why `nvidia-smi` "lies"

This subsection resolves the single most common confusion in GPU memory debugging: **`nvidia-smi` and PyTorch disagree, and neither is wrong.**

#### Reserved vs allocated

PyTorch does **not** ask the CUDA driver for memory on every tensor — `cudaMalloc`/`cudaFree` are slow and synchronizing. Instead it uses a **caching allocator**: it grabs large blocks from the driver up front, then hands out slices to your tensors from its own pool. When a tensor is freed, the block is returned to **PyTorch's pool**, *not* to the driver — so it can be reused instantly for the next allocation. Two distinct quantities result:

- **Allocated** (`torch.cuda.memory_allocated()`): bytes currently occupied by **live tensors**.
- **Reserved** (`torch.cuda.memory_reserved()`, historically "cached"): bytes PyTorch has taken from the driver into its pool — **always ≥ allocated**.

`nvidia-smi` shows roughly the **reserved** figure plus the CUDA context (what the process holds from the driver's view), which is why it often reports far more than your tensors actually need. Freeing tensors lowers *allocated* but typically **not** what `nvidia-smi` shows, because the memory stays in PyTorch's pool. `torch.cuda.empty_cache()` returns the unused *reserved* blocks to the driver (lowering `nvidia-smi`), but does **not** free live tensors and is rarely necessary — it can even hurt performance by forcing re-allocation later.

#### Fragmentation

The pool is made of blocks. If you allocate many tensors of varied sizes and free them in a different order, free space inside the pool can become **fragmented**: total free ≥ your request, but no single contiguous block is big enough. PyTorch then either splits/finds a block, calls the driver for more, or — if the driver is also out — **OOMs even though "free" memory exists**. Variable sequence lengths (very common in LLM inference and in CoT generation where outputs vary in length) are a classic fragmentation driver.

#### `PYTORCH_CUDA_ALLOC_CONF` and expandable segments

You tune the allocator with the `PYTORCH_CUDA_ALLOC_CONF` environment variable. The two most useful knobs:

- **`expandable_segments:True`** — lets the allocator grow a segment in place rather than reserving many fixed-size segments, which **substantially reduces fragmentation**. This is the single highest-leverage setting for "I get OOM but there should be room" and for long, variable-length generation workloads:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- **`max_split_size_mb:<N>`** — prevents the allocator from splitting blocks larger than N MB to satisfy small requests, which can curb a specific fragmentation pattern (large block carved into small pieces, then a later large request fails). Tune empirically (e.g. 128 or 256) if `expandable_segments` alone doesn't resolve it.

Given your CoT-generation runs produce **variable-length** outputs, `expandable_segments:True` is worth setting by default on the RunPod/Spark inference jobs.

### 6.7 Paged KV cache (paged attention) — the inference allocator analogue

Classic serving frameworks pre-allocated one contiguous KV-cache buffer **per sequence sized for the maximum possible length**. If a request might run to 8k tokens, you reserved 8k tokens of cache even if it stops at 200 — wasting most of it (internal fragmentation), and capping batch size far below what the hardware could hold.

**PagedAttention** (the core idea behind **vLLM**) borrows operating-system **virtual memory paging**: the KV cache is split into fixed-size **blocks** (pages), allocated **on demand** as a sequence grows, and tracked by a per-sequence **block table** mapping logical positions to physical blocks. Benefits:
- **Near-zero internal fragmentation** — you only allocate blocks you actually use, so realized KV memory tracks *actual* lengths, not worst-case.
- **Much higher batch size / throughput** at fixed VRAM, because the freed waste becomes usable batch capacity.
- **Prefix sharing** — common prompt prefixes (e.g. a shared system prompt across many of your annotation/judge calls) can share the same physical blocks via copy-on-write, cutting KV memory across the batch.

The practical caveat for memory accounting: vLLM **pre-reserves a large fraction of VRAM** for its paged pool (default `gpu_memory_utilization ≈ 0.90`). So `nvidia-smi` will show vLLM holding ~90% of the card immediately — that is **by design**, not a leak. Lower `gpu_memory_utilization` if you need to co-locate other processes on the same card.

### 6.8 The full toolbox for fitting a model that "doesn't fit"

Ordered roughly from cheapest/first-to-try to most involved:

1. **Lower the batch size.** Linear reduction in activations (training) and KV cache (inference). First thing to try; costs throughput, not correctness.
2. **Shorten the context / max sequence length.** KV cache and the attention activation term scale with seq_len (the score term quadratically if not using FlashAttention). Capping `max_model_len` in vLLM directly bounds KV memory.
3. **Activation/gradient checkpointing** (training). ~3–5× less activation memory for ~⅓ more compute (§6.3).
4. **Gradient accumulation** (training). Achieve a large *effective* batch with small micro-batches — large-batch statistics at small-batch memory, no recompute.
5. **Mixed precision / lower-precision weights.** BF16 over FP32 halves weights and activations. INT8/INT4 quantization (GPTQ, AWQ, bitsandbytes, NF4) shrinks weights 2–4× for inference, and is the enabling trick for QLoRA fine-tuning.
6. **Paged KV cache** (inference). Use vLLM/paged attention to eliminate KV fragmentation and maximize batch (§6.7). Quantize the KV cache itself (FP8/INT8) for a further ~2× on the cache term.
7. **FlashAttention.** Removes the O(seq²) attention-score activation, making long context feasible; also faster. Essentially always on in modern stacks.
8. **Offloading.** Move some state to **CPU RAM** or **NVMe SSD** and stream it back when needed. **ZeRO-Offload / ZeRO-Infinity** (DeepSpeed) offload optimizer states, gradients, and even parameters to CPU/NVMe; HuggingFace `accelerate` and `device_map="auto"` can offload layers. Lets you run models far larger than VRAM at the cost of **PCIe/NVMe bandwidth** (much slower than HBM — expect large slowdowns, but "slow" beats "impossible").
9. **Sharding across GPUs (preview of the parallelism section).** When one card is not enough:
   - **FSDP (Fully Sharded Data Parallel)** / **ZeRO**: shard parameters, gradients, and optimizer states across N GPUs so each holds ~1/N of them, gathering each layer's full weights just-in-time for its forward/backward. This directly attacks the 16-B/param training wall — N GPUs give ~N× the effective parameter+optimizer capacity. ZeRO **stage 1** shards optimizer states, **stage 2** adds gradients, **stage 3** adds parameters (most memory-saving, most communication).
   - **Tensor parallelism**: split individual weight matrices across GPUs (each holds a slice), used heavily for large-model *inference* (e.g. a 70B across 4 cards).
   - **Pipeline parallelism**: place different *layers* on different GPUs.
   These trade VRAM for **inter-GPU communication** over NVLink/PCIe, which is exactly why interconnect bandwidth (NVLink vs PCIe) matters — covered in the parallelism section.

### 6.9 Reading memory in practice: `nvidia-smi` and `torch.cuda.memory_summary()`

#### `nvidia-smi`

The first tool you reach for. Run it live with `nvidia-smi` or watch it update with `nvidia-smi -l 1` (every 1 s) or `watch -n1 nvidia-smi`. Key fields:

- **`Memory-Usage`** (e.g. `38912MiB / 81920MiB`): **process-reserved** memory (the PyTorch caching-allocator reserved figure plus the CUDA context, *not* just your live tensors) over total card memory. The total here is in MiB — `81920 MiB` = 80 GiB ≈ the "80 GB" card.
- **`GPU-Util`**: percent of recent time at least one kernel was running — a coarse *activity* signal, **not** how much compute capacity you are using (a single tiny kernel reads 100%). Do not mistake it for efficiency.
- **Per-process table**: PID and its memory — essential for spotting a zombie process holding the card (common after a crashed run; `kill` the PID to reclaim VRAM that `empty_cache` cannot).
- Power draw and temperature, useful for confirming the GPU is actually working.

For richer, scriptable output use `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1`.

#### `torch.cuda` memory introspection

For *what your program thinks it is using* — the ground truth for debugging OOM:

- **`torch.cuda.memory_allocated()`** — bytes in live tensors right now.
- **`torch.cuda.memory_reserved()`** — bytes held by the caching allocator (matches `nvidia-smi` closely, modulo the CUDA-context overhead `nvidia-smi` also includes).
- **`torch.cuda.max_memory_allocated()`** — **peak** live usage since the last reset; the number you actually budget against. Wrap a training step and read this to size your batch.
- **`torch.cuda.reset_peak_memory_stats()`** — reset the high-water mark before a region you want to measure.
- **`torch.cuda.memory_summary()`** — a formatted table breaking down allocated/reserved, active blocks, and fragmentation indicators by size pool. This is the **single best built-in** for diagnosing fragmentation (look for large reserved-minus-allocated gaps) and for confirming which consumer dominates.

A minimal measurement pattern:

```python
import torch
torch.cuda.reset_peak_memory_stats()
# ... run one forward+backward (training) or one generate() (inference) ...
torch.cuda.synchronize()  # ensure async kernels finished before you read
print(f"peak allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print(f"reserved:       {torch.cuda.memory_reserved()/2**30:.2f} GiB")
print(torch.cuda.memory_summary())
```

The `synchronize()` matters: CUDA kernels are **asynchronous**, so without it you may read memory stats before the work (and its allocations) have actually happened.

For the deepest dives — *which line of code* allocated the tensor that OOM'd you — PyTorch's **memory snapshot** tooling (`torch.cuda.memory._record_memory_history()` then `torch.cuda.memory._dump_snapshot("snap.pickle")`, visualized at `pytorch.org/memory_viz`) records every allocation with its Python stack trace and renders a timeline. Reach for it when the napkin math says you should fit but you OOM anyway — it will show you the transient spike or the fragmentation pattern the summary table only hints at.

#### A debugging checklist when you OOM

1. **Do the napkin math** (§6.5). Should it fit at all? If the arithmetic says no, no flag will save you — quantize, shard, or shrink.
2. If math says yes but you OOM: read **`memory_summary()`** — is `reserved − allocated` large? → **fragmentation** → set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. Check `nvidia-smi` for a **zombie process** holding the card → kill its PID.
4. **Lower batch / context** (cheapest correctness-preserving fix), or enable **checkpointing** (training).
5. Confirm you are in **BF16, not FP32**, and that FlashAttention is active.
6. Quantize weights (INT8/INT4) and/or the KV cache if still tight.
7. Only then reach for **offloading** or **multi-GPU sharding**.

The discipline that pays off: **estimate before you launch.** Nearly every OOM is foreseeable from `params × bytes`, the 16-B/param Adam rule, and the KV-cache formula. Internalize those three, and "will it fit?" becomes a 30-second calculation rather than a 30-minute crash-and-retry.

---

## 7. Batch size: throughput, latency, convergence, and memory

Batch size is the single most consequential knob you will turn when running models on a GPU, and it is unusual because it lives simultaneously in two different worlds. To the optimization theorist, the batch size determines the variance of your gradient estimate and therefore how your loss curve behaves. To the systems engineer, the batch size determines how much parallel work you hand the GPU at once and therefore whether you are using 5% or 95% of the silicon you are paying for. These two views frequently pull in opposite directions, and learning to reason about both at the same time is what this section is about.

We will build up the concept from definitions, then develop the throughput curve (the systems story), then the latency tradeoff (the serving story), then convergence (the training story), and finally inference batching in depth (the modern LLM-serving story, where "continuous batching" has become the dominant idiom). Worked numbers appear throughout.

### 7.1 Definitions: micro-batch, mini-batch, global/effective batch

The word "batch" is overloaded, so let us pin down a precise vocabulary. A **sample** (or **example**) is one independent unit of data — for a language model, typically one sequence of tokens. A **batch** is a set of samples processed together in one forward (and, in training, one backward) pass.

When you scale across multiple GPUs and across memory limits, a single "batch size" splinters into three distinct quantities:

- **Micro-batch size** (`micro_batch`): the number of samples that physically go through one forward/backward pass on **one** GPU at one time. This is the quantity bounded by GPU memory. It is the number that determines the shapes of the tensors actually sitting in VRAM during a kernel call.

- **Mini-batch / effective / global batch size** (`global_batch`): the number of samples whose gradients are averaged together before the optimizer takes **one** weight-update step. This is the statistically meaningful batch size — the one that appears in the optimization theory, the one that sets your gradient noise. Historically "mini-batch" (as opposed to full-batch gradient descent over the entire dataset) just meant "a subset"; in modern usage mini-batch and global/effective batch are used interchangeably to mean the per-update batch.

- **Gradient accumulation steps** (`accumulation_steps`): the number of micro-batches you run forward+backward, summing (accumulating) their gradients into the `.grad` buffers, **before** calling `optimizer.step()`. This is a pure systems trick that lets you decouple the effective batch from what fits in memory.

These compose multiplicatively. On a data-parallel cluster of `data_parallel_world_size` GPUs (each holding a full copy of the model and processing different data), the relationship is:

```
effective_batch = micro_batch * accumulation_steps * data_parallel_world_size
```

#### Worked example

Suppose you want an effective batch of 512 sequences for stable training, but each GPU can only hold a micro-batch of 8 sequences at your chosen sequence length before running out of memory. You have 8 GPUs.

```
micro_batch            = 8
data_parallel_world    = 8
needed accumulation    = 512 / (8 * 8) = 512 / 64 = 8
```

So each GPU runs 8 forward/backward passes of 8 sequences each (accumulating gradients), all 8 GPUs do this in parallel on different data, then a single all-reduce averages gradients across GPUs and the optimizer steps once. You have achieved a statistically-512 update while never holding more than 8 sequences in memory on any card. The cost is wall-clock time: you did 8× more sequential forward/backward passes per update than if memory had allowed `micro_batch = 64`.

**Why gradient accumulation works and what it costs.** Gradients are linear in the loss, and the loss of a batch is the mean (or sum) of per-sample losses, so summing the gradients of several micro-batches and dividing by the total count gives *exactly* the same gradient as one big batch — there is no approximation. (One important subtlety: layers with batch-dependent statistics, classically **BatchNorm**, do *not* commute with accumulation, because their normalization statistics are computed per micro-batch; this is one reason transformers, which use **LayerNorm** / **RMSNorm** — per-sample normalizations — accumulate cleanly. Another subtlety: when accumulating, if you sum rather than average the per-micro-batch losses you must scale by `1/accumulation_steps` — equivalently, average each micro-batch loss and divide by `accumulation_steps` — so the accumulated gradient has the same magnitude as a single full-batch gradient. Also note that under data parallelism the cross-GPU all-reduce *averages* gradients across ranks, so the two reductions — sum-then-normalize within a rank, average across ranks — must be made consistent or the effective learning rate will be off by a constant factor.) The cost is purely that you serialize work that a bigger machine could have done in parallel: accumulation buys you statistical batch size at the price of wall-clock latency per update, not at the price of memory.

### 7.2 The throughput curve: why batch size and utilization are linked

To understand why throughput rises with batch size and then flattens, you need the **roofline model** of arithmetic intensity. The idea is simple: every kernel must both (a) move bytes between high-bandwidth memory (HBM/VRAM) and the compute units, and (b) perform floating-point operations (FLOPs). The GPU can do these two things at fixed peak rates — a peak compute rate and a peak memory bandwidth. For an H100 SXM5, the peak *dense* (no structured-sparsity) BF16/FP16 tensor-core rate is ~989 TFLOP/s (NVIDIA also quotes ~1,979 TFLOP/s with 2:4 structured sparsity, which only applies to suitably pruned weights — for dense training/inference math you should use the ~990 TFLOP/s figure), paired with ~3.35 TB/s of HBM3 bandwidth. These exact numbers are vendor- and SKU-dependent (the H100 PCIe and H200 differ; the H200 keeps ~989 TFLOP/s dense BF16 but raises bandwidth to ~4.8 TB/s of HBM3e), but the ratio is what matters. Which one bottlenecks you depends on the kernel's **arithmetic intensity**:

```
arithmetic intensity (AI) = FLOPs performed / bytes moved   [FLOP/byte]
```

The **roofline** says: achievable FLOP/s = min(peak_compute, AI × peak_bandwidth). There is a **ridge point** at `AI = peak_compute / peak_bandwidth`. Below it you are **memory-bound** (bandwidth is the wall); above it you are **compute-bound** (the math units are the wall). For an H100 SXM that ridge is roughly `989e12 / 3.35e12 ≈ 295 FLOP/byte` (using the dense BF16 peak) — you need to do nearly 300 floating-point operations for every byte you fetch just to keep the tensor cores busy.

#### Why tiny batches waste the GPU

Consider the core operation of a transformer layer: a matrix multiply of an activation matrix `X` (shape `[B, d]`, where `B` is the number of tokens in the batch and `d` the hidden dimension) by a weight matrix `W` (shape `[d, d]`). This computes roughly `2 · B · d · d` FLOPs (the 2 is one multiply + one add per inner-product term). The bytes moved are dominated by reading the weights once: `d · d · bytes_per_element`, plus reading/writing the activations `~2 · B · d · bytes_per_element`.

The weight matrix has to be read from HBM regardless of how many tokens you push through it. So the arithmetic intensity is roughly:

```
AI ≈ (2 · B · d · d) / ((d · d + 2·B·d) · bytes_per_element)
```

When `B` is small (say `B = 1`, a single token, as in autoregressive decode), the `d·d` weight-read term dominates the denominator and `AI ≈ (2·B·d·d)/(d·d·bytes) = 2B/bytes`. With BF16 (`bytes = 2`) and `B = 1`, `AI ≈ 1 FLOP/byte` — a factor of ~300 *below* the ridge point. You are catastrophically memory-bound: the tensor cores sit idle ~99% of the time while you wait to stream weights in from HBM. **The weight you fetched was used for exactly one token's worth of math and then thrown away.**

Now raise `B`. Each additional token reuses the *same* weight matrix already being streamed in, adding `2·d·d` FLOPs while adding only `~2·d·bytes` of extra activation traffic. Arithmetic intensity climbs roughly linearly in `B` until it crosses the ridge point, at which the kernel becomes compute-bound and you are finally using the tensor cores at full tilt. This is **weight reuse**, and it is the fundamental reason batching raises throughput: the expensive, bandwidth-limited act of loading the weights gets **amortized** over more useful arithmetic.

There are three distinct reasons small batches underutilize the GPU, and it is worth separating them:

1. **Low arithmetic intensity / memory-bound (the roofline reason, above).** The weights dominate traffic and get reused too few times.

2. **Kernel-launch and overhead floors.** Every GPU kernel (a function dispatched to run on the device) costs a few microseconds of launch overhead from the CPU, plus fixed costs like reading layer-norm parameters. At `B = 1` a transformer forward pass might be hundreds of tiny kernels each taking a few µs of *overhead* on top of a few µs of *work* — the overhead can dominate. Larger batches do more work per kernel, so the fixed per-kernel cost is amortized. (This is also why **CUDA Graphs**, which capture and replay a whole sequence of kernels with one launch, and operator fusion help small-batch decode so much.)

3. **Insufficient parallelism to fill the machine (occupancy).** A GPU has many streaming multiprocessors (**SMs** — the independent compute cores of the GPU; an H100 SXM exposes 132 enabled SMs, out of 144 on the full GH100 die). Work is dispatched in **warps** (groups of 32 threads executing in lockstep — 32 is the warp size on all current NVIDIA GPUs) and **thread blocks**. **Occupancy** is the ratio of active warps to the maximum the hardware can hold; it is how the GPU hides memory latency, by having other warps ready to run while some wait on data. A tiny batch may not generate enough independent thread blocks to cover all 132 SMs — some SMs literally have nothing to do — and may not provide enough warps per SM to hide latency. Bigger batches generate more parallel work and raise occupancy.

#### The shape of the curve

Put these together and you get the canonical **throughput-vs-batch-size curve**:

| Regime | Batch size | What limits you | Throughput (samples/s) |
|---|---|---|---|
| Overhead-bound | very small (1–4) | kernel launch, low occupancy | low, roughly flat then rising |
| Memory-bound | small–medium | HBM bandwidth (weight reuse climbing) | rising steeply, ~linear in batch |
| Compute-bound | large | tensor-core FLOP/s (saturated) | **flat** — at the roofline ceiling |
| OOM / thrash | too large | out of memory | crashes, or throughput collapses if it spills |

The key qualitative facts: throughput (samples processed per second) **rises** as you increase batch size while you are memory- or overhead-bound, then **saturates** to a flat ceiling once you become compute-bound. Past saturation, doubling the batch roughly doubles the *time per batch*, so samples/second stops improving — you are simply doing more work per step at the same FLOP/s. There is no throughput reason to go beyond the saturation point; the only reasons to do so are statistical (training wants a larger effective batch) or to slightly improve the compute/overhead ratio.

A useful diagnostic: if doubling the batch size roughly halves the per-step time *per sample* (i.e. step time grows much less than 2×), you were memory- or overhead-bound and batching is helping. If doubling the batch roughly doubles step time (per-sample time flat), you have saturated and are compute-bound.

### 7.3 Latency versus throughput: the serving tradeoff

**Throughput** is samples (or tokens) completed per unit time across the whole system. **Latency** is the wall-clock time from when *one particular request* arrives to when its result is ready. These are not the same, and batching trades one for the other.

The mechanism: to batch, you must *wait* to collect several requests, then process them together. The waiting itself adds latency. Moreover a bigger batch takes longer to compute (even if per-sample time is lower), so each request inside a large batch waits for the whole batch to finish. So:

> **Bigger batch ⇒ higher throughput (good for cost/utilization) but higher per-request latency (bad for interactivity).**

#### Worked serving example

Suppose a model takes 20 ms to process a batch of 1, and 50 ms to process a batch of 32 (the batch is only 2.5× slower despite 32× the work — classic memory-bound amortization).

- **Batch of 1:** latency = 20 ms; throughput = 1/0.020 = 50 samples/s.
- **Batch of 32:** latency ≈ 50 ms (plus any time spent waiting to fill the batch); throughput = 32/0.050 = 640 samples/s.

You bought a **12.8× throughput** improvement (640/50) at the cost of **2.5× worse latency** per request (50/20, before counting queue-fill wait). For an offline batch job (e.g. scoring a dataset, generating eval completions for your reasoning chains), latency is irrelevant and you want the biggest batch that fits and saturates compute. For an interactive chat endpoint, a 2.5× latency hit might be unacceptable and you cap the batch size or set a short "max wait to fill the batch" timeout. This is the central tension every inference server must manage, and it is governed by an explicit **SLO** (service-level objective) on latency, typically expressed as a tail percentile (e.g. "p99 time-to-first-token < 500 ms").

### 7.4 Training: convergence and the statistics of batch size

Now the optimization view. In stochastic gradient descent you estimate the true gradient (the mean over the whole dataset) using a batch. The variance of that estimate falls like `1/batch_size` — quadrupling the batch halves the gradient noise (the standard error scales like `1/sqrt(batch)`). This single fact drives everything below.

#### The linear scaling rule and warmup

If a larger batch gives a less noisy, more reliable gradient, you can afford to take a bigger step. The widely-used **linear scaling rule** (popularized by Goyal et al.'s "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017) says:

> When you multiply the batch size by `k`, multiply the learning rate by `k` as well.

Intuition: with batch `kB` you take one optimizer step where batch `B` would have taken `k` steps. If each small step moved by `η·g`, then `k` such steps move by roughly `k·η·g` (when gradients are slowly varying), so the single large-batch step should use learning rate `kη` to cover the same ground. The rule holds remarkably well over a wide range and breaks down at very large batches. (Note that the linear rule is the empirically robust one for SGD-style optimizers; for Adam-family optimizers a square-root scaling `η → √k · η` is often a better starting point because Adam already normalizes by a running estimate of the gradient's second moment. Either way, treat the scaled LR as a starting guess to be tuned, not a law.)

The catch: at the start of training the weights move fast and gradients are *not* slowly varying, so a large LR immediately is destabilizing. The fix is **learning-rate warmup**: start at a small LR and ramp linearly (or otherwise) up to the scaled target over the first few hundred or few thousand steps. Warmup and linear scaling are a package deal for large-batch training.

#### Very large batches: LARS and LAMB

Push the batch into the thousands or tens of thousands and even warmup + linear scaling fails, because different layers have wildly different gradient-to-weight ratios and a single global LR over-steps some layers while under-stepping others. **LARS** (Layer-wise Adaptive Rate Scaling, You et al. 2017) and **LAMB** (its Adam-based successor, You et al. 2019, used to cut BERT pretraining to ~76 minutes at batch sizes up to ~32k) compute a *per-layer* learning rate scaled by the **trust ratio** — the ratio of the weight norm to the (update/gradient) norm, `‖weights‖ / ‖gradient‖` — so each layer takes a step scaled to its own geometry. These are the tools that make batch sizes of 32k+ trainable.

#### The critical batch size

There is a crucial, often-misunderstood concept: the **critical batch size**. Increasing batch size reduces gradient noise, which lets you take fewer, larger steps to reach a target loss — so for a while, doubling the batch roughly halves the *number of steps* needed, and total compute stays about constant while wall-clock (steps × time/step) drops with parallelism. This is the regime of **perfect scaling**: more batch = proportionally fewer steps, free speedup if you have the hardware.

But this only holds while gradient noise is the limiting factor. Once the batch is large enough that the gradient is already near-deterministic, making it *even less noisy* buys you almost nothing — you no longer save steps proportionally, and the extra compute per step is wasted. The **critical batch size** is the crossover: the largest batch that still buys you (near-)linear reduction in steps. McCandlish et al. (2018, "An Empirical Model of Large-Batch Training") formalized this and showed the critical batch size can be *predicted* from the **gradient noise scale**, and that it tends to *grow* during training (later in training the loss landscape tolerates — and benefits from — bigger batches).

Practical consequences:

- **Below the critical batch size:** scaling up batch is nearly free speedup (fewer steps, same final loss). Do it if you have GPUs.
- **Above the critical batch size:** you pay more compute per step for diminishing returns in convergence. You are now spending FLOPs to buy wall-clock time, not efficiency.

#### The large-batch generalization discussion

A long-running empirical observation (Keskar et al. 2016) is that *very* large batches can converge to "sharp" minima that generalize slightly worse than the "flat" minima found by small, noisier batches — the gradient noise of small batches acts as a mild regularizer. This is real but easy to over-state: much of the apparent generalization gap closes once you tune the LR properly (linear scaling + warmup) and train for enough steps. The modern practical stance: batch size is *primarily* a systems-and-speed knob you push as high as the critical batch size allows; it is *secondarily* a weak regularizer. Treat the LR (and schedule) as the thing you must retune whenever you change batch size, and do not chase tiny batches for generalization unless you have measured a real benefit.

The summary that ties this subsection together: **batch size is both an optimization knob (it sets gradient noise, interacts with LR, and has a critical value beyond which it wastes compute) and a systems knob (it sets memory use and GPU utilization). You must satisfy both constraints at once** — the effective batch must be statistically sound, and you realize it on the hardware via micro-batch × accumulation × data-parallel without ever exceeding memory.

### 7.5 Inference batching in depth

Inference deserves its own extended treatment because modern LLM serving has evolved a specialized and initially counterintuitive set of techniques. The key to all of it is recognizing that autoregressive generation has **two phases with completely different compute profiles**.

#### Prefill versus decode

When you send a prompt to an LLM and it generates a response, the work splits in two:

- **Prefill** (a.k.a. the prompt/encoding phase): the model processes all `P` prompt tokens *at once*, in a single forward pass, to build the **KV cache** (the stored keys and values for every layer and every prompt token; see the KV-cache section). Because all `P` tokens go through together, prefill is a big matrix-multiply with batch dimension `P` — it has **high arithmetic intensity and is compute-bound** even for a single request, once `P` is more than a couple hundred tokens. A 2000-token prompt is already 2000 "rows" of work per matmul.

- **Decode** (a.k.a. the generation/autoregressive phase): the model generates output tokens **one at a time**, because each new token depends on the previous one it just emitted. Each decode step is a forward pass with batch dimension *one token per sequence*. As we computed in §7.2, a single-token matmul against the full weight matrix has arithmetic intensity ≈ 1 FLOP/byte — **decode is severely memory-bound.** Worse, decode must also re-read the entire growing KV cache from HBM every single step.

This asymmetry is the whole story of LLM inference performance. Prefill is a brief, compute-bound burst; decode is a long, memory-bound grind, and for any response of meaningful length **decode dominates total time and total cost.** Your "tokens per second" is overwhelmingly set by decode efficiency.

#### Why batching is the cure for decode

Recall the roofline argument. A single decoding sequence pushes one token through the weights and is ~300× below the compute ridge — the tensor cores are ~99% idle, and the time per step is set entirely by how fast you can stream the weights (and KV cache) out of HBM.

Here is the magic: **streaming the weights in from HBM happens once per step regardless of how many sequences you decode in parallel.** If you batch 64 different sequences and advance all of them by one token in the same forward pass, you read each weight matrix *once* and use it for 64 tokens' worth of math instead of 1. Arithmetic intensity rises ~64×, you climb up the roofline toward the compute ceiling, and your *aggregate* tokens/second rises almost in proportion to the batch — until either you saturate compute or you run out of memory for KV caches.

This is the central, slightly counterintuitive fact of LLM serving: **in the decode phase, batching many sequences together is essentially free throughput.** Because you were memory-bound and the weights were being re-read anyway, adding more concurrent sequences costs almost no extra time per step while multiplying the useful work. Decode batching is where serving systems reclaim the GPU utilization that single-stream generation throws away.

#### Worked decode-batching example

Take a 7B model in BF16. Weights are `7e9 × 2 bytes = 14 GB`. Suppose HBM bandwidth is ~2 TB/s (an A100-80GB-class figure — the SXM A100 is ~2.04 TB/s; an H100 at ~3.35 TB/s or H200 at ~4.8 TB/s is correspondingly faster). To do one decode step you must stream all 14 GB of weights through the compute units at least once:

```
time to read weights ≈ 14e9 bytes / 2e12 bytes/s ≈ 7 ms   (memory-bound floor, ignoring KV cache)
```

- **Batch 1:** ~7 ms per step → ~1 token / 7 ms ≈ **143 tokens/s** for that one user. The tensor cores were nearly idle the whole 7 ms.
- **Batch 32:** the weight read is *still ~7 ms* (same 14 GB, read once), now producing 32 tokens → 32 / 0.007 ≈ **4,570 tokens/s aggregate** (~143 tokens/s *per user*, but 32 users served). Throughput rose ~32× at almost no latency cost — until KV-cache traffic and eventually compute start to matter.

The per-user experience barely changed; the *system* throughput multiplied. (In reality the KV-cache reads grow with batch and sequence length and eventually you do become compute- or KV-bandwidth-bound, so the speedup tapers — but the regime where batching is near-free is wide and is exactly where production serving operates.)

#### Static batching and its waste

The naive way to batch inference is **static batching**: collect `N` requests, run them all through prefill together, then decode all `N` in lockstep until *every* sequence has finished, then return the whole batch and accept the next group. The fatal flaw: sequences finish at *different* times (one user asks for a 10-token answer, another for a 2000-token essay). In static batching, the short sequences sit idle — their "slot" in the batch is occupied but doing nothing, padded out — until the longest sequence in the batch completes. With high variance in output lengths this wastes an enormous fraction of the GPU: a slot that finished at token 10 contributes nothing for the remaining 1990 steps but still can't be reused. It also adds head-of-line latency: a request that arrives just after a batch starts must wait for the entire batch to drain.

#### Dynamic / continuous (in-flight) batching

The modern solution, introduced as **continuous batching** (a.k.a. **in-flight batching**; the Orca paper, Yu et al. 2022, is the canonical reference, and **vLLM** and **TensorRT-LLM** are the widely-used implementations), operates at the granularity of a *single decode step* (technically a single iteration of the scheduler) rather than a whole request:

- The server maintains a pool of *active* sequences being decoded.
- At **every** decode step, it forms a batch from whatever sequences are currently active and advances each by one token.
- The instant a sequence emits its end-of-sequence token (or hits its length limit), it is **evicted** from the batch and its slot is **immediately** filled by a waiting request (which gets prefilled and joins the decode pool).

Because slots are recycled continuously instead of waiting for the whole batch to finish, the GPU stays full: finished short sequences are replaced by fresh work within one step. This typically yields several-fold higher throughput than static batching on realistic, variable-length workloads, and it slashes queueing latency because new requests join almost immediately rather than waiting for a batch boundary. Most servers also handle **prefill** intelligently — interleaving or "chunking" prefill of newcomers with ongoing decode (**chunked prefill**) so that a large incoming prompt does not stall the decode of everyone already in flight.

#### PagedAttention and why memory is the real limit

Once you are doing continuous batching, the binding constraint is no longer compute — it is **how many KV caches you can fit in VRAM**, because each concurrent sequence needs its own KV cache and that cache grows with its sequence length. The number of sequences you can batch (and thus your throughput) is set by KV-cache memory.

Classic implementations pre-allocated a contiguous KV-cache buffer sized for the *maximum* possible sequence length per request. That is hugely wasteful: a request that ends up only 50 tokens long still reserved space for, say, 4096, and the unused tail is **internal fragmentation** that no other request can use. **PagedAttention** (the technique at the heart of **vLLM**, Kwon et al. 2023) borrows the operating-system idea of **virtual memory and paging**: the KV cache is split into fixed-size **blocks** (pages), each holding the KV for a small fixed number of tokens (vLLM's default block size is 16). A sequence's cache is a *list of blocks* that need not be contiguous in memory, allocated lazily one block at a time as the sequence grows, and freed block-by-block when it ends. This nearly eliminates the fragmentation waste, lets the server pack far more sequences into the same VRAM, and therefore raises the achievable batch size and throughput. It also enables cheap **prefix sharing** (two requests with a common prompt prefix can share the same physical KV blocks via copy-on-write). The KV-cache section covers the memory math in detail; the point here is that **PagedAttention is the memory technology that makes large-batch continuous decoding feasible**, and in modern serving your batch size is dictated by KV-cache capacity far more than by compute.

### 7.6 Practical: choosing batch size and finding the maximum that fits

#### Finding the largest batch that fits (OOM probing / binary search)

GPU memory consumption is hard to predict exactly (it depends on activation memory, fragmentation, framework overhead, cuDNN/cuBLAS workspace, optimizer state, etc.), so the pragmatic approach is empirical. You **binary-search** the maximum batch size:

1. Pick a batch you're sure fits (e.g. 1) and one you're sure doesn't (e.g. 1024).
2. Try the midpoint. If it runs a few steps without an out-of-memory (**OOM**) error, it fits; raise the lower bound. If it OOMs, lower the upper bound.
3. Repeat until the bounds meet. Then back off ~10–20% for safety headroom (so a slightly longer sequence or a fragmentation spike doesn't OOM mid-run).

In a few `log2` iterations you converge (a 1–1024 range is at most ~10 probes). Always test at your *longest* expected sequence length, because activation and KV memory scale with sequence length and the worst case is what OOMs you. For training, remember you can fix the effective batch at the statistically-correct value and absorb a small micro-batch via gradient accumulation (§7.1) — so the binary search is over the *micro-batch*, and accumulation handles the rest.

#### The interaction with sequence length

Batch size and sequence length trade off against each other for a fixed memory budget, but **not symmetrically**, because attention cost and KV cache scale differently from the rest of the model:

- **Activation / KV memory** scales roughly with `batch × sequence_length` (linear in each). The KV cache for one sequence is `2 (K and V) × num_layers × seq_len × kv_dim × bytes`, where `kv_dim` is the total dimension across key/value heads (for multi-head attention `kv_dim = num_heads × head_dim`; with grouped-query or multi-query attention it is the much smaller `num_kv_heads × head_dim`) — directly linear in sequence length.
- **Attention compute** scales with `batch × sequence_length²` (the `seq_len²` from every token attending to every other). So doubling sequence length quadruples attention FLOPs while only doubling most other costs. (Note this quadratic term is attention-specific; the large feed-forward and projection matmuls remain linear in sequence length, so which term dominates depends on model size and how long the sequence is.)

A useful mental model many practitioners use is to hold the **total token count** `batch × seq_len` roughly constant when filling memory — but you must lower that token budget for very long sequences because the quadratic attention term and the linear-in-seq-len KV cache eat extra memory and compute. Concretely: a budget that comfortably runs batch 32 at 512 tokens may OOM at batch 8 at 4096 tokens even though `batch × seq_len` is identical, because the 4096-token case has 8× the per-sequence KV cache and 8× the per-sequence attention work.

#### "Tokens per second" is the real inference metric

For training, you care about samples/second (or, better, **tokens/second**, since sequences vary in length) and ultimately **time-to-target-loss**. For inference, the headline metrics are explicitly token-based, and you should report and optimize them directly:

- **Throughput: total tokens/second** across all concurrent sequences — the number that determines your cost per million tokens and how many users you can serve per GPU. Continuous batching is about maximizing *this*.
- **Per-user decode speed: tokens/second per sequence** (the inter-token latency, often reported as **TPOT**, time-per-output-token, or equivalently **ITL**, inter-token latency) — what determines whether the stream "feels" fast to one user. Batching barely changes this until you saturate, which is exactly why batching is such a good deal.
- **Time-to-first-token (TTFT)**: how long until the first output token appears — dominated by **prefill** time (and any queue wait). Long prompts and busy servers inflate it; chunked prefill exists to protect it.

The reason "samples/second" is the wrong primary inference unit is that requests have wildly different lengths: one 2000-token generation is 2000× the work of a 1-token one but counts as the same "sample." Token-level accounting is the only honest measure, and it is what every serious serving benchmark (and your cloud bill) is denominated in.

#### A practical recipe to pick batch size

To close, here is the decision procedure that falls out of everything above:

- **Offline / batch inference (your eval and chain-generation jobs):** latency does not matter. Use the largest batch that fits at your longest sequence length (binary-search it), so you saturate compute and minimize total wall-clock and cost. If using vLLM, just feed it all your prompts and let continuous batching + PagedAttention manage the batch automatically — it will pack the GPU far better than a hand-set static batch.
- **Interactive serving:** set a latency SLO (e.g. p99 TTFT and TPOT targets), then raise the maximum batch / number of concurrent sequences as high as you can while staying under the SLO. Use continuous batching; tune `max_num_seqs` and the KV-cache memory fraction (in vLLM, `gpu_memory_utilization`); use chunked prefill to protect TTFT.
- **Training:** first fix the **effective batch** from optimization considerations (near the critical batch size; retune LR with the linear scaling rule + warmup whenever you change it). Then realize it on the hardware as `micro_batch × accumulation_steps × data_parallel_world_size`, binary-searching the micro-batch to the largest that fits at your sequence length and backing off for headroom. Treat accumulation as the free knob that decouples statistics from memory.

The unifying thread of this entire section: **small batches waste the GPU because they are memory- and overhead-bound; batching amortizes weight loading and fills the machine, raising throughput until you hit the compute roofline or a memory wall; but every increase in batch costs latency (serving) or can cross the critical batch size and waste compute (training).** Mastering batch size is mastering exactly where, for your specific workload, those curves cross.

---

## 8. Distributed training: multi-GPU and multi-node parallelism

When a model and its training workload outgrow a single GPU, you must spread the work across many GPUs — possibly across many physical machines ("nodes"). This section builds up the machinery for doing so, starting from *why* you would distribute at all, through the low-level communication primitives that make it possible, the interconnect hardware that determines their cost, and finally the family of *parallelism strategies* (data, tensor, pipeline, sequence, expert) and how they compose into the "3D" configurations used to train frontier models. The goal is that by the end you can read a training config, predict roughly where its bottleneck will be, and choose a sensible mapping of strategy onto hardware.

### 8.1 Why distribute at all: two distinct problems

It is essential to separate two reasons for going multi-GPU, because they call for different solutions and they trade off against each other.

**Problem A — throughput (speed).** Your model fits comfortably on one GPU, but a single GPU processes too few tokens per second. You want to finish in days instead of months. The natural fix is to process more data in parallel: put a copy of the model on each of *N* GPUs, feed each a different slice of the batch, and you get roughly *N×* the token throughput. This is **data parallelism**, and it does not reduce per-GPU memory at all — every GPU still holds the whole model.

**Problem B — capacity (memory).** Your model (plus everything training needs alongside the weights) does not fit in one GPU's memory at all. No amount of "run more copies" helps; you must *split the model itself* across devices so that each GPU stores only a fraction. This is the domain of **model parallelism** (tensor, pipeline) and of **sharding** (ZeRO/FSDP).

To see why Problem B is so common, you must account for the full **training memory budget**, which is far larger than the weights alone. Consider a model with *P* parameters trained in mixed precision with the Adam optimizer — the standard recipe.

| Memory component | Bytes per parameter | For *P* = 7e9 |
|---|---|---|
| FP16/BF16 weights (used in forward/backward) | 2 | 14 GB |
| FP16/BF16 gradients | 2 | 14 GB |
| FP32 master copy of weights (optimizer) | 4 | 28 GB |
| Adam first moment *m* (FP32) | 4 | 28 GB |
| Adam second moment *v* (FP32) | 4 | 28 GB |
| **Subtotal (model + optimizer states)** | **16** | **112 GB** |

That is the famous "**16 bytes per parameter**" figure for mixed-precision Adam training (2 + 2 + 4 + 4 + 4). (Some implementations also keep an FP32 copy of the gradients, pushing this toward 18 bytes/param; 16 is the canonical baseline.) For a 7-billion-parameter model that is **112 GB before you store a single activation** — already more than the 80 GB of an A100 or an 80 GB H100 (or even the 141 GB of an H200 once activations are added at scale). *Activations* (the intermediate tensors saved during the forward pass so the backward pass can compute gradients) are stored on top of this, and they scale with batch size and sequence length; for long-context training they can rival or exceed the 112 GB.

The contrast in scaling is the key intuition:

- **Inference** memory ≈ weights + KV cache. A 7B model in BF16 needs ~14 GB of weights and fits on one 24 GB card. This is why you, running *inference* first, may never hit Problem B.
- **Training** memory ≈ 8× the weights (16 vs 2 bytes/param) **plus** activations. The same 7B model needs >112 GB — guaranteeing multi-GPU.

So: distribute for **speed** → data parallelism; distribute for **capacity** → split or shard the model. Real large runs do both at once.

### 8.2 Collective communication primitives

All multi-GPU training is, underneath, a sequence of **collective operations** — communication patterns in which a *group* of processes (here, one process per GPU, each called a **rank**, numbered 0…N−1) cooperate to move and combine data. They are called "collective" (as opposed to point-to-point send/recv) because every rank in the group participates in a single logical operation. The vocabulary below is borrowed from MPI (the Message Passing Interface, the classic HPC standard) and implemented for GPUs by NVIDIA's **NCCL** ("Nickel", the NVIDIA Collective Communications Library).

Throughout, let *N* = number of ranks and *V* = the size in bytes of the data buffer on one rank.

#### The five primitives

**Broadcast.** One rank (the "root") holds a buffer; afterwards all *N* ranks hold an identical copy. Used to distribute initial weights so every replica starts identical.
*Before:* rank 0 has `[a b c d]`. *After:* every rank has `[a b c d]`.

**Reduce.** Every rank contributes a buffer; they are combined element-wise with an operator (usually **sum**, sometimes max/min/mean) and the single result lands on the root rank.
*Before:* rank *i* has vector *xᵢ*. *After:* root has Σᵢ *xᵢ*.

**All-reduce.** Like reduce, but the summed result is delivered to *every* rank, not just the root. This is **the workhorse of data-parallel training**: each GPU computes gradients on its own data slice, and an all-reduce produces the summed (then averaged) gradient on every GPU so they can all take an identical optimizer step.
*Before:* rank *i* has gradient *gᵢ*. *After:* every rank has Σᵢ *gᵢ*.

**Reduce-scatter.** Combine element-wise like a reduce, but instead of one rank getting the whole result, the result vector is *partitioned* into *N* chunks and rank *i* receives only chunk *i*. So each rank ends up with the reduced value of one slice. This is the first half of an all-reduce, and a building block of ZeRO/FSDP.
*Before:* each rank has the full vector of length *L*. *After:* rank *i* holds Σ over ranks of the *i*-th chunk (length *L/N*).

**All-gather.** The inverse of scatter: each rank starts with its own chunk; afterwards every rank holds the full concatenation of all chunks. Used in FSDP to reconstruct a full weight tensor from shards just before it is needed.
*Before:* rank *i* has chunk *cᵢ* (length *L/N*). *After:* every rank has `[c₀ c₁ … c_{N−1}]` (length *L*).

**All-to-all.** A "transpose" of data across ranks: each rank holds *N* chunks, one destined for each rank; afterwards rank *j* has collected the *j*-th chunk from every rank. Every rank both sends to and receives from every other rank. This is the heavy primitive behind **Mixture-of-Experts** routing (tokens fly to whichever GPU holds their chosen expert) and behind **sequence parallelism**.

The fundamental identity to remember: **all-reduce = reduce-scatter + all-gather**. You first reduce-scatter so each rank owns the sum of one slice, then all-gather so everyone has all slices. This decomposition is exactly why the ring algorithm below is efficient, and why ZeRO can "split" the all-reduce into its two halves and interleave them with computation.

#### Ring vs tree all-reduce, and the cost model

How fast is a collective? Model communication cost with two terms: a **latency** term α (the fixed cost to send any message, dominated by per-hop overhead, in seconds) and a **bandwidth** term — the time to push the bytes, equal to (bytes moved on the slowest link) / β, where β is link bandwidth in bytes/s. A collective's cost is roughly:

> time ≈ (number of communication steps) × α + (total bytes traversing the bottleneck link per rank) / β

**Ring all-reduce** is the classic bandwidth-optimal algorithm and the NCCL default on many topologies. Arrange the *N* ranks in a logical ring. Split each rank's buffer of *V* bytes into *N* chunks. The reduce-scatter phase runs *N−1* steps: at each step every rank simultaneously sends one chunk to its right neighbour and receives one from its left, adding the received chunk into its local copy. After *N−1* such steps each rank owns the fully-reduced value of one chunk. The all-gather phase is another *N−1* steps that circulate those finished chunks around the ring until everyone has all of them.

The beautiful property: at every step, **every link in the ring is busy in both directions**, and each rank sends only *V/N* bytes per step. Total bytes sent per rank across both phases:

> 2 × (N−1) × (V/N) ≈ 2V as N grows.

This is independent of *N* — the bandwidth cost of a ring all-reduce is ~**2V regardless of how many GPUs** you add (each moves about 2× its buffer over the wire). That is why data parallelism scales so well on bandwidth. The downside is the **latency** term: 2(N−1) sequential steps, so latency grows linearly with *N*. For small messages or large *N*, this hurts.

**Tree (or "double-binary-tree") all-reduce** trades bandwidth for latency. Combine values up a binary tree to the root (reduce) then broadcast down — only ~2·log₂(N) steps, so the latency term grows logarithmically rather than linearly. NCCL actually implements double binary trees so that all links stay busy. NCCL automatically picks ring vs tree (and tunes chunk sizes) based on message size, GPU count, and measured topology: trees win for **small messages across many nodes** (latency-bound), rings win for **large messages** (bandwidth-bound, e.g. gradient all-reduce of a big model).

A worked bandwidth example. Suppose you all-reduce the gradients of a 7B model in FP16: *V* = 7e9 × 2 = 14e9 bytes = 14 GB. Over **NVLink**, taking ~300 GB/s as an *achievable effective* per-GPU rate for a ring all-reduce (well below the ~900 GB/s aggregate bidirectional figure an H100's links can deliver in total — collectives never hit the theoretical peak), the ~2V = 28 GB of traffic per GPU takes ≈ 28/300 ≈ **93 ms**. Over a **200 Gb/s InfiniBand** link (HDR, = 25 GB/s), the same costs ≈ 28/25 ≈ **1.1 s** — more than 10× slower. This single comparison is why *where* you place each parallelism dimension (fast NVLink inside a node vs slower network between nodes) dominates training efficiency. Note also: this gradient all-reduce can be **overlapped** with the backward pass (see §8.4), so its wall-clock cost is partly hidden — but only if the link is fast enough to finish before compute does.

### 8.3 Interconnects and topology

The cost model above has β (link bandwidth) and α (latency) as inputs. The physical interconnect sets those numbers, and they differ by *orders of magnitude* depending on whether two GPUs are in the same box or different boxes. This hierarchy is the single most important fact for choosing a parallelism layout.

#### Intra-node: NVLink and NVSwitch

Within one server, GPUs can be wired together with **NVLink**, NVIDIA's high-bandwidth GPU-to-GPU interconnect, which is far faster than the PCIe bus that connects a GPU to the CPU. **NVSwitch** is an on-board crossbar switch that lets every GPU in the node talk to every other at full NVLink bandwidth simultaneously (an "all-to-all" topology), rather than only to immediate neighbours.

| Link (representative; generation-dependent) | Approx. per-GPU bandwidth |
|---|---|
| PCIe 4.0 ×16 | ~32 GB/s per direction (~64 GB/s bidirectional aggregate) |
| PCIe 5.0 ×16 | ~64 GB/s per direction (~128 GB/s bidirectional aggregate) |
| NVLink 3.0 (A100, 12 links, via NVSwitch) | ~600 GB/s aggregate bidirectional per GPU |
| NVLink 4.0 (H100, 18 links, via NVSwitch) | ~900 GB/s aggregate bidirectional per GPU |

These figures are vendor- and generation-dependent (the NVLink numbers are the bidirectional *aggregate* across all of a GPU's links, summing both directions; newer parts like the Blackwell B200 push NVLink 5.0 to ~1.8 TB/s); treat them as representative orders of magnitude. The takeaway is stable across generations: **NVLink is ~10–15× the bandwidth of PCIe**, and an NVSwitch fabric gives uniform full-bandwidth all-to-all among the typically 8 GPUs in a node. This is why the most communication-hungry parallelism (tensor parallelism, §8.6) is kept *inside* a node.

#### Inter-node: InfiniBand, RoCE, Ethernet

To connect *nodes*, you leave the chassis and use a network fabric. The options:

- **InfiniBand (IB)** — a purpose-built HPC network with very low latency and hardware support for RDMA (below). Typical modern links are **200–400 Gb/s** per port (HDR is 200 Gb/s, NDR is 400 Gb/s). This is the gold standard for large training clusters.
- **RoCE** (RDMA over Converged Ethernet) — RDMA semantics carried over Ethernet hardware; cheaper and more commodity than IB, slightly higher latency, needs careful network configuration (priority flow control / ECN) to avoid congestion.
- **Plain TCP/IP Ethernet** — the fallback. Works everywhere, but higher latency and CPU overhead; fine for pure data parallelism of small models, painful for tightly-coupled tensor/pipeline parallelism.

Note the unit convention: networks are quoted in **gigabits per second (Gb/s)**; divide by 8 for **gigabytes per second (GB/s)**. So 400 Gb/s = 50 GB/s — still **~18× slower than a single H100's ~900 GB/s NVLink**. The intra-node vs inter-node bandwidth gap is the crux of distributed-training design.

#### GPUDirect RDMA

**RDMA** (Remote Direct Memory Access) lets one machine's network card read or write another machine's memory *without involving either CPU* — no syscall, no buffer copy through the OS, very low latency. **GPUDirect RDMA** extends this so the network card reads/writes **GPU** memory directly, bypassing the CPU and system RAM entirely: gradients can flow GPU→NIC→network→NIC→GPU. NCCL uses GPUDirect RDMA (and the intra-node analogue, GPUDirect P2P over NVLink) automatically when the hardware and drivers support it. Without it, every cross-node transfer would bounce through host memory, adding latency and stealing CPU/PCIe bandwidth.

#### Why topology dictates strategy

Putting it together, here is the rule that governs every layout decision:

> Place the parallelism dimensions that communicate *most frequently and most heavily* on the *fastest* links.

- **Tensor parallelism** all-reduces twice **per layer** — enormous, latency-sensitive traffic → must live on **NVLink within a node** (typically ≤8 GPUs).
- **Pipeline parallelism** sends only the activations at stage *boundaries*, a handful of times per micro-batch → tolerates **inter-node** links.
- **Data parallelism / FSDP** communicates once per step (gradient all-reduce) or per layer (FSDP all-gather) but can **overlap** with compute → can span **many nodes** over InfiniBand.

This is exactly the ordering you will see in §8.9's 3D layouts.

### 8.4 Data parallelism and PyTorch DDP

**Data parallelism (DP)** is the simplest and most common strategy and the answer to Problem A (throughput). The recipe:

1. **Replicate** the full model onto each of *N* GPUs; all replicas start with identical weights (a broadcast from rank 0).
2. **Shard the data**: split each global mini-batch into *N* equal "local batches", one per GPU. If each GPU processes a local batch of 8 and *N* = 16, the **effective global batch size is 128**.
3. Each GPU runs the **forward and backward pass independently** on its local batch, producing local gradients *gᵢ*.
4. **All-reduce** the gradients across all GPUs and average: ḡ = (1/N) Σᵢ *gᵢ*. Now every GPU holds the same averaged gradient.
5. Each GPU applies the **same optimizer step** with ḡ. Because they started identical and applied an identical update, the replicas **stay bit-for-bit in sync** — this is the **synchronous SGD** semantics.

The mathematical point: averaging the gradients of *N* local batches is *exactly* the gradient of one big batch of size *N*×(local), provided the per-example loss is the mean over the batch. So synchronous data parallelism is mathematically equivalent to training with a larger batch on one (hypothetical) giant GPU — the result does not depend on *N*, only the effective batch size does.

#### PyTorch DistributedDataParallel (DDP)

In PyTorch, the production implementation is **`DistributedDataParallel`**. You launch *N* processes (e.g. via `torchrun --nproc_per_node=8`), one per GPU; each process holds one replica. DDP registers **autograd hooks** on the parameters so that the gradient all-reduce starts automatically during the backward pass.

Two engineering ideas make DDP efficient:

**Gradient bucketing.** Issuing a separate tiny all-reduce for each of the thousands of parameter tensors would be dominated by the per-message latency term α. Instead DDP groups parameters into **buckets** (default ~25 MB). When all gradients in a bucket are ready, that whole bucket is all-reduced as one large message — amortizing latency and getting good bandwidth utilization.

**Compute/communication overlap.** Backpropagation computes gradients **layer by layer, from output back to input**. The gradients of the *last* layers are ready first — while the backward pass is still grinding through the *earlier* layers. DDP exploits this: as soon as a bucket (typically of late-layer params) fills, its all-reduce is launched *asynchronously on a separate CUDA/NCCL stream* and proceeds **in parallel with the ongoing backward compute** of earlier layers. Ideally, by the time backward finishes, almost all gradient communication is already done, so the all-reduce is nearly "free" in wall-clock terms. This overlap is the single most important performance feature of DDP. (For it to actually hide the communication, the interconnect must be fast enough to drain the buckets before compute completes — recall the 93 ms NVLink vs 1.1 s IB contrast in §8.2.)

A subtlety: DDP requires gradients to be ready predictably, so unused parameters or control-flow that skips some layers need `find_unused_parameters=True`, which costs an extra traversal. And because every step ends in a synchronizing all-reduce, **the slowest GPU sets the pace** — the straggler problem of §8.10.

### 8.5 ZeRO and FSDP: sharding the redundant state

Plain DDP has a glaring inefficiency for large models: **every GPU stores a full copy of the 16-bytes-per-parameter optimizer state**, even though, after the gradient all-reduce, all replicas compute the *identical* optimizer step. That redundancy is pure waste. **ZeRO** (the Zero Redundancy Optimizer, from Microsoft's DeepSpeed) and its PyTorch-native sibling **FSDP** (Fully Sharded Data Parallel) eliminate it by **partitioning the training state across the data-parallel ranks** instead of replicating it — turning a data-parallel group into a way of also solving Problem B (capacity).

The state is sharded in three escalating stages. Let *N* be the number of data-parallel ranks and recall the per-parameter budget: 2 (BF16 weight) + 2 (grad) + 12 (FP32 master + Adam *m* + *v*) = 16 bytes.

| Stage | What is sharded | Per-GPU state (bytes/param) | 7B on 8 GPUs |
|---|---|---|---|
| Baseline DDP | nothing (all replicated) | 16 | 112 GB each |
| **ZeRO-1** | optimizer states (the 12) | 4 + 12/N | (4 + 1.5)·7e9 ≈ 38.5 GB |
| **ZeRO-2** | + gradients (the 2) | 2 + 14/N | (2 + 1.75)·7e9 ≈ 26.25 GB |
| **ZeRO-3 / FSDP** | + parameters (the last 2) | 16/N | 16/8·7e9 = 14 GB |

**ZeRO-1** shards only the optimizer states (FP32 master weights + Adam moments). Each GPU still has full BF16 weights and full gradients, runs a normal forward/backward, but after the gradient **reduce-scatter** each rank holds the gradient slice for *only its* parameter shard, updates only that shard's optimizer state, and then an **all-gather** of the updated weights restores full weights everywhere. Communication volume is essentially the same as DDP's all-reduce (recall all-reduce = reduce-scatter + all-gather).

**ZeRO-2** additionally shards the gradients: a rank only ever *keeps* the gradient slice it is responsible for, freeing the 2 bytes/param of full-gradient storage.

**ZeRO-3 / FSDP** shards the **parameters themselves**. No GPU holds the full weight tensor at rest — each holds a 1/N slice of every layer. This achieves the dream scaling: per-GPU memory ≈ (total state)/N, so you can fit arbitrarily large models by adding GPUs. The price is extra communication, paid **just in time**.

#### How FSDP gathers parameters just-in-time

The mechanism is the crux of FSDP. The model is divided into **units** (e.g. each transformer block is one FSDP unit). At rest, each unit's parameters live sharded across the *N* ranks. During execution:

1. **Forward, entering a unit:** the ranks **all-gather** that unit's parameter shards so every rank momentarily holds the **full** weights of just that one layer.
2. Compute the layer's forward pass.
3. **Immediately free** the gathered full weights, keeping only the local shard. Memory for full weights is thus held for *one layer at a time*, not the whole model.
4. **Backward, entering a unit:** **all-gather** the weights again (they were freed), compute gradients, then **reduce-scatter** the gradients so each rank keeps only its shard's gradient. Free the full weights again.

So FSDP trades **memory for communication**: where DDP communicates ~2V once per step, FSDP performs an all-gather in the forward, an all-gather in the backward, and a reduce-scatter in the backward **per unit** — roughly **1.5× the communication volume** of DDP (the three collectives each move ~V/N per rank per step, versus DDP's two-phase ~2V all-reduce; informally people quote "about 50% more traffic," though the exact ratio depends on bucketing and prefetch). As before, these gathers are **prefetched and overlapped**: while computing layer *k*, FSDP is already all-gathering layer *k+1*. On fast NVLink this overlap hides most of the cost; on slow inter-node links the extra traffic can become the bottleneck, which is why FSDP/ZeRO-3 is happiest within high-bandwidth islands or combined with other strategies.

Two practical knobs: **activation checkpointing** (a.k.a. gradient checkpointing) — discard most activations in the forward pass and *recompute* them in the backward pass, trading roughly one extra forward pass (~33% more compute, often cited loosely as "~30%") for a large drop in activation memory; and **CPU offload** (ZeRO-Infinity) — park optimizer states or even parameters in CPU RAM or NVMe and stream them in, trading PCIe bandwidth for the ability to fit truly enormous models on few GPUs.

### 8.6 Tensor (intra-layer) parallelism

Data parallelism and FSDP still execute each *individual operation* on one GPU. **Tensor parallelism (TP)** — also called intra-layer model parallelism, pioneered by NVIDIA's **Megatron-LM** — instead splits a *single* matmul across GPUs, so that one layer's computation is genuinely co-executed by several devices. It is the tool for layers too big (or activations too big) to compute on one GPU, and for cutting latency on the critical path.

#### Splitting a matmul

Consider an MLP block, the heart of a transformer feed-forward layer: *Y* = GeLU(*X A*) then *Z* = *Y B*, where *A* and *B* are big weight matrices. Megatron splits them cleverly to **minimise synchronization**:

- Split *A* **column-wise** across *G* GPUs: *A* = [*A₁ A₂ … A_G*]. Each GPU computes *Y_i* = GeLU(*X Aᵢ*) using the **full input *X*** but producing only its **slice of columns** of *Y*. GeLU is element-wise, so no communication is needed across the nonlinearity — this is the key trick.
- Split *B* **row-wise**, matching: *B* = [*B₁; B₂; … ; B_G*]. Each GPU computes the partial product *Y_i B_i*. The full output is *Z* = Σᵢ *Y_i B_i* — a sum of the per-GPU partials, so the GPUs do **one all-reduce** to add their partials together.

The result: the whole MLP block needs exactly **one all-reduce in the forward pass** (and, symmetrically, one in the backward pass). Attention is split analogously by putting **different attention heads on different GPUs** (heads are independent until the output projection), again costing one all-reduce per direction.

So a transformer layer under TP costs roughly **two all-reduces per layer per direction** (one for attention, one for MLP). For a model with *L* = 32 layers, that is on the order of **64 all-reduces of full activation tensors in the forward pass** (and another ~64 in the backward), i.e. ~128 per full step.

#### Why TP must stay on NVLink within a node

Each of those all-reduces moves a tensor the size of the layer's activations — for batch *B*, sequence *S*, hidden *H*, that is *B·S·H* elements — and it sits **directly on the critical path** (the next operation cannot start until the all-reduce completes, so unlike DDP's gradient all-reduce it **cannot be overlapped** with other compute). Doing dozens of such blocking all-reduces per pass over a 50 GB/s inter-node link would dwarf the compute. Therefore TP is almost always confined to the GPUs **within a single node**, wired by NVLink/NVSwitch, and the TP "width" is capped at the node's GPU count (typically **TP ≤ 8**). This single constraint — TP wants the fastest link and there are only ~8 such GPUs in a box — is why TP alone cannot scale to thousands of GPUs, and why it is combined with pipeline and data parallelism (§8.9).

### 8.7 Pipeline (inter-layer) parallelism

**Pipeline parallelism (PP)** splits the model **by depth**: the *L* layers are partitioned into *S* contiguous **stages**, and stage *s* lives on GPU *s*. A batch flows stage 0 → stage 1 → … → stage *S−1* in the forward pass and back in reverse for the backward pass. Unlike TP, the only data crossing a GPU boundary is the **activation tensor at each stage boundary** — sent point-to-point (a send/recv, not a collective) a handful of times per batch. This tiny, infrequent communication is why **PP tolerates slower inter-node links** and is the natural dimension to span nodes.

#### The pipeline bubble and micro-batching

The naïve problem: if you feed one batch straight through, stage 1 sits idle while stage 0 works, then stage 2 idles while stage 1 works, and so on. Only one of *S* GPUs is busy at any instant — utilization 1/*S*, catastrophic.

The fix is **micro-batching**: split the mini-batch into *m* smaller **micro-batches** and pump them through the pipeline back-to-back, like a factory assembly line. As soon as stage 0 finishes micro-batch 1 and passes it to stage 1, stage 0 starts micro-batch 2. Once the pipeline is "full", all *S* stages work simultaneously on different micro-batches.

But the pipeline must **fill up** at the start and **drain** at the end, and during those ramps some stages are idle. That wasted time is the **pipeline bubble**. For the simple **GPipe** schedule (all forwards, then all backwards), the bubble fraction is:

> bubble fraction = (S − 1) / (m + S − 1)

A worked example: *S* = 4 stages, *m* = 4 micro-batches → bubble = 3/7 ≈ **43% idle** — terrible. Increase to *m* = 32 micro-batches → bubble = 3/35 ≈ **8.6%**. The lesson: **make the number of micro-batches large relative to the number of stages** (rule of thumb *m* ≥ 4S) to amortize the bubble. The cost of more micro-batches is that GPipe must keep the activations of all *m* in flight, raising activation memory.

#### Better schedules: 1F1B and interleaving

- **1F1B ("one-forward-one-backward")**, used by PyTorch and Megatron: once the pipeline is primed, each stage alternates a forward micro-batch and a backward micro-batch. This keeps the **same bubble fraction** as GPipe but slashes peak **activation memory**, because a micro-batch's activations are consumed (by its backward) and freed much sooner instead of all *m* piling up. This memory win is the main reason 1F1B is the default.
- **Interleaved 1F1B (virtual pipeline stages)**: give each GPU *several non-contiguous* slices of the model (e.g. GPU 0 holds layers 0–1 *and* 16–17). With *v* interleaved chunks per device the bubble shrinks by roughly a factor of *v* — bubble ≈ (S−1)/(v·m) — at the cost of *v*× more point-to-point messages. This is what Megatron uses at very large scale.

PP's other subtleties: stages must be **load-balanced** (equal compute per stage, else the slowest stage stalls the line — the embedding and final-softmax layers are notoriously uneven), and the bubble never fully vanishes, so PP is the *least* efficient dimension per-GPU and is used only as much as memory forces you to.

### 8.8 Sequence, context, and expert parallelism

Two further axes extend the toolkit for specific bottlenecks.

#### Sequence / context parallelism

The activations and especially the attention computation scale with **sequence length *S***; for long-context training (tens or hundreds of thousands of tokens) the activation memory of even a single layer can blow up. **Sequence parallelism** partitions tensors along the **sequence dimension** across GPUs so each holds only a slice of the tokens' activations.

- In **Megatron's sequence parallelism**, the operations that TP leaves replicated (LayerNorm, dropout, the residual adds — which TP does *not* split) are themselves split along the sequence axis, removing that replicated activation memory. The TP all-reduce is correspondingly re-expressed as a reduce-scatter + all-gather pair so the two schemes mesh (note the total communication volume is unchanged, since all-reduce = reduce-scatter + all-gather).
- **Context parallelism** (e.g. **Ring Attention**) targets attention directly: because attention needs every query to see every key, the key/value blocks are passed **around a ring of GPUs** (a point-to-point ring exchange) so each GPU computes attention for its slice of queries against all keys, never materializing the full *S×S* attention matrix on one device. This is what enables million-token context windows.

The communication primitive here is typically **all-to-all** or ring exchanges, and like TP it is bandwidth-heavy, so it is kept on fast links.

#### Expert parallelism (Mixture-of-Experts)

A **Mixture-of-Experts (MoE)** layer replaces one big feed-forward network with *E* parallel "expert" FFNs plus a **router** that sends each token to only its top-*k* experts (often *k* = 1 or 2). Because each token uses only a few experts, you get a huge increase in parameters (capacity/quality) at nearly constant compute per token.

**Expert parallelism (EP)** places **different experts on different GPUs**. Now the challenge is routing: a token on GPU 0 may need an expert living on GPU 5. The solution is the **all-to-all** collective, used **twice per MoE layer**:

1. **Dispatch all-to-all**: every GPU sends each of its tokens to whichever GPU holds the token's chosen expert. After this, each GPU has gathered exactly the tokens routed to its local experts.
2. Each GPU runs its experts on the tokens it received.
3. **Combine all-to-all**: the expert outputs are shipped back to the GPUs the tokens originated from.

All-to-all is the most network-stressing collective (every rank talks to every rank), and its volume depends on the *routing*, which is data-dependent — leading to **load imbalance** (some experts get more tokens) that frameworks fight with auxiliary load-balancing losses and per-expert **capacity factors** (a cap on tokens per expert, dropping or rerouting the overflow). EP is usually combined with the other dimensions, since the experts collectively may still exceed one node's memory.

### 8.9 Composing it all: 3D / N-D parallelism

No single dimension scales to thousands of GPUs: TP is capped at a node (~8), PP's bubble grows with stages, DP's batch size cannot grow without bound (huge batches hurt convergence). Frontier training therefore **composes** them — the famous **3D parallelism** = **DP × TP × PP**, with sequence and expert parallelism as additional axes when needed. Conceptually you arrange the GPUs in a grid and assign each axis of the grid to one parallelism strategy.

#### The mapping rule, made concrete

The placement follows directly from the communication-vs-bandwidth hierarchy of §8.3:

| Dimension | Communication intensity | Overlappable? | Place on |
|---|---|---|---|
| **Tensor (TP)** | Highest — 2 all-reduces/layer, on critical path | No | **NVLink, within a node** (≤8) |
| **Sequence/Expert** | High — all-to-all per MoE/attn | Partly | Fast links, within/near node |
| **Pipeline (PP)** | Low — activations at stage boundaries | Mostly | **Across nodes** (InfiniBand) |
| **Data (DP/FSDP)** | Medium — gradients once/step, overlappable | Yes | **Outermost, across many nodes** |

**Worked layout.** Suppose you have **512 GPUs** = 64 nodes × 8 GPUs, and you want a large model that needs both splitting and throughput. A canonical choice:

- **TP = 8** → fill exactly one node with tensor parallelism (all those per-layer all-reduces ride NVLink).
- **PP = 8** → split the model's layers across 8 nodes (only activations cross the slower InfiniBand between them).
- One TP×PP "model replica" therefore occupies **8 × 8 = 64 GPUs** (8 nodes).
- **DP = 8** → 512 / 64 = 8 such replicas run data-parallel, each on its own group of 8 nodes, gradients all-reduced (and overlapped) over the network.

Total = TP(8) × PP(8) × DP(8) = 512. The **effective batch** = (per-GPU micro-batch) × (micro-batches per pipeline) × DP(8). This is precisely how systems like Megatron-DeepSpeed are configured. The ordering of the loop nest — TP innermost (fastest link), DP outermost (slowest, but overlappable) — is the heart of the design.

#### Rules of thumb

1. **Size TP to the node** (≤ NVLink domain, usually 8). Never let TP cross nodes.
2. **Use PP only as much as memory forces**, since the bubble wastes GPU; keep *m* ≥ 4×stages.
3. **Prefer FSDP/ZeRO-3 over PP** when the interconnect is fast enough, as it is simpler and avoids bubbles; reach for PP when even FSDP's sharded model won't fit or inter-node bandwidth can't sustain FSDP's extra traffic.
4. **Fill remaining GPUs with DP/FSDP** on the outside — it overlaps best and tolerates the network.
5. **Add sequence parallelism** for long context, **expert parallelism** for MoE, slotting their all-to-alls onto the fastest available links.

### 8.10 Frameworks, configs, and operational concerns

#### The framework landscape

| Framework | What it gives you | Typical use |
|---|---|---|
| **PyTorch DDP** | Plain data parallelism | Model fits on one GPU; want speed |
| **PyTorch FSDP** | ZeRO-3-style param/grad/opt sharding, native | Large model, fast interconnect, want simplicity |
| **DeepSpeed** | ZeRO-1/2/3, ZeRO-Infinity (CPU/NVMe offload), pipeline, MoE | Memory-constrained; offload; rich config via JSON |
| **Megatron-LM** | Best-in-class TP + PP (+ sequence parallel) | Frontier dense/MoE models; 3D parallelism |
| **Megatron-DeepSpeed / NeMo / Megatron-Core** | TP+PP+DP+ZeRO composed | Full 3D at scale |

For *your* near-term path: **DDP** when an R1-1.5B-class model fits on the card (it does — ~3 GB BF16 weights), and **FSDP** the moment you fine-tune something that doesn't, or want to push batch size. The DGX Spark (GB10: 128 GB unified LPDDR5X memory at ~273 GB/s — generous *capacity* but modest *bandwidth* versus an HBM datacenter card, so it is an inference/light-fine-tune box, not a pretraining one) and single-node RunPod boxes will keep you in the **intra-node** regime where DDP/FSDP shine and you can ignore PP/TP entirely until you go multi-node.

#### What a config looks like

A DeepSpeed config is a JSON blob; the essential knobs:

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "overlap_comm": true
  },
  "bf16": {"enabled": true}
}
```

This says: each GPU processes 4 samples per micro-step; accumulate gradients over 8 micro-steps before an optimizer step (so the **effective batch per GPU** = 4 × 8 = 32, and global = 32 × number of GPUs); shard everything (ZeRO-3) with optimizer state offloaded to CPU; overlap communication with compute; train in BF16. **Gradient accumulation** is the cheap way to grow the effective batch without more memory: run several micro-batches, summing gradients, and only all-reduce + step once — it also reduces communication frequency.

In PyTorch-native FSDP the equivalent is wrapping the model: `FSDP(model, sharding_strategy=FULL_SHARD, auto_wrap_policy=transformer_block_policy, mixed_precision=bf16_policy)`, launched with `torchrun --nnodes=… --nproc_per_node=8`. The `auto_wrap_policy` is what decides the **FSDP units** of §8.5 (usually "one transformer block per unit").

#### Synchronization, stragglers, and fault tolerance

Three operational realities bite at scale:

**Stragglers.** Synchronous SGD ends every step with a collective, so **the slowest rank gates the whole cluster** — one GPU throttling on heat, one node on a congested network link, or uneven pipeline stages, and all *N* GPUs wait at the barrier. Mitigations: balance pipeline stages, ensure uniform data/sequence lengths per rank (a long sequence on one rank stalls everyone — hence sorting/packing batches by length), and monitor per-rank step times. *Asynchronous* SGD avoids the barrier but perturbs the gradient semantics and is rarely used for LLM pretraining.

**Synchronization correctness.** Every rank must call every collective **in the same order**; a divergent code path (e.g. one rank taking an `if` branch that skips an all-reduce) causes a **hang/deadlock** as the others wait forever at a collective that never arrives. This is the most common multi-GPU bug. Set `NCCL_DEBUG=INFO` to see what NCCL is doing, and watch for the **NCCL watchdog timeout** that fires after a collective stalls.

**Fault tolerance.** With 512 GPUs running for weeks, hardware *will* fail mid-run. The defences: frequent **checkpointing** (save model + optimizer + RNG state; with sharded states each rank writes its own shard in parallel — "distributed checkpoint"), and **elastic / restartable** launchers (`torchrun` supports `--max-restarts` and rendezvous so a run can resume from the last checkpoint with a fresh set of nodes). Because a checkpoint of a large model is itself huge (the full 16-bytes/param optimizer state), checkpoint **frequency** is a cost trade-off: too rare and a failure wastes hours of compute; too frequent and I/O dominates. Asynchronous and sharded checkpointing libraries exist precisely to hide this cost.

For your scale today — single node, a handful of NVLink-connected GPUs (or the DGX Spark's single unified-memory GB10) — stragglers and fault tolerance are minor; the lessons here matter the moment you scale a fine-tune across multiple nodes.

---

## 9. Inference systems deep dive (the reader's primary use case)

This section is the operational heart of the primer. Everything before it — GPU architecture, memory hierarchies, FLOPs, bandwidth, precision — exists so that the behaviour described here stops feeling like a collection of magic flags and starts feeling like the inevitable consequence of how the hardware works. The central claim to internalise is this: **autoregressive LLM inference is two completely different workloads wearing one trench coat**, and almost every serving optimisation is a response to the painful second one.

### 9.1 The two phases: prefill vs decode

When you send a prompt to an LLM and ask it to generate a completion, the forward computation splits into two temporally distinct phases with opposite performance characteristics.

#### Prefill (the "prompt processing" phase)

In prefill, the model ingests the entire prompt at once. Suppose your prompt is 2,000 tokens long. The model embeds all 2,000 tokens and pushes them through every transformer layer **simultaneously**, as one big matrix. There is no sequential dependency *within* the prompt for this pass: token 1,999 does not need token 2,000's output to be computed, because in a causal transformer each position attends only to itself and earlier positions, and all of those earlier positions are already known (they are the prompt). So prefill is a dense batched matrix-multiply problem over a sequence dimension of length 2,000.

This makes prefill **compute-bound**. Recall from the FLOPs section that a transformer forward pass costs roughly `2 * N` FLOPs per token, where `N` is the parameter count (the factor of 2 is one multiply + one add per parameter; attention adds a smaller sequence-length-dependent term on top). For a prompt of `S` tokens through an `N`-parameter model:

```
Prefill FLOPs ≈ 2 * N * S
```

Worked example, our 1.5B reasoning model (R1-Distill-1.5B), prompt of 2,000 tokens, in BF16:

```
Prefill compute ≈ 2 * 1.5e9 * 2000 = 6.0e12 FLOPs = 6 TFLOP
```

An A100 (80 GB, SXM) delivers roughly 312 TFLOP/s of BF16 tensor-core throughput (vendor figure, dense; with structured sparsity the marketing number doubles to ~624, but you should plan around the dense figure). For reference, the H100 SXM is ~990 TFLOP/s dense BF16 (and ~1,979 with sparsity — vendor datasheets usually lead with the sparse number, so read the footnotes). So the *arithmetic* lower bound for prefilling that prompt on an A100 is:

```
6.0e12 / 312e12 ≈ 19 milliseconds
```

In practice you will not hit peak — call it 40–60% of peak for a small model with imperfect kernels — so maybe 35–50 ms. The point stands: prefill chews through a lot of FLOPs but does so with high efficiency because it is a fat matmul that keeps the tensor cores fed. Prefill is where the GPU looks like the dense linear-algebra monster it was built to be.

The time to finish prefill is essentially the **TTFT** (time to first token), which we define precisely in §9.8.

#### Decode (the "generation" phase)

Now the model must generate the completion, one token at a time. To produce token 2,001 it runs a forward pass; to produce token 2,002 it runs *another* forward pass that depends on token 2,001 having been sampled first; and so on. This is **inherently sequential** — you cannot compute the 50th generated token before the 49th, because the 49th is part of the input that produces the 50th. There is no way to parallelise across the time axis of generation. (Speculative decoding, §9.7, cheats this *statistically*, but does not break the dependency.)

Crucially, each decode step processes **exactly one new token** (per sequence). That one token must still be multiplied against **every weight matrix in the model**. So each decode step reads the entire parameter set out of GPU memory to do the arithmetic for a single token's worth of work.

This is the defining fact of LLM inference. Let us quantify the imbalance.

For one decode step on the 1.5B model in BF16:

- **Compute:** `2 * N * 1 ≈ 2 * 1.5e9 = 3.0e9 FLOPs` (3 GFLOP).
- **Memory traffic:** you must read all weights once: `1.5e9 params * 2 bytes = 3.0e9 bytes = 3.0 GB` (plus KV-cache reads, §9.2).

The **arithmetic intensity** — FLOPs performed per byte moved (defined in the roofline section) — is therefore approximately:

```
3.0e9 FLOPs / 3.0e9 bytes = 1 FLOP/byte
```

Compare that to the A100's "ridge point". The A100 has ~312 TFLOP/s of compute and ~2.0 TB/s of HBM bandwidth (2,039 GB/s, HBM2e), so its ridge point is `312e12 / 2.0e12 ≈ 156 FLOP/byte`. Any workload below ~156 FLOP/byte of arithmetic intensity is **memory-bandwidth-bound** on an A100. Single-stream decode sits at ~1 FLOP/byte — it is off the cliff by more than two orders of magnitude. The tensor cores are starved, sitting idle ~99% of the time, waiting for weights to stream in from HBM.

#### The bandwidth bound on decode latency

Because decode is memory-bound, you can predict its speed with a back-of-envelope that ignores compute entirely. The minimum time per decode step is roughly:

```
time_per_token ≈ (bytes that must be read per step) / (memory bandwidth)
```

For our 1.5B model in BF16 on an A100 (2.0 TB/s):

```
time_per_token ≈ 3.0e9 bytes / 2.0e12 bytes/s = 1.5 ms/token  (weights only, batch size 1)
```

So the *ceiling* is ~667 tokens/s for a single stream, and real systems land lower (KV reads, kernel launch overhead, sampling, imperfect bandwidth utilisation). On an H100 (~3.35 TB/s HBM3) the same model floors at ~0.9 ms/token. Notice what this formula does **not** contain: the A100's 312 TFLOP/s of compute is nowhere in it, because compute is not the bottleneck. **You could double the GPU's FLOPs and single-stream decode would not get one microsecond faster.**

#### Why decode dominates end-to-end latency

Put the two phases together for a realistic reasoning-model request: a 2,000-token prompt and a 4,000-token chain-of-thought completion (long CoT is exactly what R1-style models produce).

| Phase | Work | Approx. time on A100 (1.5B, BF16) |
|---|---|---|
| Prefill | 2,000 tokens, one fat matmul | ~40 ms |
| Decode | 4,000 sequential steps × ~1.5 ms | ~6,000 ms |

Decode is ~150× the wall-clock cost of prefill here. **For any generation of non-trivial length, latency is decode, and decode is bandwidth.** This single observation explains why essentially every inference optimisation below is really an attack on the memory-bandwidth wall.

### 9.2 The KV cache: the central resource in LLM serving

#### What is cached and why

Self-attention at decode step *t* needs, for the new token, to attend to **all previous positions**. Attention computes, for each position, a key vector **K** and a value vector **V**. The key vectors of all earlier tokens are needed to compute attention scores; the value vectors are needed to compute the weighted sum. Critically, **the K and V vectors of a given token never change once that token is in the sequence** — they depend only on that token and the ones before it, which are fixed.

So the naive approach — re-run attention over the whole sequence every step, recomputing every token's K and V — would redo enormous amounts of identical work (it would make each decode step cost like a prefill of the whole sequence-so-far, turning generation into an O(n²) disaster). The fix is the **KV cache**: after computing each token's K and V, you **store** them in GPU memory. At the next step you compute K and V for only the one new token, append them, and reuse all the cached ones. This is what turns the per-step attention cost from "re-process the whole sequence" into "process one token against a cached history."

The KV cache is therefore not an optional optimisation; it is the thing that makes autoregressive decoding tractable at all. But it has a cost: **memory, and memory bandwidth**.

#### The memory formula (cross-reference §6)

The size of the KV cache for a single sequence is:

```
KV bytes = 2 * L * n_kv_heads * d_head * seq_len * bytes_per_element
```

where the leading `2` is for K **and** V, `L` is the number of transformer layers, `n_kv_heads` is the number of key/value heads (with multi-head attention this equals the number of attention heads; with **grouped-query attention (GQA)** or **multi-query attention (MQA)** it is much smaller — see below), `d_head` is the dimension per head, `seq_len` is the current sequence length (prompt + generated so far), and `bytes_per_element` is 2 for FP16/BF16.

Worked example for R1-Distill-1.5B. Its config is 28 layers, hidden size 1,536, 12 attention heads, and it uses GQA with 2 KV heads; `d_head = 1536 / 12 = 128`. For a sequence of 6,000 tokens (2k prompt + 4k CoT) in BF16:

```
KV bytes = 2 * 28 * 2 * 128 * 6000 * 2
         = 172,032,000 bytes
         ≈ 172 MB ≈ 0.17 GB per sequence
```

That is small — because this model is small *and* uses GQA (only 2 KV heads instead of 12, a 6× saving). Contrast a 70B-class model with full multi-head attention (MHA), ~80 layers, 64 KV heads, `d_head=128`, at 8,000 tokens:

```
KV bytes = 2 * 80 * 64 * 128 * 8000 * 2 ≈ 21 GB per sequence
```

— more than a quarter of an 80 GB GPU **for one request's cache**. (Real 70B models such as Llama-2/3-70B actually use GQA with only 8 KV heads, which cuts this 8× to ~2.6 GB; the 64-head MHA figure here is the deliberately worst-case illustration of why GQA was adopted.) This is why GQA/MQA (which shrink `n_kv_heads`) became universal in modern models, and why KV-cache management is *the* central engineering problem in serving large models.

#### How the cache grows, and why it bounds throughput

The cache grows by **one token's worth of K and V per layer, per decode step**. Per-token growth for our 1.5B model:

```
per-token KV = 2 * 28 * 2 * 128 * 2 bytes = 28,672 bytes ≈ 28 KB/token
```

Two consequences follow, and they are the crux of serving economics:

1. **The KV cache, not the weights, is the variable that scales with your workload.** Weights are a fixed cost paid once (3 GB for our model). Every concurrent request and every generated token adds KV cache on top. The amount of free HBM after loading weights divided by the per-sequence KV size sets a hard ceiling on how many sequences you can serve concurrently — i.e. on your maximum batch size, and therefore (see §9.3) on your throughput.

2. **Decode also re-reads the KV cache every step.** The memory-traffic budget per decode step is not just "all the weights" but "all the weights **plus** the entire KV cache so far." At batch size 1 with a short sequence the weights dominate; but as sequences get long and batches get large, KV-cache reads can rival or exceed weight reads, and they become the new bandwidth bottleneck. This is why long-context decode slows down as the sequence grows, and why FlashAttention-style kernels (§9.5) matter so much.

### 9.3 Why batching is the only escape, and what it costs

We established that single-stream decode wastes >99% of the GPU's compute because arithmetic intensity is ~1 FLOP/byte. The way out follows directly from the structure of the bottleneck.

When you read a weight matrix out of HBM to multiply it by **one** token's activation vector, you have paid the full bandwidth cost of fetching that matrix. If instead you had **32 tokens from 32 different requests** waiting, you could multiply the same just-fetched weight matrix against all 32 of them before discarding it — turning a matrix-vector product (GEMV, low intensity) into a matrix-matrix product (GEMM, high intensity). The weight read is **amortised across the batch**: you moved the same bytes but did 32× the FLOPs.

This is the single most important lever in LLM serving. **Batching across concurrent requests is the only way to raise decode arithmetic intensity and pull the workload back toward the compute roofline.** Concretely, with batch size `B`, the decode-step traffic is `weights (fixed) + B * per-sequence-KV`, while the FLOPs are `B * 2N`. As long as weight reads dominate, going from `B=1` to `B=32` gives you ~32× the throughput for almost the same wall-clock per step — until one of two things happens:

- **You hit the compute roofline.** Once arithmetic intensity crosses the ridge point (~156 FLOP/byte on A100), you become compute-bound and further batching stops being free; latency per step starts climbing roughly linearly with `B`. For a 1.5B model, the crossover batch size is in the low hundreds.
- **You run out of KV-cache memory.** Each added sequence costs its full KV footprint. On a small model this is the looser constraint; on a 70B model the KV cache hits the memory wall long before the compute wall.

The practical upshot for a researcher sampling many reasoning chains: **never feed prompts one at a time.** If you want 64 chains from one prompt, submit them as a batch of 64; the GPU does roughly the same number of HBM weight-reads as for a single chain. This is discussed concretely in §9.10.

### 9.4 Continuous (in-flight) batching

Classic "static" batching has a fatal inefficiency for generation. Suppose you batch 8 requests together. Request 3 wants 50 tokens; request 7 wants 4,000 tokens (a long CoT). With static batching the whole batch runs until the *longest* member finishes, and requests that completed early leave their "slot" idle — the GPU keeps processing a near-empty batch while the stragglers grind on. Worse, new requests that arrive mid-flight must wait for the entire batch to drain before they can start.

**Continuous batching** (also called **in-flight batching**) fixes this by scheduling at the **granularity of a single decode step rather than a whole request**. After every token-generation step, the scheduler can:

- **evict** sequences that just emitted their end-of-sequence token, freeing their slots and KV cache immediately, and
- **admit** newly arrived requests (running their prefill and splicing them into the running batch).

The batch composition is fluid — sequences flow in and out continuously. This keeps the effective batch size high and the GPU saturated even under a chaotic, heterogeneous request stream with wildly varying generation lengths (exactly the reasoning-model regime). Continuous batching, introduced by the Orca system and now standard in vLLM, TGI, TensorRT-LLM, and SGLang, is typically a **2–4× throughput improvement** over static batching with no model change. It is the default you should always have on.

### 9.5 Attention and memory kernels: PagedAttention, FlashAttention, FlashDecoding

#### PagedAttention and KV-cache paging (vLLM)

Continuous batching exposes a memory-management problem. If you pre-allocate each sequence a contiguous KV-cache buffer sized for the maximum possible length (say 8,192 tokens), but most sequences only generate a few hundred tokens, you waste enormous memory on reservations never used — **internal fragmentation**. And because sequences have different lengths and come and go, the free memory becomes a patchwork of differently-sized holes — **external fragmentation** — so you can fail to fit a new sequence even when the *total* free memory would suffice. Measurements in the vLLM paper found naive allocation wasted 60–80% of KV memory.

**PagedAttention** borrows the operating-system idea of **virtual memory and paging**. The KV cache for a sequence is split into fixed-size **blocks** (pages), each holding the K/V for a fixed number of tokens (e.g. 16). A sequence's logical token positions are mapped to physical blocks through a **block table** (the analogue of an OS page table), and the physical blocks need **not be contiguous**. When a sequence needs another token's worth of cache, the allocator hands it any free block. This drives fragmentation waste down to under ~4% and lets you pack far more concurrent sequences into the same HBM — directly raising the achievable batch size and thus throughput. The attention kernel is modified to gather K/V through the block table rather than assuming a flat contiguous layout.

Paging also enables near-free **KV-cache sharing**. If many sequences share a common prefix (e.g. the same long system prompt, or many sampled chains from one prompt — *your* exact use case), the shared prefix's KV blocks can be stored **once** and pointed to by every sequence's block table, with **copy-on-write** if they later diverge. Sampling 64 chains from one 2,000-token prompt then stores that prompt's KV once, not 64 times.

#### Prefix caching

**Prefix caching** generalises that sharing across *requests over time*. The server hashes prompt prefixes and retains their computed KV blocks; a later request that begins with the same prefix **skips prefill for the shared portion entirely** and reuses the cached KV. For workflows with a fixed system prompt or few-shot preamble repeated across thousands of calls, this eliminates redundant prefill and slashes TTFT. (vLLM's "automatic prefix caching" / APC and SGLang's RadixAttention are two implementations; RadixAttention organises cached prefixes in a radix tree for efficient longest-prefix matching.)

#### Chunked prefill

A long prefill is a big compute burst. If a 4,000-token prefill for a newly admitted request runs as one monolithic kernel, it **stalls the decode steps of all currently-running sequences** for the duration — every active user sees a latency hiccup (a "decode stall"). **Chunked prefill** breaks a long prefill into smaller chunks (e.g. 512 tokens) and **interleaves** those chunks with ongoing decode steps in the same batch. This smooths tail latency and lets you co-schedule the compute-heavy prefill work with the bandwidth-heavy decode work so neither resource sits idle — prefill chunks keep the tensor cores busy while decode keeps the memory bus busy. It is a key knob for balancing TTFT against inter-token latency.

#### FlashAttention

The attention computation itself can be memory-bound in a different way. The naive implementation forms the full `S × S` attention-score matrix in HBM, writes it, reads it back to softmax, writes again, reads to multiply by V — multiple round-trips of an `O(S²)` matrix through the slow HBM. **FlashAttention** restructures attention as a **fused, tiled, online-softmax** kernel: it loads blocks of Q, K, V into fast on-chip **SRAM** (the shared memory / register space described in the architecture section), computes partial attention for each tile, and combines tiles using a running (online) softmax that never materialises the full score matrix in HBM. The result is mathematically identical attention but with `O(S)` rather than `O(S²)` HBM traffic, dramatically less memory use, and a large speedup — and it is what makes long-context prefill practical. FlashAttention-2 and -3 further improve work partitioning across warps and (on Hopper) exploit asynchronous tensor-core and TMA features.

#### FlashDecoding

FlashAttention was tuned for prefill, where the query dimension is long (many tokens at once) and there is ample parallel work. In **decode**, the query is a **single token**, so attention over a long KV cache has very little parallelism along the query axis — much of the GPU sits idle while one token attends to thousands of cached keys. **FlashDecoding** adds parallelism along the **key/value (sequence) dimension**: it splits the long KV cache into chunks processed in parallel across many GPU thread blocks, each computing a partial attention result, then combines them with a final reduction (again via online softmax). For long-context decode this can be several times faster, because it occupies the otherwise-idle streaming multiprocessors. This matters specifically for long-CoT reasoning where the KV cache is large and decode-time attention is a real cost.

### 9.6 Other decode-time optimisations

#### Tensor-parallel inference

When a model (or its KV cache) does not fit on one GPU, or when you want to cut per-token latency below what one GPU's bandwidth allows, you split each weight matrix **across** GPUs — **tensor parallelism (TP)**, described in the parallelism section. For inference specifically: TP=2 means each GPU holds half of every weight matrix and half the attention heads, each does half the matmul, and they combine via an **all-reduce** collective (a communication primitive that sums partial results across GPUs and broadcasts the total back — see the collectives section) **once per layer, per token** (in practice two all-reduces per layer — one after the attention block, one after the MLP block).

The crucial inference-specific point: because decode is bandwidth-bound, splitting the weights across 2 GPUs also splits the *weight-read traffic* across 2 memory systems, so per-token latency can nearly **halve** — *if* the per-layer all-reduce is cheap. That "if" requires a fast interconnect: **NVLink** (hundreds of GB/s between GPUs in a box) makes TP scale well; doing TP over **PCIe** or, worse, across nodes over Ethernet makes the all-reduce latency dominate and can erase the benefit. **Rule of thumb: use TP only within an NVLink-connected node, and only when the model doesn't fit on one GPU or when you genuinely need lower latency than one GPU can give.** For a 1.5B model that fits comfortably in a fraction of one A100, **tensor parallelism is pure overhead — keep it on one GPU.**

#### Quantized inference

Decode time is dominated by *reading weights*. If you store weights in fewer bytes, you read fewer bytes, and a bandwidth-bound workload speeds up almost proportionally. This is the real reason **weight quantization** helps inference latency — not (primarily) to save capacity, but to save bandwidth. Common schemes:

- **INT8 / FP8** weights: halve weight traffic vs BF16 → up to ~2× decode speedup. FP8 on Hopper (H100) has hardware tensor-core support; INT8 is widely supported.
- **4-bit weight-only** (GPTQ, AWQ, and the `Q4_K_M`-style formats used by llama.cpp): weights stored in ~4 bits, dequantized on the fly to do the matmul. ~4× less weight traffic; activations often kept in FP16. Excellent for *latency and capacity* in low-batch / single-GPU settings.

Two caveats a researcher must keep front of mind. **(1) Quantization is lossy** — it perturbs the model's outputs, and for a reasoning model whose correctness depends on long fragile chains of arithmetic, even small perturbations can change *which* chain gets sampled and degrade accuracy. If you are measuring reasoning behaviour or correctness, validate that the quantized model matches the full-precision one on your metric before trusting results. **(2) At large batch sizes** the workload becomes compute-bound, and 4-bit weight-only schemes can actually be *slower* than FP16 because of dequantization overhead. Quantization's latency win is a low-batch, bandwidth-bound phenomenon. (Precision details and numeric formats are in the precision section; here the point is purely the bandwidth lever.)

#### CUDA graphs for decode

Each decode step launches dozens to hundreds of tiny GPU kernels (one or more per layer). At ~1.5 ms/token, the **CPU-side overhead of launching all those kernels** — a few microseconds each, plus Python and framework overhead — becomes a measurable fraction of the step, and can even leave the GPU idle between kernels ("**launch-bound**"). A **CUDA graph** records the entire sequence of kernel launches for one decode step **once**, then replays the whole graph with a single launch on subsequent steps, collapsing hundreds of CPU-side launches into one. This removes launch overhead and tightens inter-token latency, especially for small models (like our 1.5B, where kernels are short and overhead is proportionally large) and small batches. vLLM, TensorRT-LLM, and others capture decode in CUDA graphs by default; it is one reason a dedicated serving engine beats a naive `model.generate()` loop.

### 9.7 Speculative decoding

Speculative decoding is the cleverest attack on the decode bandwidth wall, because it exploits the very thing that makes decode wasteful — **idle compute**.

#### The idea

Single-stream decode wastes compute: the GPU reads all the weights to produce one token. What if, for nearly the same memory traffic, you could produce *several* tokens? You can't make the big model generate multiple tokens in one pass — but you can **guess** several future tokens cheaply with a small fast **draft model**, then have the big **target model verify all the guesses in a single forward pass**.

#### The accept/verify mechanism

1. A small **draft model** (e.g. a 0.5B model, or the target model's own early layers, or extra prediction heads) autoregressively proposes `k` candidate tokens (say `k=4`) — cheap, because it is small.
2. The big **target model** runs **one** forward pass over those `k` proposed tokens *in parallel* (this is just like a length-`k` prefill — compute-bound and cheap relative to its memory cost). This yields the target model's true probability distribution at each of the `k` positions **simultaneously**.
3. A **verification** step walks left to right: each drafted token is accepted with a probability derived from the ratio of target to draft probabilities (a **modified rejection-sampling** scheme that provably preserves the target model's exact output distribution). At the first rejected token, you discard the rest of the draft and resample that one position from a corrected distribution.
4. You always make **at least one** token of progress (the corrected token), and up to `k+1` when all drafts are accepted.

The key guarantee: **the output distribution is identical to ordinary sampling from the target model.** Speculative decoding is not an approximation of the model — it produces statistically the same tokens, just faster. (This is unlike quantization, which *changes* the model.)

#### Why it helps a memory-bound regime

The target model's one verification pass over `k` tokens costs essentially the **same memory traffic** (one weight read) as a single normal decode step, but can yield up to `k+1` tokens. You are spending the *idle compute* (the verification of `k` tokens is more FLOPs, which the bandwidth-bound GPU had to spare) to buy back *memory-bound* decode steps. Expected speedup is the **mean number of tokens accepted per target pass**, which depends on how well the draft predicts the target — typically **1.5–3×** end-to-end for well-matched draft/target pairs, sometimes more on predictable text. The cost is extra implementation complexity and the draft model's memory/compute, and the acceptance rate falls if the draft is poorly aligned with the target.

#### Variants

- **Draft-model speculative decoding** (classic): a separate small model. Needs a draft that shares the target's tokenizer and predicts it well.
- **Medusa:** instead of a separate model, bolt **extra decoding heads** onto the target model that each predict a future position (token *t+1*, *t+2*, …) from the current hidden state; verify the resulting candidate tree in one pass. No separate model to host; lighter to deploy.
- **EAGLE:** does the drafting **autoregressively at the feature (hidden-state) level** rather than the token level, which predicts the target far more accurately and yields higher acceptance rates than Medusa (roughly 1.5–1.6× faster than Medusa) — currently among the strongest speculative methods. EAGLE-2/3 add dynamic draft trees and, in EAGLE-3, fuse features from all layers, pushing acceptance rates to ~0.8 on coding and instruction-following.
- **Self-speculative / Lookahead** approaches avoid any auxiliary model (e.g. by skipping layers, or n-gram-based "lookahead decoding"), trading some acceptance rate for zero extra weights.

For a researcher, speculative decoding is most worth it when you are **latency-sensitive at low batch size** (you want one chain fast). At high batch sizes you are already compute-bound and there is no idle compute to exploit, so speculative decoding's benefit shrinks or reverses — another instance of the same batch-size-dependent theme.

### 9.8 Serving metrics: latency, throughput, and goodput

You cannot optimise what you cannot name. These are the standard serving metrics; internalise them because every framework reports them and every SLO is written in them.

| Metric | Definition | Bound by |
|---|---|---|
| **TTFT** (time to first token) | Time from request arrival to the first output token. Includes queueing + prefill. | Prefill (compute), queue depth, prompt length |
| **TPOT** (time per output token) / **ITL** (inter-token latency) | Average time between successive generated tokens after the first. | Decode (bandwidth), batch size |
| **End-to-end latency** | Total request time `≈ TTFT + (num_output_tokens − 1) * TPOT`. | Both; dominated by decode for long outputs |
| **Throughput (tokens/s)** | Total output tokens per second across **all** concurrent requests. | Aggregate bandwidth + batch efficiency |
| **Throughput (requests/s)** | Completed requests per second. | As above, plus output-length distribution |
| **Goodput** | Throughput counting **only** requests that met their latency SLO (e.g. TTFT < 1 s **and** TPOT < 50 ms). | The metric that actually matters in production |

**The fundamental tension is latency vs throughput, and the knob is batch size.** Increasing batch size raises throughput (tokens/s aggregated over requests, because weight reads are amortised — §9.3) but *worsens* per-request latency once you cross into the compute-bound region (each step does more work, so TPOT rises) and lengthens queueing. Decreasing batch size does the opposite. Plotting latency against throughput as you sweep batch size traces a **Pareto front**; an **SLO** (service-level objective, e.g. "p99 TPOT < 50 ms") picks the operating point — the largest batch size whose latency still satisfies the SLO, which maximises throughput **subject to** the latency constraint. **Goodput** is the honest objective because it refuses to credit throughput won by violating latency.

For a researcher running offline sampling (no live users, no latency SLO), the calculus simplifies enormously: **you only care about total throughput (tokens/s and tokens/$).** You should push batch size as high as KV-cache memory allows, because there is no latency SLO to protect. This is the opposite of a production chat deployment.

### 9.9 Single-GPU vs multi-GPU, and disaggregated serving

#### When is multi-GPU worth it?

Three distinct reasons to use more than one GPU, with different mechanisms (detailed in the parallelism section; summarised here for inference):

1. **The weights don't fit.** A 70B model in BF16 is ~140 GB and cannot fit on one 80 GB GPU — you *must* shard it, typically with **tensor parallelism** within an NVLink node. This is a capacity necessity, not a choice.
2. **The KV cache doesn't fit / you want more concurrency.** Even if weights fit, large-batch long-context serving can exhaust HBM with KV cache; more GPUs add aggregate HBM.
3. **You need lower latency than one GPU's bandwidth allows.** TP splits weight-read traffic across GPUs' memory systems, cutting TPOT — but only if interconnect (NVLink) is fast enough that the per-layer all-reduce doesn't eat the gains.

Conversely, the costs: every layer's all-reduce is a synchronisation point; sub-NVLink interconnect can make TP a net loss; and **for any model that fits comfortably on one GPU, multi-GPU inference adds communication overhead for no benefit.** For the reader's 1.5B model, the right answer is unambiguous: **one GPU, no model parallelism.** If you have *N* GPUs and a small model, the throughput-optimal pattern is **data parallelism at the process level** — run *N* independent single-GPU server replicas and shard your prompts across them — not tensor parallelism.

#### Disaggregated prefill/decode serving

A frontier-serving idea worth knowing. Prefill is compute-bound; decode is bandwidth-bound. Running both on the same GPU means they **interfere**: a big prefill burst stalls decodes (the problem chunked prefill partially mitigates), and the two phases want different hardware characteristics. **Disaggregated serving** (e.g. DistServe, and production stacks like NVIDIA Dynamo) runs **prefill on one pool of GPUs and decode on a separate pool**, transferring the computed KV cache from prefill workers to decode workers over a fast interconnect. Each pool can then be scaled, batched, and even hardware-matched independently — you can hit tight TTFT and TPOT SLOs simultaneously instead of trading one for the other. It adds the complexity of a KV-cache transfer hop, and is a large-scale-production technique — **not something a single researcher needs**, but it cements the mental model that prefill and decode are genuinely different machines.

### 9.10 The serving-framework landscape

You will not write these kernels yourself; you will choose a framework. Each fills a niche. (Specific feature sets move fast — treat this as a conceptual map, not a spec sheet.)

| Framework | Niche / strength | When to reach for it |
|---|---|---|
| **vLLM** | The de-facto open-source throughput server. Originated PagedAttention; excellent continuous batching, prefix caching, broad model + quantization support, simple OpenAI-compatible API. | Default choice for research inference on NVIDIA GPUs — including sampling many chains from a small model. **This is almost certainly your tool.** |
| **TensorRT-LLM** | NVIDIA's maximally-optimised engine; compiles model-specific CUDA kernels for the lowest latency / highest throughput on NVIDIA hardware. Supports FP8, in-flight batching, speculative decoding. | When you need every last token/s or ms on NVIDIA in production and can pay the build/complexity cost. |
| **SGLang** | High-performance server with **RadixAttention** for aggressive prefix/KV reuse and a front-end language for structured/multi-call LLM programs. | Workloads with heavy prefix sharing, structured generation, agentic/multi-turn pipelines, constrained decoding. |
| **TGI** (Text Generation Inference) | Hugging Face's production server; solid continuous batching, tight HF ecosystem integration, easy deployment. | Production deployments already living in the HF ecosystem. |
| **llama.cpp / Ollama** | CPU-first and consumer-GPU inference with GGUF quantized weights (4-bit etc.); Ollama is a friendly wrapper. Runs on a laptop or Mac (Metal), no datacenter GPU needed. | Local experimentation, no-GPU or Apple-silicon machines, quick qualitative checks, capacity-constrained single-box use. **Not** the tool for high-throughput batched research sampling on an A100. |

A useful way to hold these: **vLLM/SGLang/TGI/TensorRT-LLM are datacenter-GPU throughput servers built around continuous batching + paged KV; llama.cpp/Ollama are local single-stream-friendly runtimes built around aggressive quantization and CPU/consumer hardware.** Your DGX Spark (GB10: 128 GB of LPDDR5x unified memory, but only ~273 GB/s of memory bandwidth — an order of magnitude below an A100's ~2 TB/s) is an interesting middle case — it can *hold* large quantized models thanks to the big unified pool, but its low bandwidth makes it a development box, not a throughput server. It is well-suited to llama.cpp/Ollama-style workflows for iteration, while RunPod A100/H100 instances are where you run vLLM for serious batched sampling.

### 9.11 Concrete guidance: running a small reasoning model for research inference

Now the payoff — turning all of the above into an operational recipe for the reader's actual job: sampling many chains-of-thought from a ~1.5B reasoning model on a rented A100/H100.

#### Use a batched server, not a Python loop

The single biggest mistake is calling `model.generate()` on one prompt at a time in a `for` loop. From §9.3 you know why: each call re-reads all weights from HBM to serve a tiny batch, wasting >99% of compute. Use **vLLM** (or any continuous-batching engine) and hand it **all your prompts/samples at once**. vLLM's `LLM.generate()` takes a *list* of prompts and internally runs continuous batching + PagedAttention; `SamplingParams(n=64)` requests 64 chains per prompt and the shared prompt prefix is cached once (§9.5). The throughput difference between a naive loop and a batched vLLM run on the same A100 is routinely **10–50×**.

#### Exploit your structure: many chains, shared prompt

Your workload is unusually friendly. Sampling *n* chains from one prompt means:

- The **prompt's KV cache is computed once** and shared across all *n* chains via paged copy-on-write — prefill is paid once, not *n* times.
- The *n* chains decode **as a batch**, so the per-step weight read is amortised across all *n* (§9.3) — exactly the high-arithmetic-intensity regime you want.

So structure your runs as *(few prompts) × (many samples each)* submitted together, and let the engine batch them. If you have many *different* prompts too, submit them all; the scheduler will fill the batch.

#### Maximising tokens/sec on one GPU — a checklist

1. **Right precision:** run in **BF16** by default on A100/H100 (native tensor-core support, numerically safe). Only reach for FP8/INT8/4-bit if memory- or bandwidth-constrained — and **validate accuracy first** (§9.6), which matters acutely for a reasoning model.
2. **Push batch size / concurrency to the KV-cache limit.** With a 1.5B model on an 80 GB A100, weights take ~3 GB, leaving ~75 GB for KV cache; at ~28 KB/token that is roughly 2.5 million cached tokens of headroom — you can run **hundreds to low-thousands of concurrent sequences**. Set vLLM's `--gpu-memory-utilization` high (e.g. 0.9) and `--max-num-seqs` large; you are throughput-bound, not latency-bound, so there is no SLO to protect.
3. **Cap `max_tokens` sanely.** Reasoning models can run on for thousands of tokens; an over-generous cap wastes KV memory (lowering achievable batch size) and compute on runaway chains. Set it to your task's real ceiling.
4. **Keep continuous batching and CUDA graphs on** (vLLM defaults). For a small model, CUDA graphs (§9.6) meaningfully cut per-token overhead because the kernels are short.
5. **Consider speculative decoding only if latency-bound.** For bulk offline sampling you are throughput-bound and already saturating compute via batching, so speculative decoding (§9.7) usually won't help and can hurt. Skip it for offline sampling; consider it only when you need a single chain back quickly.
6. **Do not use tensor parallelism** for a 1.5B model. One GPU per replica; if you have several GPUs, run several independent replicas (data parallelism) and shard prompts across them.

#### When is offload or quantization worth it — and when not?

- **CPU/disk offload** (keeping weights or KV in CPU RAM and streaming them in) trades the GPU's ~2–3 TB/s HBM for CPU RAM bandwidth (~tens of GB/s) or, worse, PCIe (~tens of GB/s). For a memory-bound decode workload that is **catastrophic** — you would bottleneck on the slow link and crater tokens/s. Offload exists to let you *run a model that otherwise wouldn't fit at all*, not to go fast. **A 1.5B model fits trivially in 80 GB, so never offload it.** Offload is a tool for cramming a 70B+ model onto too-small a GPU, accepting a large speed penalty.
- **Quantization** for a 1.5B model on an A100/H100 is usually **not worth the accuracy risk for research correctness measurements.** The model already fits with vast headroom (so you don't need it for capacity), and at the high batch sizes you'll run for throughput you're drifting toward compute-bound, where 4-bit weight-only quantization's bandwidth win shrinks (§9.6). Reserve quantization for (a) capacity-constrained boxes like a laptop or the DGX Spark for development, or (b) when you have *validated* that the quantized model reproduces your metric. For your DGX Spark, a 4-bit GGUF via llama.cpp/Ollama is a perfectly good way to iterate locally before sending the real batched run to a rented A100/H100 in BF16.

#### A concrete sizing example

Suppose you want **10,000 reasoning chains**: 100 prompts × 100 samples, average prompt 1,500 tokens, average completion 3,000 tokens, on one H100 (80 GB, ~3.35 TB/s) in BF16.

- **Decode floor per token (single stream):** `3.0e9 bytes / 3.35e12 ≈ 0.9 ms`. But you will run a large batch, so aggregate tokens/s is far higher than one stream's `1/0.9ms ≈ 1,100 tok/s`.
- **Total output tokens:** `10,000 chains × 3,000 = 3.0e7 tokens`.
- With effective batched decode throughput on the order of **several×10⁴ tokens/s** for a 1.5B model on an H100 (realistic for vLLM at high concurrency; the exact figure is workload- and version-dependent, so measure it), `3.0e7` tokens completes in **roughly 10–20 minutes** of GPU time — versus many hours with a naive one-prompt-at-a-time loop. **Measure your actual tokens/s** with a small pilot batch, then divide your total token budget by it to estimate cost and wall-clock before committing to the full run.

The throughline of this entire section: **decode is memory-bandwidth-bound, batching is how you reclaim the wasted compute, the KV cache is the resource that limits batching, and every serving feature — paging, continuous batching, prefix sharing, speculative decoding — is a different angle of attack on those same few facts.** Hold that model in your head and the flags stop being magic.

---

## 10. Practical workflow: profiling, debugging, and an optimization checklist

Everything in the preceding sections — FLOPs, bandwidth, precision, parallelism — only matters insofar as it changes a wall-clock number on *your* run. This section is the operational bridge: how to *measure* where your time and memory actually go, how to *diagnose* the bottleneck class, how to *fix* the common ones, and a literal checklist you can run down before and during a run. The single most important habit to internalize: **measure before you optimize.** Researchers routinely "optimize" the wrong thing — adding `torch.compile` to a run that is actually starved by its data loader, or buying a bigger GPU for a workload bottlenecked on a `.item()` call in the loop. The tools below exist to stop you from doing that.

### 10.1 The diagnostic toolkit

Think of diagnosis as a funnel: start with cheap, coarse, always-on tools (`nvidia-smi`) to form a hypothesis about the *bottleneck class*, then reach for progressively heavier, more precise tools (PyTorch profiler, Nsight) to localize the offending code. The four bottleneck classes you are triaging between are:

| Class | Meaning | Tell-tale sign |
|---|---|---|
| **Compute-bound** | Limited by GPU arithmetic throughput (the Tensor Cores / CUDA cores are the constraint) | High GPU-Util, high SM occupancy, achieved TFLOP/s near the card's roofline for that precision |
| **Memory-bound** | Limited by reading/writing the GPU's own HBM (high-bandwidth memory) | High GPU-Util but low achieved FLOP/s; kernels dominated by elementwise/normalization ops; achieved GB/s near the card's HBM bandwidth |
| **Comms-bound** | Limited by inter-GPU or inter-node communication (NCCL collectives) | Multi-GPU only; large gaps in the timeline labelled `ncclAllReduce`/`ncclAllGather`; scaling efficiency falls off as you add GPUs |
| **Input-bound (host-bound)** | The GPU is *idle waiting for the CPU* — data loading, preprocessing, Python overhead, or host-device copies | Low *average* GPU-Util that sawtooths between 0% and 100%; CPU cores pegged; the timeline shows GPU gaps between steps |

Keep this table in mind; every tool below is ultimately a way to figure out which row you are in.

#### nvidia-smi and nvitop — the always-on dashboard

`nvidia-smi` (NVIDIA System Management Interface) ships with the driver and is your first look at any GPU. Run it as a live monitor:

```bash
nvidia-smi -l 1          # refresh every 1 second
# or, less flicker, only the fields you care about:
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
           --format=csv -l 1
```

The header block reports, per GPU: the driver and CUDA versions, the **power draw / power cap** (e.g. `350W / 400W`), **temperature**, the **memory used / total** (e.g. `38000MiB / 81920MiB`), and **GPU-Util**. The process table at the bottom lists which PIDs hold memory — invaluable when a crashed run leaves a zombie process squatting on 40 GB of VRAM (kill it with `nvidia-smi` → note PID → `kill -9 <pid>`).

The single most important caveat in this entire section: **"GPU-Util 100%" does not mean your GPU is working efficiently.** The `utilization.gpu` field is defined as *the percentage of the last sampling interval during which at least one kernel was executing on the device.* It is a **duty-cycle**, not a throughput or efficiency metric. A kernel that uses a single CUDA core at 2% of the card's FLOP/s, but runs continuously, will show **GPU-Util 100%**. So GPU-Util tells you the GPU is *busy*, not that it is busy doing *useful work fast*. Concretely:

- **Util sawtoothing between 0% and 100%** (watch it for 20–30 seconds) → almost always **input-bound**: the GPU finishes a step, then sits idle waiting for the next batch from the CPU/data loader.
- **Util pinned at ~100% but the run is slow** → you are *compute- or memory-bound*; nvidia-smi cannot distinguish these. You need the profiler to see *which* and at what efficiency.
- **`utilization.memory`** is *not* "how full is VRAM" — it is the percentage of time the memory *interface* was being read/written. High memory-util with modest compute is a hint (not proof) of a memory-bound kernel.
- **memory.used** *is* the VRAM occupancy. Watch it climb across the first few steps; if it grows every step you have a leak (often retaining a tensor with its autograd graph — see §10.3).

`nvitop` (`pip install nvitop`) is a much nicer `htop`-style TUI over the same NVML (NVIDIA Management Library) data: colourized per-GPU bars, per-process GPU memory *and* host CPU/RAM, and a scrolling history. For interactive watching, prefer it. For logging into a file or a script, prefer `nvidia-smi --query-gpu ... --format=csv`.

What nvidia-smi/nvitop **cannot** tell you: which kernel is slow, why a kernel is slow, whether you are compute- vs memory-bound, or where the host-side gaps come from. For that, go up the funnel.

#### torch.cuda.Event — precise timing of a code region

CUDA kernel launches are **asynchronous**: when Python returns from `y = model(x)`, the kernel has merely been *enqueued* on the CUDA stream; it may not have run yet. This breaks naive timing:

```python
# WRONG — measures launch (enqueue) time, not execution time
t0 = time.time()
y = model(x)
t1 = time.time()        # returns almost instantly; kernels still running on GPU
```

To time GPU work you must either synchronize the CPU with the device, or use CUDA events that are timestamped *on the device's own timeline*:

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# warm up first (see below), then:
start.record()
y = model(x)
end.record()
torch.cuda.synchronize()          # wait until 'end' has actually occurred
elapsed_ms = start.elapsed_time(end)   # milliseconds, GPU-timeline accurate
```

`torch.cuda.synchronize()` blocks the CPU until all enqueued GPU work completes. You need it before reading the result because `elapsed_time` is only valid once `end` has been reached. The alternative — `t = time.time(); ...; torch.cuda.synchronize(); dt = time.time()-t` — also works but conflates launch overhead with execution; events are cleaner.

Two non-negotiable rules for honest GPU timing:

1. **Warm up.** The first few iterations pay one-time costs you do not want in your measurement: cuDNN/cuBLAS autotuning (`torch.backends.cudnn.benchmark` picks algorithms by trial), lazy CUDA context creation, `torch.compile` JIT compilation, and allocator warm-up. Run 5–10 untimed iterations first, then time the steady state.
2. **Synchronize at the boundaries**, and time *many* iterations and average — single-shot GPU timings are noisy.

#### The PyTorch profiler — your default workhorse

`torch.profiler` is the right tool 80% of the time: it attributes wall-clock time to PyTorch operators *and* the underlying CUDA kernels, on both CPU and GPU timelines, with negligible setup.

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),  # skip 1, warm 2, record 3
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        run_one_step(batch)
        prof.step()          # advances the schedule

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # open in chrome://tracing or perfetto.dev
```

How to read the output:

- The **table** sorted by `cuda_time_total` ranks operators by GPU time. If `aten::mm`/`aten::addmm`/`aten::bmm` (matrix multiplies) and attention/FlashAttention kernels dominate, you are likely **compute-bound** — good, that is where the FLOPs *should* go. If the top entries are elementwise ops (`aten::add`, `aten::mul`, `aten::native_layer_norm`, `aten::gelu`, copies, casts), you are likely **memory-bound** and a candidate for fusion.
- **`Self CUDA`** time excludes children; **`CUDA total`** includes them. Sort by self-time to find the true leaf-kernel hotspots.
- The **`cuda_time` vs `cpu_time` ratio** is a fast input-bound test: if total CPU time greatly exceeds total CUDA time, the GPU is starved and your bottleneck is on the host.
- The **Chrome/Perfetto trace** (`trace.json` opened at [perfetto.dev](https://perfetto.dev) or `chrome://tracing`) is the most informative artifact: two timelines (CPU thread(s) and CUDA stream(s)) with every operator and kernel as a coloured block. **Gaps on the CUDA stream = the GPU is idle.** Look at what the CPU thread is doing during those gaps (data loading? a blocking copy? Python?) — that *is* your bottleneck, visually.

`profile_memory=True` additionally tracks allocations so you can attribute peak memory to operators. The `schedule` argument matters: profiling has overhead, so you skip/warm a few steps and only *record* a handful — never profile a whole training run.

#### Nsight Systems vs Nsight Compute — when you need the real microscope

The two NVIDIA Nsight tools answer two different questions and are frequently confused:

| Tool | Granularity | Question it answers | Typical use |
|---|---|---|---|
| **Nsight Systems** (`nsys`) | **System timeline** — CPU, GPU, CUDA API, NCCL, cuDNN, memory copies, OS threads | *Where does my wall-clock time go across the whole pipeline?* | Find host-device gaps, data-loader stalls, serialization between streams, NCCL bubbles in multi-GPU |
| **Nsight Compute** (`ncu`) | **Single-kernel microarchitecture** — per-kernel occupancy, achieved vs peak FLOP/s and GB/s, memory-vs-compute roofline, warp stalls, register/shared-mem usage | *Why is this one kernel slow, and is it compute- or memory-bound?* | Optimizing a custom CUDA/Triton kernel; confirming whether a hotspot is at the memory or compute roofline |

Rule of thumb: **`nsys` first to find the slow region, `ncu` only if the slow region is a specific kernel you can change.** For most research-inference work you will rarely need `ncu` — you are running vendor kernels (cuBLAS, FlashAttention) you cannot rewrite. `nsys` is the one to learn.

```bash
# Nsight Systems: capture a timeline of a short run (let it run ~20-60s then stop)
nsys profile -o myrun --trace=cuda,nvtx,osrt,cudnn,cublas python infer.py
# open myrun.nsys-rep in the Nsight Systems GUI

# Nsight Compute: profile a few invocations of one kernel (this is SLOW — it replays kernels)
ncu --set full --launch-count 3 --kernel-name-base demangled \
    -k "regex:.*flash.*" -o kernel_report python infer.py
```

Annotate your own code with **NVTX ranges** (`torch.cuda.nvtx.range_push("dataloading")` / `range_pop()`, or the `torch.profiler` equivalents) so the `nsys` timeline labels *your* phases ("dataload", "forward", "decode-step") rather than just raw kernels. This single habit makes timelines an order of magnitude easier to read.

A caution on `ncu`: it works by **replaying each kernel many times** to collect hardware counters, so a profiled run can be 10–100× slower and needs elevated GPU performance-counter permissions (on cloud boxes you may have to enable them or run privileged). Never `ncu` a whole training loop; target one kernel with `--launch-count`.

#### DCGM — fleet-level and longitudinal monitoring

**DCGM** (NVIDIA Data Center GPU Manager) is the production/cluster counterpart to nvidia-smi. Where nvidia-smi is a snapshot, DCGM is a daemon (`nv-hostengine`) that continuously records rich telemetry — SM activity, Tensor Core activity (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`), achieved memory bandwidth, PCIe/NVLink traffic, power, thermal throttling, and ECC errors — and exposes it for export (e.g. the `dcgm-exporter` → Prometheus → Grafana stack many clusters run). For a single RunPod box you generally do not need it; on a shared cluster or DGX it is how you answer "was my GPU thermally throttling overnight?" or "what was the *average Tensor-Core utilization* across my 8-hour job?" — questions a one-second snapshot cannot. The key metric beyond nvidia-smi's duty-cycle is **`*_ACTIVE` ratios** (e.g. Tensor-Core-active fraction), which *do* approximate real efficiency, unlike GPU-Util. (Caveat: a `*_ACTIVE` ratio measures the fraction of cycles the pipe was issuing instructions, not the fraction of peak FLOP/s — it is a much better efficiency proxy than duty-cycle, but still not the same as achieved-vs-peak throughput, for which you need `ncu`'s roofline.)

#### Putting it together — a 60-second triage

1. `nvidia-smi -l 1` (or `nvitop`) for 30 s. Memory near cap? Note it (OOM risk). Util sawtoothing 0↔100% → **input-bound**, skip to §10.2 data loading. Util pinned ~100% → continue.
2. Run `torch.profiler` for ~6 steps. Sort by `cuda_time_total`. Matmuls/attention on top → **compute-bound** (you're spending FLOPs where you should; gains come from precision/batching). Elementwise/norm/copy on top → **memory-bound** (fuse, `torch.compile`).
3. CPU-time ≫ CUDA-time in the table, or visible CUDA-stream gaps in the Perfetto trace → **host/input-bound**; find the gap's cause on the CPU timeline.
4. Multi-GPU and not scaling → look for `nccl*` blocks in an `nsys` timeline → **comms-bound** (§10.2).
5. Only if a single *changeable* kernel dominates → `ncu` it to see its roofline position.

### 10.2 The classic bottlenecks and their fixes

#### Data loading — the most common cause of a "slow GPU"

The pattern: an expensive GPU sits at 30% average utilization because each step it must wait for the CPU to fetch, decode, and collate the next batch. The fix is a properly configured `DataLoader`:

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # CPU subprocesses that prepare batches in parallel
    pin_memory=True,        # stage batches in page-locked host memory -> faster H2D
    prefetch_factor=4,      # each worker pre-builds this many batches ahead
    persistent_workers=True,# don't tear down/respawn workers every epoch
    drop_last=True,
)
```

- **`num_workers`** — the number of *separate CPU processes* (not threads; this sidesteps Python's Global Interpreter Lock, the GIL, which would otherwise serialize pure-Python preprocessing) that build batches concurrently with GPU compute. Too low → the GPU starves. Too high → you oversubscribe CPU cores and thrash. Heuristic start: `num_workers ≈ number of physical CPU cores`, then tune. On a RunPod pod, check how many vCPUs you actually got (`nproc`) — a tiny instance may give you only 4–8.
- **`pin_memory=True`** — allocates the batch in **page-locked (pinned) host memory**. Normal host memory is *pageable*: the OS can move it, so a GPU DMA (Direct Memory Access) copy must first stage it through a pinned bounce buffer. Pinning skips that, enabling a faster, *asynchronous* host-to-device (H2D) copy. Pair it with `tensor.to(device, non_blocking=True)` to actually get the async overlap.
- **`prefetch_factor`** / **`persistent_workers`** — keep batches queued ahead of the GPU and avoid per-epoch worker respawn cost (which also re-reads the dataset object).

How to confirm data loading is the culprit: the Perfetto trace shows the CUDA stream idle between steps while a CPU worker thread is busy; or simply replace the real data with a single cached batch (`batch = next(iter(loader)); while True: step(batch)`) — if throughput jumps dramatically, your real bottleneck was the input pipeline. For heavy decode/augmentation (images, audio), push preprocessing onto the GPU (e.g. NVIDIA DALI) or precompute and cache to a fast format. For LLM *inference* specifically, "data loading" is usually trivial (short prompts), so this bottleneck is more a *training* concern — but tokenization in the loop can still stall you; tokenize ahead of time.

#### Small-kernel / launch-overhead bottleneck

Every CUDA kernel launch costs on the order of a few microseconds of CPU-side overhead to enqueue. A model built from thousands of tiny ops (lots of small elementwise/pointwise kernels) can become **launch-bound**: the GPU finishes each kernel faster than the CPU can dispatch the next, so the GPU idles between kernels even though "GPU-Util" may read high. Tells: in the profiler, large total *CPU* time in framework dispatch, many sub-10-µs kernels, GPU gaps between them. Fixes, in rough order:

- **Bigger batch** — amortizes per-launch overhead over more work and raises arithmetic intensity (FLOPs per byte), pushing you toward compute-bound (the good regime). Usually the first lever.
- **Operator fusion via `torch.compile`** — `model = torch.compile(model)` traces the graph and fuses chains of pointwise ops into single kernels (and can pick better kernels), directly attacking both launch overhead and memory traffic. Often a 1.2–2× speedup on modern PyTorch, though the gain is workload-dependent (memory-bound, small-kernel models benefit most; a model already dominated by big cuBLAS matmuls may see little). (Costs a one-time compile at first call — hence "warm up".)
- **CUDA graphs** — capture a fixed sequence of launches once and replay the *entire* graph with a single launch, eliminating per-kernel CPU dispatch. Ideal for the **decode loop of autoregressive LLM inference**, where you launch the same kernel sequence for every generated token with static shapes. `torch.compile(mode="reduce-overhead")` uses CUDA graphs under the hood; vLLM/TensorRT-LLM apply them aggressively. Requirement: **static shapes and static memory addresses** — graphs replay a fixed recording, so dynamic shapes or reallocations break them.
- **Use fused library kernels** — FlashAttention (fuses the whole attention computation, avoiding materializing the N×N scores matrix in HBM), fused optimizers (`torch.optim.AdamW(fused=True)`), fused LayerNorm/RMSNorm.

#### Synchronization stalls — the silent throughput killer

Because launches are async, the CPU normally *runs ahead*, queuing many kernels so the GPU never starves. Any operation that forces the CPU to **wait for the GPU to finish** ("a synchronization point") destroys this overlap and serializes your pipeline. The usual offenders, all of which trigger an *implicit* `cuda.synchronize()`:

- **`loss.item()`**, **`tensor.cpu()`**, **`tensor.numpy()`**, **`float(tensor)`**, **`print(tensor)`** — any read of a GPU value back to Python *must* block until that value is computed.
- **`if tensor > 0:`** or any control flow conditioned on a GPU tensor.
- **`tensor.to('cpu')`** without `non_blocking`, and any **`.nonzero()`/`.unique()`/boolean-mask indexing** whose output shape depends on data (the CPU must learn the size).

The classic bug: logging `loss.item()` *every* step. Each call drains the queue and stalls the GPU. Fixes: accumulate losses on-device (keep them as tensors) and only `.item()` every N steps for logging; move prints/metrics outside the hot loop; use `non_blocking=True` for D2H copies you can overlap. To *find* these, set the environment variable `CUDA_LAUNCH_BLOCKING=1` (which makes every launch synchronous — useful for debugging but do **not** benchmark with it on) or look in the Perfetto trace for `cudaStreamSynchronize`/`cudaMemcpy` blocks chopping up your stream. A single stray `.item()` in the inner loop can sharply cut throughput; this is one of the highest-ROI things to audit.

#### H2D / D2H transfer bottlenecks

Host↔device copies traverse **PCIe** (or, between GPUs, NVLink). PCIe Gen4 ×16 has a theoretical ceiling of **32 GB/s** *per direction* (PCIe Gen5 ×16 doubles this to ~64 GB/s); in practice, after protocol overhead, you typically measure **~25–28 GB/s** per direction on Gen4 — an order of magnitude below a modern datacenter GPU's HBM bandwidth (A100 80GB ≈ 2.0 TB/s, H100 SXM ≈ 3.35 TB/s, H200 ≈ 4.8 TB/s), and a chasm below its on-chip compute. (NVLink, GPU-to-GPU, is far faster than PCIe — e.g. ~900 GB/s aggregate per GPU on H100-class NVLink — which is why it matters so much for multi-GPU; see the comms section.) So *moving* data on and off the GPU is expensive relative to *computing* on it. Symptoms: prominent `cudaMemcpyAsync`/`Memcpy HtoD/DtoH` blocks in the timeline; throughput that tanks when you shuttle results back to the CPU every step. Fixes:

- **Keep data resident on the GPU.** Move the model and recurring tensors to the device *once*; don't ping-pong intermediate results to the host.
- **Pin host memory** (`pin_memory=True` / `tensor.pin_memory()`) and copy **`non_blocking=True`** so the transfer overlaps with compute on another stream.
- **Batch and reduce transfers** — one large copy beats many small ones (each copy has fixed overhead); reduce/aggregate on-device before bringing a small result back.
- **Overlap** copies with compute using separate CUDA streams (the DataLoader's pinned-memory path does a simple version of this for you).

*(Note: a unified-memory system like the DGX Spark / GB10 — where the 128 GB of LPDDR5X is physically shared between CPU and GPU at ~273 GB/s — changes this calculus: there is no discrete PCIe hop for host↔device data, but that shared bandwidth is also far below a discrete GPU's dedicated HBM, so memory-bound kernels behave differently. Profile on the actual target.)*

#### CPU preprocessing and Python overhead

Even with enough workers, if a single sample's preprocessing is genuinely CPU-heavy (tokenizing long documents, image decode+augment, audio resampling), the CPU may not keep up. Options: precompute and cache (tokenize the dataset once to disk as token-id tensors / a memory-mapped array or Arrow/`webdataset` shards), move augmentation to the GPU (DALI, Kornia), use a faster tokenizer (HuggingFace `tokenizers` is Rust-backed), or use more/cheaper CPU workers. For inference servers, do tokenization off the critical path.

#### Comms-bound (multi-GPU) — brief, since most research inference is single-GPU

When you shard across GPUs (data-parallel training, or tensor-parallel inference of a model too big for one card), GPUs must exchange data via **NCCL** (NVIDIA Collective Communications Library) **collectives** — a *collective* is a group communication primitive every rank participates in: `all-reduce` (sum gradients across all GPUs and give everyone the result), `all-gather` (concatenate shards), `reduce-scatter`, `broadcast`. If communication doesn't overlap with compute, GPUs stall waiting on each other — *comms-bound*. Diagnose with `nsys` (look for `nccl*` blocks and gaps) or by checking **scaling efficiency**: if 2 GPUs give <1.8× throughput, comms (or load imbalance) is eating the difference. Mitigations: ensure GPUs are connected by **NVLink** not just PCIe (huge difference for tensor parallelism, which is comms-heavy — check `nvidia-smi topo -m`); use **gradient accumulation** to do more compute per communication; enable gradient bucketing/overlap (PyTorch DDP does this by default); keep tensor-parallel groups *within* a node and reserve the slower inter-node network for the less chatty parallelism dimension; and tune NCCL env vars (§10.6). For your single-GPU R1-1.5B inference this section is mostly moot — flagged only so you recognize the pattern if you later scale out.

### 10.3 Debugging out-of-memory (OOM) systematically

A `CUDA out of memory` error means the allocator could not find a free block of the requested size in VRAM. First, understand *where* the memory goes (per §6's accounting): **model weights + optimizer states + gradients + activations + KV-cache (inference) + allocator reserved/fragmented overhead + CUDA context (~0.3–1 GB, version- and GPU-dependent)**. Reach for these in roughly this order.

**Step 1 — read the actual numbers.** Don't guess; print the breakdown:

```python
print(torch.cuda.memory_summary())          # detailed allocator report
print(f"alloc  {torch.cuda.memory_allocated()/1e9:.2f} GB")   # live tensors
print(f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")  # held by the caching allocator
torch.cuda.max_memory_allocated()            # peak — the number that must fit
```

The gap between **allocated** (bytes in live tensors) and **reserved** (bytes the caching allocator holds from the driver) is **fragmentation**: free space exists but is chopped into pieces too small for the next request. PyTorch caches freed blocks to avoid slow `cudaMalloc` calls, so reserved ≥ allocated always.

**Step 2 — the OOM triage ladder (cheapest/most-reversible first):**

1. **Reduce batch size.** Activation memory scales ~linearly with batch; halving the batch ~halves activations. For training, recover the effective batch with **gradient accumulation** (process micro-batches, sum gradients, step once every K micro-batches — same math, K× less activation memory at the cost of K× more forward/backward passes).
2. **Reduce sequence length / context.** Attention activation and (for inference) **KV-cache** memory scale with sequence length — the KV-cache size is `2 (K&V) × n_layers × seq_len × n_kv_heads × head_dim × dtype_bytes × batch` (the `2` is for the separate Key and Value tensors; with multi-/grouped-query attention `n_kv_heads` is smaller than the number of *query* heads, which is exactly why GQA shrinks the cache). Shorter contexts and a smaller `max_model_len` cut this directly.
3. **Lower precision.** BF16/FP16 halves weight+activation bytes vs FP32; 8-bit or 4-bit quantization (bitsandbytes, GPTQ, AWQ) cuts weights further. For *inference* of a small model this is the biggest, easiest lever (see §10.4).
4. **Gradient (activation) checkpointing** *(training only)* — don't store all activations for the backward pass; store a few checkpoints and *recompute* the rest during backward. Trades roughly one extra forward pass (commonly ~20–35% added compute, strategy-dependent) for a large activation-memory saving (often the difference between fitting and not). `torch.utils.checkpoint` / HF `gradient_checkpointing_enable()`.
5. **Shed optimizer/gradient memory** *(training)* — Adam/AdamW keeps two FP32 moment buffers per parameter (the dominant term in its ~12 bytes/param of mixed-precision optimizer state: 4 bytes each for the two moments plus a 4-byte FP32 master copy of the weights). A fused/8-bit optimizer (bitsandbytes `AdamW8bit`) shrinks those moments; ZeRO/FSDP shards optimizer states, gradients, and params across GPUs if you have more than one.
6. **Free what you're accidentally holding.** The classic leak: appending `loss` (which carries its **autograd graph** and all referenced activations) to a list — use `loss.item()` or `loss.detach()`. Wrap eval/inference in `torch.no_grad()` (or `torch.inference_mode()`) so no graph is built at all. Call `del` on big intermediates and, if needed, `torch.cuda.empty_cache()` (this only returns *reserved-but-unused* memory to the driver; it does not fix a real shortage and shouldn't be in your hot loop).

**Step 3 — fight fragmentation.** If `memory_reserved` is much larger than `max_memory_allocated` yet you still OOM, you're fragmented. Set, *before* CUDA initializes:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`expandable_segments` lets the caching allocator grow segments rather than carving fixed blocks, which markedly reduces fragmentation for workloads with **varying tensor sizes** — exactly the LLM case, where sequence lengths (and thus activation/KV shapes) differ across requests. It is one of the highest-value, lowest-effort OOM mitigations for variable-length inference. Related knobs in the same variable: `max_split_size_mb` (cap block splitting) and `garbage_collection_threshold`.

**Step 4 — the inference-specific levers.** With a serving engine (vLLM/TGI), OOM is usually KV-cache pressure: lower `--gpu-memory-utilization` (leave headroom), reduce `--max-model-len` and `--max-num-seqs` (fewer concurrent sequences = smaller KV pool), enable **paged attention** (vLLM's default — pages the KV-cache to kill fragmentation, analogous to OS virtual memory), and quantize the KV-cache (FP8) if supported. For your R1-1.5B on an 80 GB A100/H100, weights are ~3 GB in BF16 (1.5B params × 2 bytes ≈ 3.0 GB) and OOM is essentially impossible at sane batch sizes — OOM becomes a real concern only when you scale to 7B–70B or long contexts.

### 10.4 Precision choices in practice (inference-focused recap)

This was covered theoretically in §7; here is the *operational* decision for a research-inference run:

- **BF16 is the sane default** for inference on Ampere/Hopper (A100/H100/H200) and the Blackwell-generation DGX Spark. It has FP32's 8-bit exponent range (so no loss-scaling gymnastics, unlike FP16) with a 16-bit footprint, and Tensor Cores run it at full rate. Load with `torch_dtype=torch.bfloat16`.
- **FP16** is fine for pure inference too (no gradient underflow concerns without training) and has more mantissa bits than BF16 (10 vs 7), so it is marginally more precise in the mantissa, but mind activation overflow on some models given its much smaller exponent range. Prefer BF16 unless a model is known to need FP16.
- **8-bit / 4-bit quantization** (GPTQ, AWQ, bitsandbytes NF4, FP8 on Hopper/Blackwell) — use when weights don't fit or to raise throughput/lower latency on memory-bound decode. Expect small quality degradation; for a 1.5B model the memory savings are usually unnecessary, so only reach for it if you specifically want speed on the memory-bound token-by-token decode phase.
- **Verify, don't assume.** After changing precision, confirm Tensor Cores are actually engaged (profiler shows kernels with `_h884`/`_i8i32`/`tensor`/`hmma`/`hgemm` in their names; DCGM Tensor-active ratio > 0) and that outputs are numerically sane (perplexity or a few sanity generations).

### 10.5 Reproducibility and determinism on GPU

Bitwise-identical results across runs are *not* the default on GPUs, for a fundamental reason: many CUDA kernels (reductions, scatter/atomics, some convolutions and attention backward passes) **accumulate in a non-deterministic order** across thousands of parallel threads, and floating-point addition is **not associative** (`(a+b)+c ≠ a+(b+c)` in finite precision). Different thread-scheduling each run → different summation order → different low-order bits. To pin this down:

```python
import torch, numpy as np, random, os
SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)     # error out if a nondeterministic kernel is used
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False        # benchmark=True picks algos by timing -> nondeterministic choice
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for deterministic cuBLAS GEMMs
```

Caveats and costs:

- **`torch.use_deterministic_algorithms(True)`** forces deterministic kernel variants (and raises an error if none exists for an op, so you discover the nondeterminism explicitly). The deterministic variants are often **slower** — this is the speed/reproducibility trade-off; expect anywhere from negligible to ~10–30% slowdown depending on the ops. Use it for debugging and for publishable, must-replicate results; consider turning it off for exploratory speed.
- **`cudnn.benchmark = False`**: when `True`, cuDNN times several algorithms on the first call and caches the fastest — great for speed with fixed shapes, but the *choice* can vary run-to-run and across shapes, so it's both nondeterministic and counterproductive if your shapes vary every step.
- **Multi-GPU / different hardware**: determinism guarantees are *within the same GPU model, driver, and library versions*. An A100 and an H100 (or two driver versions) can produce different bits even with all flags set. Reproducibility across *machines* requires pinning the whole stack.
- **Sampling in generation**: for LLM decoding, set `do_sample=False` (greedy) or fix the sampling seed/`temperature`; otherwise generations vary by construction. Seed the framework's generation RNG explicitly.

For most research, **"seeded and statistically reproducible"** (same seed → same results on the same box) is what you need and is achievable with the block above. **"Bitwise reproducible across arbitrary hardware"** is a much stronger, costlier guarantee you rarely need.

### 10.6 Environment hygiene — defeating "it works on my machine"

A large fraction of GPU pain is version-mismatch pain. The stack has four layers that must be mutually compatible:

| Layer | What it is | How to check |
|---|---|---|
| **NVIDIA driver** | Kernel-level GPU driver; defines the *maximum* CUDA version supported | `nvidia-smi` (top-right "CUDA Version" = max supported, *not* what's installed) |
| **CUDA toolkit / runtime** | The CUDA libraries your framework was built against (cuBLAS, cuDNN, NCCL ship alongside) | `nvcc --version` (toolkit), `python -c "import torch; print(torch.version.cuda)"` (what PyTorch wants) |
| **Framework (PyTorch)** | Ships with its *own* bundled CUDA runtime when installed via the official wheels | `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| **GPU compute capability** | The architecture's SM version (Ampere `sm_80`, Hopper `sm_90`, Blackwell `sm_100`/`sm_120`); kernels must be compiled for it | `python -c "import torch; print(torch.cuda.get_device_capability())"` |

Key facts that resolve most confusion:

- The **driver is backward-compatible** with older CUDA runtimes: a newer driver runs older-CUDA-built PyTorch fine. The constraint is the *floor* — the driver must be new enough for the CUDA version your PyTorch was built with.
- **`nvidia-smi`'s "CUDA Version" is the maximum the driver supports**, *not* the toolkit you have installed and *not* what PyTorch uses. PyTorch's official wheels bundle their own CUDA runtime, so you usually **do not need a system CUDA toolkit at all** for inference — just a sufficiently new driver. (You need a matching system toolkit mainly when *compiling* custom CUDA/C++ extensions, where `nvcc`'s version must align.)
- **Install PyTorch with the CUDA build that matches your driver**, from the official index, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124` (pick the CUDA tag that fits your driver — newer toolkits like `cu126`/`cu128` exist and are needed for Blackwell-class GPUs). The single most common failure — `torch.cuda.is_available()` returns `False` — usually means a CPU-only wheel got installed or the driver is too old for the wheel's CUDA.
- **`torch.cuda.is_available() == False` checklist:** Is there a GPU (`nvidia-smi` works)? Did you install a CUDA wheel (not the default CPU wheel)? Is the driver new enough? Is `CUDA_VISIBLE_DEVICES` accidentally set to empty? Inside a container, was it started with `--gpus all` (Docker) / the NVIDIA Container Toolkit?

**NCCL environment variables** (multi-GPU only) you may need to set:

- `NCCL_DEBUG=INFO` — print NCCL's chosen transport and topology; first thing to set when distributed init hangs.
- `NCCL_P2P_DISABLE=1` — disable peer-to-peer; a diagnostic/workaround when P2P is misconfigured (slower but unblocks).
- `NCCL_IB_DISABLE=1` — disable InfiniBand (force TCP) when IB is absent/misconfigured.
- `NCCL_SOCKET_IFNAME=eth0` — pin the network interface when the host has several and NCCL picks the wrong one (a classic multi-node hang).

**Containerization** is the durable fix for "works on my machine": use NVIDIA's NGC base images (e.g. `nvcr.io/nvidia/pytorch:24.xx-py3`) which ship a known-good driver-compatible CUDA/cuDNN/NCCL/PyTorch stack, and run with the **NVIDIA Container Toolkit** (`docker run --gpus all ...`). On RunPod you typically select such an image as the pod template, which is why RunPod "just works" — the image already encodes a consistent stack. Pin exact versions (image tag, `requirements.txt` with `==`) so a pod you spin up next month is identical to today's.

### 10.7 The optimization checklists

Run these top-to-bottom; each step assumes the ones above are done. **Stop when you've met your latency/throughput/cost target** — past that, you're burning time, not GPU-hours.

#### Inference checklist (your primary use case)

1. **Fit the model.** Compute weight bytes (`params × dtype_bytes`) + KV-cache + context overhead; confirm it fits VRAM with headroom. If not: quantize (§10.3 step 3/4) or shard. *(R1-1.5B BF16 ≈ 3 GB — trivially fits an A100/H100.)*
2. **Pick the right precision.** BF16 default; quantize only if memory-constrained or you need extra decode speed (§10.4).
3. **Use an inference engine, not a raw `model.generate` loop, for any throughput-sensitive run.** vLLM / TensorRT-LLM / TGI give you **continuous batching** (a.k.a. in-flight batching — new requests join the running batch instead of waiting), **paged KV-cache**, CUDA graphs, and fused kernels essentially for free. This is usually a multiple-× win over naive HF generation and the single highest-ROI step for serving many prompts.
4. **Maximize batch / GPU utilization.** Decode is **memory-bandwidth-bound** (one token at a time → low arithmetic intensity), so throughput rises with batch size until you hit the compute roofline or VRAM. Push batch size / concurrent sequences up while watching memory. (Prefill, by contrast, is compute-bound.)
5. **Profile** (`torch.profiler` / `nsys`) to confirm where time goes and that Tensor Cores are engaged. Verify you're compute- or bandwidth-bound, not host-bound.
6. **Remove host-side stalls** (§10.2): no `.item()`/`.cpu()` per token in custom loops, tokenize off the critical path, pinned + `non_blocking` transfers, results aggregated on-device.
7. **Reduce launch overhead**: `torch.compile` (or `mode="reduce-overhead"` for CUDA graphs); ensure FlashAttention is in use. (Engines in step 3 already do most of this.)
8. **Scale out only if a single GPU can't meet the target** — tensor-parallel within a node (needs NVLink) for models too big for one card, or replicate the server across GPUs for more aggregate throughput. Adding GPUs to an already-host-bound workload buys nothing.

#### Training checklist (for when you fine-tune later)

1. **Fit one step** in memory: weights + grads + optimizer states + activations. If OOM: BF16, gradient checkpointing, smaller batch, 8-bit optimizer, then FSDP/ZeRO sharding (§10.3).
2. **Mixed precision**: `torch.autocast` + BF16 (no loss scaler needed; FP16 needs `GradScaler`). Tensor Cores do the matmuls.
3. **Saturate the input pipeline** *first* (§10.2): enough `num_workers`, `pin_memory`, `prefetch`, `persistent_workers`. A starved GPU makes every other optimization invisible — fix this before profiling kernels.
4. **Maximize batch** (raises arithmetic intensity and throughput) and recover large effective batch with **gradient accumulation** if memory-limited.
5. **`torch.compile` the model** for fusion + better kernels (one-time compile cost).
6. **Profile** to find the dominant cost; kill synchronization stalls (`.item()` every step → every N steps).
7. **Use fused kernels**: FlashAttention, fused AdamW, fused LayerNorm.
8. **Scale to multiple GPUs** only after a single GPU is efficient: DDP (model fits) or FSDP/ZeRO (it doesn't); ensure compute/comms overlap and NVLink for the chatty dimensions (§10.2).

### 10.8 Mini-runbook: a small-model research-inference run

Concretely, for an R1-1.5B-class model on a single RunPod A100/H100 or the DGX Spark — the exact regime in this project:

1. **Spin up** a pod from an NGC/RunPod PyTorch image (consistent stack, §10.6). `nvidia-smi` to confirm the GPU, driver, and free memory; `nproc` to see how many vCPUs (sets your `num_workers` ceiling).
2. **Load in BF16** (`torch_dtype=torch.bfloat16`, `device_map="cuda"`). Weights ≈ 3 GB; on an 80 GB A100/H100 you have tens of GB free (on the 128 GB unified-memory DGX Spark, plenty as well) — no quantization, no sharding needed.
3. **Wrap inference in `torch.inference_mode()`** (no autograd graph, lower memory, typically a bit faster than `no_grad` because it also skips version-counter bookkeeping). Greedy or seeded sampling for reproducibility (§10.5).
4. **For many prompts, use vLLM** (continuous batching + paged attention) rather than looping `model.generate`. Set `gpu_memory_utilization` conservatively at first (e.g. 0.85) and raise it if you need a bigger KV pool. For a handful of prompts or interactive poking, plain HF generation is fine.
5. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** if you run variable-length prompts (reduces fragmentation, §10.3).
6. **Watch `nvitop`** while it runs. Util sawtoothing → you're host-bound (tokenization/Python in the loop — batch it, move it off the critical path). Util pinned high, memory comfortable → you're in the good regime; raise batch size for more throughput until memory or the compute roofline stops you.
7. **If you need to time something**, use `torch.cuda.Event` with warm-up and `synchronize` (§10.1), and average over many iterations — never `time.time()` around an un-synchronized launch.
8. **If it's slower than expected**, run `torch.profiler` for a few steps, sort by `cuda_time_total`, and read off the bottleneck class from §10.1's triage before changing anything. **Measure, then optimize — never the reverse.**

For this project's scale (a 1.5B model, single GPU, research inference), the realistic bottlenecks you'll actually hit are, in order: **(1) host-bound stalls** from Python/tokenization or per-step `.item()`-style syncs, **(2) under-batching** leaving the GPU idle, and **(3) not using a continuous-batching engine** when generating over many prompts. Memory, multi-GPU comms, and kernel-level micro-optimization are essentially non-issues until you scale up to 7B+ models, long contexts, or multi-GPU training.

---

## Appendix A: Glossary of terms

A fast-reference list of the systems jargon used throughout. Each term is defined in depth in
the section noted.

| Term | One-line meaning | See |
|---|---|---|
| **Arithmetic intensity** | FLOPs performed per byte moved from memory; decides compute- vs memory-bound. | §4 |
| **Autocast / AMP** | Automatic mixed precision: run most ops in BF16/FP16, keep sensitive ones in FP32. | §5 |
| **Bandwidth (memory)** | Bytes/second a link (HBM, NVLink, PCIe) can move. | §1, §2 |
| **Batch (micro / global)** | Samples processed together per step; global = micro × accumulation × data-parallel size. | §7 |
| **Coalescing** | Adjacent threads reading adjacent memory so accesses merge into few transactions. | §2 |
| **Collective** | Multi-GPU communication primitive (all-reduce, all-gather, reduce-scatter…). | §8 |
| **Compute-bound** | Limited by arithmetic units, not memory traffic. | §4 |
| **Continuous batching** | Adding/removing sequences from an inference batch every step (in-flight batching). | §7, §9 |
| **CUDA core / SM** | Scalar ALU lane / Streaming Multiprocessor (a GPU's independent compute unit). | §2 |
| **Decode** | The token-by-token (sequential, memory-bound) phase of LLM generation. | §9 |
| **DDP / FSDP** | Distributed Data Parallel / Fully Sharded Data Parallel (PyTorch scaling). | §8 |
| **FLOP / FLOP/s** | One floating-point operation / rate of them; a fused multiply-add counts as 2 FLOPs. | §4 |
| **Gradient checkpointing** | Recompute activations in the backward pass to save memory (time-for-memory). | §6 |
| **HBM** | High-Bandwidth Memory — the GPU's main (global) VRAM. | §2 |
| **KV cache** | Stored keys/values for past tokens so decode needn't recompute them; dominates serving memory. | §6, §9 |
| **Kernel** | A function compiled to run on the GPU across many threads. | §3 |
| **Kernel fusion** | Merging ops into one kernel to avoid round-trips to global memory. | §3 |
| **Memory-bound** | Limited by memory bandwidth, not arithmetic; typical of LLM decode and elementwise ops. | §4, §9 |
| **MFU / HFU** | Model / Hardware FLOPs Utilization — achieved FLOP/s as a fraction of peak. | §4 |
| **Occupancy** | Fraction of a GPU's max resident warps actually active; aids latency-hiding. | §2 |
| **PagedAttention** | vLLM's paged, non-contiguous KV-cache allocation that cuts fragmentation. | §9 |
| **Pipeline / Tensor parallelism** | Split a model across GPUs by layers / within layers. | §8 |
| **Prefill** | The parallel, compute-bound phase that ingests the prompt. | §9 |
| **Quantization (PTQ/QAT)** | Representing weights/activations in fewer bits (INT8/INT4/FP8). | §5 |
| **Roofline** | Model plotting attainable FLOP/s vs arithmetic intensity to find the bottleneck. | §4 |
| **SIMT** | Single Instruction, Multiple Threads — NVIDIA's execution model (a SIMD variant). | §1, §2 |
| **Speculative decoding** | A small draft model proposes tokens a big model verifies in parallel. | §9 |
| **TTFT / TPOT / ITL** | Time-To-First-Token / Time-Per-Output-Token / Inter-Token Latency (serving metrics). | §9 |
| **Tensor Core** | Hardware unit doing matrix-multiply-accumulate at high throughput in low precision. | §2 |
| **Warp** | A group of 32 threads scheduled and executed together in lockstep. | §2 |
| **ZeRO** | DeepSpeed's sharding of optimizer states / gradients / params across data-parallel ranks. | §8 |

---

*End of primer.*
