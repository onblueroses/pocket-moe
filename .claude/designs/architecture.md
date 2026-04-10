# pocket-moe Architecture - Design Document

**Mode**: Standard
**Date**: 2026-04-10
**Status**: Final

## Problem Statement

No usable open-source tool runs 30B+ MoE models on phones at interactive speeds. Research prototypes exist (PowerInfer-2: 11.68 tok/s on Mixtral-47B) but aren't packaged for developers. flash-moe proved the technique works on MacBook/Metal. The gap: a portable, well-engineered implementation that works on Linux desktop and Android phones with Vulkan compute.

**Success criteria:**
- Run Qwen3-30B-A3B (30.5B total, 3.3B active) on an Android phone
- Achieve 2+ tok/s sustained (after thermal throttling)
- Expert weights stream from storage, not resident in RAM
- Total RAM usage under 4GB (fits 8GB phone)
- Clean, documented C codebase that others can learn from

## Understanding

### Facts
- Qwen3-30B-A3B: 128 experts/layer, 8 active, 48 layers. ~17GB at Q4.
- Per-token: 8 experts x ~2MB = ~16MB I/O per layer (cache miss case)
- flash-moe achieves 4.4 tok/s on 397B model (MacBook, Metal, NVMe)
- PowerInfer-2 achieves 11.68 tok/s on 47B MoE (phone, heterogeneous)
- UFS 4.0: 4.2 GB/s sequential, 500K+ IOPS
- Linux NVMe: 7 GB/s sequential (desktop), lower on phone
- Phone thermal: 44% perf drop within 2 inference runs (iPhone 16 Pro)
- Android page cache = Linux page cache (same LRU, same eviction)

### Context
- Dev machine: Arch Linux, Intel Arc Pro 130T/140T (Vulkan 1.3), NVMe SSD
- No macOS, no Metal. Vulkan is the GPU path.
- Has iPhone (model TBD) + can get Android device
- ggml provides mature Vulkan compute shaders for quantized matmul
- llama.cpp's Vulkan backend is production-ready
- Android NDK cross-compilation works from Linux, deploy via adb push
- io_uring available (kernel 6.19)

### Constraints
- Must work on Linux desktop (development) AND Android (target)
- No CUDA dependency (portability)
- RAM budget: 4GB max on phone (leaves 4GB for OS + cache on 8GB device)
- Must handle thermal throttling gracefully
- Custom weight format (not GGUF) optimized for expert streaming

### Unknowns (Open)
- [ ] Intel Arc Vulkan compute performance for quantized matmul
- [ ] io_uring behavior on Android (kernel version dependency)
- [ ] Actual page cache hit rate for Qwen3-30B-A3B expert patterns
- [ ] Whether ggml's Vulkan backend can be cleanly extracted from llama.cpp
- [ ] Qwen3-30B-A3B expert activation locality across typical prompts
- [ ] Android thermal headroom API accuracy vs actual throttling onset

## Research & Input

### flash-moe Key Lessons
- OS page cache is the best cache. Custom caching (Metal LRU, malloc, LZ4) was slower.
- FMA dequant kernel: reformulate dequant+matmul as fma(nibble, scale*x, bias*x)
- Deferred GPU command buffers overlap compute with I/O
- 58 experiments documented - exhaustive what-works/what-doesn't log
- Single-threaded pread was sufficient on Mac NVMe (OS handled parallelism)

### Linux-Specific Opportunities
- io_uring: async I/O with submission/completion queues. Better than pread for 8+ concurrent reads.
- mincore() / cachestat(): check page cache residency before deciding to read
- posix_fadvise(FADV_SEQUENTIAL): hint to kernel for readahead
- madvise(MADV_WILLNEED): prefetch specific page ranges

### ggml as Compute Backend
- ggml provides: tensor operations, Vulkan compute shaders, quantization (Q4_0, Q4_K_M, etc.)
- ggml does NOT provide: inference loop, model loading, expert routing, I/O scheduling
- Can link ggml as a library, use its Vulkan matmul, write everything else ourselves
- Avoids 6+ months of writing/debugging Vulkan compute shaders

## Solutions Considered

### Option A: ggml Compute + Custom Everything Else
**Approach**: Link ggml for Vulkan tensor ops. Write custom: weight format, I/O pipeline (io_uring), inference loop, expert router, thermal adaptation.

**Pros**: Best of both worlds. Novel I/O and streaming work (portfolio value). Proven GPU compute. Fastest to first token.

**Cons**: ggml dependency adds complexity. API may not expose what we need for expert-level granularity. May need to fork ggml.

**Sacrifices**: Not "from scratch" - less impressive to purists.

### Option B: Pure C + Raw Vulkan
**Approach**: Write everything from scratch. GLSL compute shaders for dequant+matmul, custom Vulkan pipeline setup, custom everything.

**Pros**: Maximum portfolio impact. Full understanding of every layer. No dependencies except Vulkan SDK.

**Cons**: 6+ months before generating a single token. Vulkan boilerplate is enormous. Debugging compute shaders is painful. Risk of never finishing.

**Sacrifices**: Time. Possibly the project itself.

### Option C: Fork llama.cpp, Add Expert Streaming
**Approach**: Fork llama.cpp, add expert offloading/streaming layer on top of existing MoE support.

**Pros**: Working inference in days. Community, tooling, model support.

**Cons**: Not novel. "I modified llama.cpp" is not a portfolio piece. Locked into GGUF format. Hard to do the I/O innovation within their architecture.

**Sacrifices**: Novelty, learning depth, custom weight format.

## Tradeoffs Matrix

| Criterion | A: ggml + custom | B: Pure from scratch | C: Fork llama.cpp |
|-----------|-------------------|----------------------|-------------------|
| Time to first token | 4-6 weeks | 6+ months | 1 week |
| Portfolio impact | High | Highest | Low |
| Novel contribution | I/O + streaming + thermal | Everything | Minimal |
| Risk of not finishing | Low | High | Very low |
| Learning depth | Deep (I/O, inference, MoE) | Deepest | Shallow |
| Maintainability | Good | Hard (Vulkan boilerplate) | Inherited complexity |
| Android portability | Good (ggml has Android) | Manual | Good |

## Recommendation: Option A

**C core + ggml for Vulkan compute. Custom weight format, I/O (io_uring), inference loop, expert streaming, and thermal adaptation.**

**Reasoning**: The novel contribution is the expert streaming technique + thermal adaptation on phones, not the matmul shader. Using ggml for compute lets us focus on what's actually new. Still deeply technical (C, io_uring, custom binary format, inference loop design). High portfolio value. Realistic timeline for a long-term research project.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  pocket-moe                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ Tokenizer│  │ Inference │  │ Expert Router │ │
│  │ (BPE, C) │  │   Loop    │  │  (top-K CPU)  │ │
│  └────┬─────┘  └────┬─────┘  └───────┬───────┘ │
│       │              │                │          │
│  ┌────▼──────────────▼────────────────▼───────┐ │
│  │              Layer Pipeline                 │ │
│  │  1. Attention (ggml Vulkan)                │ │
│  │  2. Router gate projection (ggml)          │ │
│  │  3. Top-K selection (CPU)                  │ │
│  │  4. Expert I/O (io_uring)                  │ │
│  │  5. Expert FFN (ggml Vulkan)               │ │
│  │  6. Residual + LayerNorm (ggml)            │ │
│  └────────────────────┬───────────────────────┘ │
│                       │                          │
│  ┌────────────────────▼───────────────────────┐ │
│  │            I/O Subsystem                    │ │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────┐ │ │
│  │  │io_uring │  │ mmap     │  │ Page Cache│ │ │
│  │  │ expert  │  │ resident │  │ Monitor   │ │ │
│  │  │ reads   │  │ weights  │  │ (mincore) │ │ │
│  │  └─────────┘  └──────────┘  └───────────┘ │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │            Thermal Subsystem                │ │
│  │  Monitor -> Adapt -> Log                    │ │
│  │  (sysfs on Linux, AThermal on Android)     │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │            ggml (linked library)            │ │
│  │  Vulkan compute | Quantization | Tensors   │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Weight Format: pocket-moe binary (`.pmoe`)

Two files per model:

### resident.pmoe
Contiguous, mmap-friendly. Loaded once at startup, stays in RAM.
```
[header: 64 bytes]
  magic: "PMOE" (4 bytes)
  version: uint32
  config: ModelConfig struct
  resident_size: uint64
  expert_count_per_layer: uint32
  num_layers: uint32

[embeddings]
[per-layer resident weights:]
  [attention: q_proj, k_proj, v_proj, o_proj]
  [layer_norm weights]
  [router gate projection]
  [shared experts (if any)]
[final norm + lm_head]
```

### experts.pmoe
Contiguous per (layer, expert). Experts for layer 0 first, then layer 1, etc.
```
[header: 64 bytes]
  magic: "PMEX" (4 bytes)
  version: uint32
  num_layers: uint32
  experts_per_layer: uint32
  expert_chunk_size: uint64  (fixed per expert: gate+up+down projections)

[offset table: num_layers * experts_per_layer * uint64]
  offset[layer][expert] -> byte offset in this file

[expert data:]
  [layer 0, expert 0: gate_proj | up_proj | down_proj]  (Q4 packed)
  [layer 0, expert 1: gate_proj | up_proj | down_proj]
  ...
  [layer 0, expert 127: ...]
  [layer 1, expert 0: ...]
  ...
```

**Why separate files**: resident weights are mmap'd and pinned. Expert file is mmap'd but pages are allowed to be evicted by the OS. Different madvise() strategies for each.

**Why contiguous per (layer, expert)**: each expert load is a single pread/page-fault of a contiguous chunk. No scatter-gather needed.

## I/O Pipeline: io_uring

```
Per layer:
  1. Router outputs 8 expert indices
  2. For each expert: check mincore() - is it already in page cache?
  3. For cache misses: submit io_uring SQE (pread, offset from expert table)
  4. io_uring_submit() - non-blocking
  5. Start Vulkan command buffer for attention (overlaps with I/O)
  6. io_uring_wait_cqe() - block until all expert reads complete
  7. Submit expert FFN to Vulkan
```

**Fallback**: if io_uring unavailable (old kernel, Android), fall back to pthreads + pread(). Same interface, different backend.

## Thermal Adaptation

### Linux Desktop
Read `/sys/class/thermal/thermal_zone*/temp`. Map to thermal states based on absolute temperature thresholds (configurable).

### Android
Use `AThermal_getCurrentThermalStatus()` (NDK API 30+). Maps directly to THERMAL_STATUS_NONE through THERMAL_STATUS_SHUTDOWN.

### Adaptation Strategy
```
NOMINAL:  8 active experts, no cooldown
FAIR:     6 active experts, no cooldown
SERIOUS:  4 active experts, 50ms inter-token cooldown
CRITICAL: pause, wait for NOMINAL, resume
```

Reducing active experts from 8 to 6 or 4 degrades quality but maintains interactivity. This is the project's novel contribution - no existing system does thermal-adaptive expert selection.

## Implementation Plan

### Phase 0: Foundation (weeks 1-2)
1. Set up ggml as a git submodule, build it with Vulkan on Arch
2. Install Vulkan SDK + Intel driver, verify compute shaders work
3. Write a minimal ggml Vulkan test: load Q4 tensor, run matmul
4. Design and implement the .pmoe weight format header/reader
5. Write extract_weights.py to actually split Qwen3-30B-A3B safetensors

### Phase 1: Core Inference (weeks 3-6)
1. Implement BPE tokenizer in C (port from flash-moe or ggml)
2. Implement single-layer forward pass: attention + expert FFN
3. Expert router: gate projection -> top-K selection
4. Wire up io_uring for expert loading (with pread fallback)
5. Full model forward pass: all 48 layers, generate tokens
6. Benchmark on Linux: tok/s, I/O latency, cache hit rate

### Phase 2: Optimization (weeks 7-10)
1. Page cache monitoring (mincore) and statistics
2. Expert prefetching experiments (predict next layer's experts)
3. Overlap I/O with compute (submit io_uring before GPU dispatch)
4. Profile and optimize hot paths
5. Thermal monitoring (Linux sysfs)
6. Document experiments in research/experiments.md

### Phase 3: Android Port (weeks 11-14)
1. Cross-compile with Android NDK
2. Deploy and run on phone via adb push
3. Wire Android thermal API (AThermal)
4. Benchmark: tok/s, thermal curves, cache behavior
5. Thermal-adaptive expert selection
6. Compare phone vs desktop metrics

### Phase 4: Polish + Publish (weeks 15-16)
1. Clean up codebase, document API
2. Write research/paper.md with findings
3. Benchmark suite (automated)
4. README with real performance numbers
5. Publish

## Open Questions
- Can ggml's Vulkan backend be used standalone (without llama.cpp's inference loop)?
- Does io_uring work on Android's kernel? (Pixel phones: yes, kernel 5.10+. Others: varies.)
- Is the Intel Arc Pro 130T fast enough for development, or will GPU compute be the bottleneck?
- Should we support multiple models (Mixtral, DeepSeek-V2-Lite) or focus solely on Qwen3-30B-A3B?
- What quantization format does ggml's Vulkan backend actually support? (Q4_0 guaranteed, K-quants uncertain)
