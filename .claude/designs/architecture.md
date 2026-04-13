# pocket-moe Architecture - Design Document

**Mode**: Deep
**Date**: 2026-04-13 (revised)
**Status**: Final (Rev 2)
**Previous**: Rev 1 (2026-04-10) - pre-research assumptions

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
- ggml provides mature Vulkan compute shaders for quantized matmul
- llama.cpp's Vulkan backend is production-ready
- Android NDK cross-compilation works from Linux, deploy via adb push

### Constraints
- Must work on Linux desktop (development) AND Android (target)
- No CUDA dependency (portability)
- RAM budget: 4GB max on phone (leaves 4GB for OS + cache on 8GB device)
- Must handle thermal throttling gracefully
- Custom weight format (not GGUF) optimized for expert streaming

### Research Findings (2026-04-13)

Deep research resolved several open questions and surfaced new ones.

**Resolved unknowns:**

- [x] **ggml standalone viability** - Fully confirmed. Clean API via `ggml_backend_vk_init()`, K-quants (Q4_K_M) supported on Vulkan. Precedent: whisper.cpp, stable-diffusion.cpp both use ggml standalone. No llama.cpp dependency needed.

- [x] **io_uring on Android** - **Dead end.** Google blocks io_uring via seccomp-bpf for all app processes. CVE-2022-20409, CVE-2023-21400, CVE-2024-0582 drove this. 60% of Google's kernel VRP submissions were io_uring bugs. No opt-out exists. pthreads + pread() is the only viable I/O path on Android.

- [x] **Page cache hit rate** - 71-73% natural hit rate confirmed by two independent sources (danveloper flash-moe: 71%, FlashMoE paper arXiv:2601.17063: 73% LRU baseline). Custom caching was 38% slower than OS cache. "Trust the OS" strategy validated.

- [x] **Cross-layer expert prediction** - Fate paper (arXiv:2502.12224) measured 88-93% cosine similarity between adjacent layer gate inputs. Predicting layer N+1 experts from layer N achieves 97.15% prefetch accuracy (prefill near 100%, decoding floor at 76.94%). This is a major optimization opportunity.

- [x] **Expert locality** - ExpertFlow measured 91.96% cache hit with predictive caching vs 76.61% for LRU alone. Strong consecutive token overlap. At 32 tokens only 47% of experts loaded; grows to 67% at ~256 tokens (sublinear, manageable).

- [x] **Thermal-adaptive inference novelty** - Confirmed genuinely novel. No existing engine (llama.cpp, MLC-LLM, PowerInfer-2, ExecuTorch) implements thermal-adaptive behavior. Open gap in literature.

- [x] **Intel Arc Vulkan for ML** - Problematic. ~32% efficiency on quantized ops. Q4_K_M achieves 53-64% of memory bandwidth. Driver crashes reported on Arrow Lake iGPU. No benchmarks for A130T/A140T specifically. Viable for development (I/O is the bottleneck, not compute), but expect slow iteration and possible crashes.

**New unknowns (from this review):**

- [ ] ggml graph construction overhead per layer per token (could cap throughput)
- [ ] Qwen3-30B-A3B output quality at top-6 and top-4 expert routing
- [ ] Android low-memory killer behavior with mmap'd expert files under memory pressure
- [ ] UFS controller thermal throttling interaction with GPU thermal state
- [ ] Whether ggml tensor data pointers can be swapped without rebuilding graphs
- [ ] cachestat() availability on Android kernels (mincore() confirmed available)
- [ ] AThermal API reporting lag vs actual thermal state on budget Android devices

### Key Design Changes from Rev 1

| Change | Reason |
|--------|--------|
| pthreads + pread() is primary I/O, io_uring is Linux-only optimization | io_uring blocked on Android via seccomp |
| Explicit platform backend separation | Dev and target platforms diverge on I/O, thermal, cache monitoring |
| Cross-layer expert prediction promoted to core design | 97% accuracy makes it more impactful than I/O optimization |
| Phase 0.5 added: expert reduction ablation | No published quality data for Qwen3 at top-6/top-4; gates thermal story |
| Phase order: measurement before optimization | Original plan optimized I/O before measuring baseline |
| Headers cleaned: Metal/GCD/iOS references removed | Artifacts from pre-Vulkan pivot |

## Research & Input

### flash-moe Key Lessons
- OS page cache is the best cache. Custom caching (Metal LRU, malloc, LZ4) was slower.
- FMA dequant kernel: reformulate dequant+matmul as fma(nibble, scale*x, bias*x)
- Deferred GPU command buffers overlap compute with I/O
- 58 experiments documented - exhaustive what-works/what-doesn't log
- Single-threaded pread was sufficient on Mac NVMe (OS handled parallelism)
- 71% natural page cache hit rate observed

### Cross-Layer Expert Prediction (NEW)
- Fate paper: adjacent layer gate inputs have 88-93% cosine similarity
- Predicting next-layer experts from current-layer gate logits: 97.15% accuracy
- PreScope's LLaPor predictor: >90% top-4 prediction accuracy
- Implication: during current-layer GPU compute, prefetch predicted next-layer experts
  via madvise(MADV_WILLNEED). Natural 71% cache hit rate could reach 90%+.
- This turns the pipeline from reactive (route -> read -> compute) to
  predictive (route -> compute + prefetch next layer -> read from cache)

### Linux-Specific Opportunities
- io_uring: Linux-only async I/O. Better than pread for batched concurrent reads.
  Useful for desktop development and profiling. Not available on Android.
- mincore() / cachestat(): check page cache residency before deciding to read
- posix_fadvise(FADV_SEQUENTIAL): hint to kernel for readahead
- madvise(MADV_WILLNEED): prefetch specific page ranges (key for cross-layer prediction)

### ggml as Compute Backend
- ggml provides: tensor operations, Vulkan compute shaders, quantization (Q4_0, Q4_K_M, etc.)
- ggml does NOT provide: inference loop, model loading, expert routing, I/O scheduling
- Can link ggml as a library, use its Vulkan matmul, write everything else ourselves
- Avoids 6+ months of writing/debugging Vulkan compute shaders
- K-quants fully supported on Vulkan (Q4_0 through Q6_K, plus I-quants)
- Graph execution is atomic (ggml_cgraph). MoE requires multiple graphs per layer.
- **Risk**: graph construction overhead per layer is unmeasured

## Solutions Considered

### Option A: ggml Compute + Custom Everything Else (CHOSEN)
**Approach**: Link ggml for Vulkan tensor ops. Write custom: weight format, I/O pipeline, inference loop, expert router, thermal adaptation, cross-layer prediction.

**Pros**: Best of both worlds. Novel I/O and streaming work (portfolio value). Proven GPU compute. Fastest to first token.

**Cons**: ggml dependency adds complexity. API documented as unstable (may break over 16-week timeline). Graph construction overhead is an unknown risk.

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
| Novel contribution | I/O + prediction + thermal | Everything | Minimal |
| Risk of not finishing | Low | High | Very low |
| Learning depth | Deep (I/O, inference, MoE) | Deepest | Shallow |
| Maintainability | Good | Hard (Vulkan boilerplate) | Inherited complexity |
| Android portability | Good (ggml has Android) | Manual | Good |

## Recommendation: Option A (unchanged)

**C core + ggml for Vulkan compute. Custom weight format, I/O, inference loop, expert streaming, cross-layer prediction, and thermal adaptation.**

**Reasoning**: The novel contribution is the expert streaming technique + cross-layer prediction + thermal adaptation on phones, not the matmul shader. Using ggml for compute lets us focus on what's actually new.

## Architecture

### Layered Design: Portable Core + Platform Backends

```
┌─────────────────────────────────────────────────────────────┐
│                        pocket-moe                           │
├──────────────────────────── PORTABLE CORE ──────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │Tokenizer │  │Inference │  │  Expert   │  │Cross-Lyr │  │
│  │ (BPE, C) │  │  Loop    │  │  Router   │  │Predictor │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘  │
│       │              │              │              │        │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐  │
│  │                  Layer Pipeline                       │  │
│  │  1. Attention (ggml Vulkan)                          │  │
│  │  2. Router gate projection (ggml)                    │  │
│  │  3. Top-K selection (CPU)                            │  │
│  │  4. Expert I/O (platform backend)                    │  │
│  │  5. Expert FFN (ggml Vulkan)                         │  │
│  │  6. Residual + LayerNorm (ggml)                      │  │
│  │  7. Predict next-layer experts + prefetch (async)    │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │              Weight Format (.pmoe)                    │  │
│  │  resident.pmoe (mmap, pinned) + experts.pmoe (mmap)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ggml (linked library)                    │  │
│  │  Vulkan compute | Quantization | Tensors             │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
├──────────────────────── PLATFORM BACKENDS ──────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Linux Desktop  │  │    Android      │                  │
│  │                 │  │                 │                  │
│  │ io_uring batch  │  │ pthreads pool   │                  │
│  │ + pread fallbk  │  │ + pread()       │                  │
│  │                 │  │                 │                  │
│  │ sysfs thermal   │  │ AThermal NDK    │                  │
│  │ (/sys/class/    │  │ API 30+         │                  │
│  │  thermal/)      │  │                 │                  │
│  │                 │  │                 │                  │
│  │ mincore() +     │  │ mincore()       │                  │
│  │ cachestat()     │  │                 │                  │
│  │                 │  │                 │                  │
│  │ relaxed memory  │  │ LMK-aware mmap  │                  │
│  │ (no LMK)       │  │ + mlock hints   │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Layer Expert Prediction Pipeline

```
Layer N:                                Layer N+1:
┌──────────────┐                        ┌──────────────┐
│ 1. Attention │                        │ 1. Attention │
│ 2. Gate proj │─── gate logits ──┐     │ 2. Gate proj │
│ 3. Top-K     │                  │     │ 3. Top-K     │
│ 4. Expert IO │                  │     │ 4. Expert IO │ ← cache warm
│ 5. Expert FFN│                  │     │ 5. Expert FFN│
│ 6. Residual  │                  │     │ 6. Residual  │
│ 7. Prefetch ─│──────────────────┘     │ 7. Prefetch  │
└──────────────┘  predict N+1 experts   └──────────────┘
                  madvise(WILLNEED)
                  async during step 5-6
```

Prediction method: use layer N's gate logits (before softmax) to approximate layer N+1's top-K. Adjacent layers have 88-93% cosine similarity in gate inputs (Fate paper). Even a simple "reuse current gate logits" heuristic achieves high accuracy. Refinement: learn a lightweight linear transform between adjacent gate spaces (trained offline on a few hundred prompts).

### I/O Pipeline: Platform-Abstracted

```c
// Platform-agnostic interface
typedef struct ExpertIO ExpertIO;
ExpertIO *expert_io_create(int fd, int queue_depth, PlatformConfig *platform);
int expert_io_submit(ExpertIO *io, ...);   // non-blocking
int expert_io_wait(ExpertIO *io);          // block until done
int expert_io_check_cache(ExpertIO *io, ...); // mincore probe
int expert_io_prefetch(ExpertIO *io, ...);    // madvise WILLNEED

// Linux backend: io_uring if available, pthreads+pread fallback
// Android backend: pthreads+pread always
```

Per layer:
1. Router outputs 8 expert indices
2. Check mincore() - is each expert already in page cache?
3. For cache misses: submit parallel pread() (or io_uring SQE on Linux)
4. Non-blocking return
5. Start GPU compute for attention overlap
6. Wait for all reads to complete
7. Submit expert FFN to Vulkan
8. During expert FFN: predict next-layer experts, prefetch via madvise

## Weight Format: pocket-moe binary (`.pmoe`)

*Unchanged from Rev 1.*

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

**Why separate files**: resident weights are mmap'd and pinned (mlock on Android where possible). Expert file is mmap'd but pages are evictable. Different madvise() strategies for each.

**Why contiguous per (layer, expert)**: each expert load is a single pread/page-fault of a contiguous chunk. No scatter-gather needed.

## Thermal Adaptation

### Signals (platform-abstracted)
```c
typedef struct ThermalBackend ThermalBackend;
ThermalBackend *thermal_create(PlatformConfig *platform);
ThermalState thermal_poll(ThermalBackend *backend);
```

| Platform | Signal Source |
|----------|---------------|
| Linux | `/sys/class/thermal/thermal_zone*/temp` with configurable thresholds |
| Android | `AThermal_getCurrentThermalStatus()` (NDK API 30+) |
| Fallback | tok/s decline detection (no OS API needed) |

### Adaptation Strategy
```
NOMINAL:  8 active experts, no cooldown
FAIR:     6 active experts, no cooldown
SERIOUS:  4 active experts, 50ms inter-token cooldown
CRITICAL: pause, wait for NOMINAL, resume
```

### Important Caveats (from research)
- AThermal API reports thresholds, not temperatures. Some devices return THERMAL_STATUS_NONE regardless of actual state.
- **tok/s decline is the most reliable thermal proxy.** If tok/s drops >20% from baseline, assume thermal throttling regardless of API reporting.
- UFS storage also throttles under heat. Reducing GPU expert count doesn't reduce storage I/O thermal load. At SERIOUS level, the I/O cooldown addresses both GPU and storage thermal budgets.
- Expert reduction quality impact on Qwen3-30B-A3B is **unknown** and must be validated before building this system (Phase 0.5 ablation).

## ggml Integration: Risk Mitigation

### Graph Construction Overhead (CRITICAL UNKNOWN)

MoE requires at least 2 ggml graph builds per layer (attention + expert FFN), totaling 96+ per token across 48 layers. Graph construction involves ggml_init, tensor allocation, graph building, backend scheduling.

**Must measure in Phase 0:**
```c
// Benchmark: build and execute 100 minimal ggml Vulkan graphs
// Target: < 0.5ms per graph (< 48ms overhead per token)
// Danger zone: > 2ms per graph (> 192ms overhead, caps at ~5 tok/s)
```

**Mitigation strategies if overhead is too high:**
1. **Template graphs**: Pre-build graph structures, swap tensor data pointers between invocations. Requires testing whether ggml supports pointer swaps without full rebuild.
2. **Batched graph**: Combine attention + expert FFN into single graph where possible (sacrifice I/O overlap for reduced graph count).
3. **Direct Vulkan dispatch**: For expert FFN only, bypass ggml graph and call Vulkan compute pipeline directly. Last resort - defeats the purpose of using ggml.

### API Stability
ggml's internal API is documented as unstable. Over a 16-week project, breaking changes are likely. Mitigations:
- Pin ggml submodule to a known-working commit
- Wrap ggml calls behind a thin abstraction layer (pocket_compute_*)
- Monitor ggml-org/ggml releases for breaking changes

## Implementation Plan (Revised)

### Phase 0: Validation (weeks 1-2)
1. Set up ggml as git submodule, build with Vulkan on Arch Linux
2. Install Vulkan SDK + Intel driver, verify compute shaders work
3. **Measure ggml graph construction overhead** (critical: < 0.5ms per graph target)
4. Write minimal ggml Vulkan test: load Q4_K_M tensor, run matmul
5. Design and implement .pmoe weight format header/reader
6. Write extract_weights.py to split Qwen3-30B-A3B safetensors

**Gate**: If graph overhead > 2ms, investigate template graphs before proceeding.

### Phase 0.5: Expert Reduction Ablation (week 2-3)
1. Run Qwen3-30B-A3B in llama.cpp with standard top-8 routing (baseline quality)
2. Modify llama.cpp to force top-6 routing, measure perplexity + qualitative output
3. Modify llama.cpp to force top-4 routing, measure perplexity + qualitative output
4. Document results in research/experiments.md

**Gate**: If top-4 produces unusable output, thermal adaptation must use cooldown pauses instead of expert reduction below top-6. Adjust thermal strategy accordingly.

### Phase 1: First Token (weeks 3-6)
1. Implement BPE tokenizer in C
2. Implement single-layer forward pass: attention + expert FFN
3. Expert router: gate projection -> top-K selection
4. Wire up pthreads + pread() for expert loading (primary path)
5. Full model forward pass: all 48 layers, generate first token
6. Basic timing instrumentation (per-layer, per-subsystem)

### Phase 1.5: Measurement (week 6-7)
1. Page cache monitoring (mincore) and statistics per layer
2. I/O latency distribution (histogram of expert load times)
3. Expert activation pattern logging (which experts, how often, overlap)
4. Cache hit rate over time (does it stabilize? how fast?)
5. Per-layer timing breakdown (attention vs routing vs I/O vs expert FFN)
6. Document in research/experiments.md

### Phase 2: Prediction + Optimization (weeks 7-10)
1. Implement cross-layer expert prediction (gate logit reuse heuristic)
2. Wire madvise(MADV_WILLNEED) prefetching for predicted experts
3. Measure cache hit rate improvement (target: 71% -> 85%+)
4. io_uring backend for Linux desktop (optimization, not required)
5. Overlap I/O with compute (submit reads before GPU dispatch)
6. Profile and optimize hot paths

### Phase 3: Android Port (weeks 11-14)
1. Cross-compile with Android NDK
2. Deploy and run on phone via adb push
3. Wire Android thermal API (AThermal)
4. Implement tok/s-based thermal detection (backup to AThermal)
5. Thermal-adaptive expert selection (using Phase 0.5 ablation results)
6. Benchmark: tok/s, thermal curves, cache behavior on phone
7. Test LMK interaction: does Android reclaim expert pages under pressure?
8. Compare phone vs desktop metrics

### Phase 4: Polish + Publish (weeks 15-16)
1. Clean up codebase, document API
2. Write research/paper.md with findings
3. Benchmark suite (automated)
4. README with real performance numbers
5. Publish

## Contemplation Summary

Domains visited: process_flow > meta_cognitive > skillful_means > non_dual > meditation > non_dual

Key turns: (1) The io_uring finding, rather than being a setback, clarified that the real I/O story is simpler - pthreads + pread is sufficient and the optimization energy should go to cross-layer prefetching instead. (2) The ggml graph construction overhead emerged as the most dangerous unmeasured variable - it sits in the critical path of every token and could silently cap throughput. (3) The deepest shift was recognizing the project optimizes for the wrong things in its original phase order - the research contribution lives in measurement and adaptation, not in I/O plumbing. Measurement must precede optimization.

## Open Questions
- Can ggml's Vulkan backend be used standalone? **Resolved: Yes.** (Rev 2)
- Does io_uring work on Android? **Resolved: No.** Google seccomp-blocks it. (Rev 2)
- Is the Intel Arc Pro 130T fast enough for development? Partially resolved: ~32% efficiency, driver issues possible. Workable since I/O is the bottleneck, not compute.
- Should we support multiple models or focus solely on Qwen3-30B-A3B? (Open)
- What quantization format does ggml's Vulkan backend support? **Resolved: Q4_0 through Q6_K, I-quants.** (Rev 2)
- What is the ggml graph construction overhead per layer? (NEW - Critical)
- Does Qwen3-30B-A3B quality degrade gracefully at top-6 and top-4? (NEW - Gates thermal story)
- How does Android LMK interact with mmap'd expert weights under memory pressure? (NEW)
- Can ggml tensor data pointers be swapped without rebuilding graphs? (NEW)
