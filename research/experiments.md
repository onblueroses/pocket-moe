# Experiment Log

Following flash-moe's tradition of documenting what works AND what doesn't.

| # | Date | Experiment | Result | Notes |
|---|------|-----------|--------|-------|
| | | | | |

## Hypotheses to Test

### Phase 0: Validation

1. **ggml graph construction overhead** - How long does it take to build and execute a minimal ggml Vulkan graph? Target: < 0.5ms. If > 2ms, expert FFN dispatch alone adds 192ms per token across 48 layers. This is the critical unknown.

2. **Intel Arc Vulkan compute stability** - Can the Intel Arc Pro 130T/140T run ggml Vulkan matmul reliably? Reports of driver crashes on Arrow Lake iGPU exist. Need to verify before investing weeks of development.

3. **ggml tensor pointer swap** - Can ggml tensor data pointers be reassigned between graph executions without rebuilding the graph? If yes, template graphs can amortize construction overhead.

### Phase 0.5: Expert Reduction Ablation

4. **Qwen3-30B-A3B at top-6** - Does reducing active experts from 8 to 6 produce acceptable output quality? Measure perplexity on a standard benchmark (WikiText-2 or similar) and qualitative output on diverse prompts.

5. **Qwen3-30B-A3B at top-4** - Same as above but at top-4. If quality is unusable, thermal adaptation must rely on cooldown pauses rather than expert reduction below 6.

### Phase 1.5: Measurement

6. **OS page cache hit rate over time** - flash-moe measured 71% on Mac NVMe. FlashMoE paper measured 73% LRU baseline. What does pocket-moe see on Linux NVMe for Qwen3-30B-A3B? Does it stabilize? How quickly?

7. **Expert activation locality** - Do consecutive tokens activate the same experts? Measure per-layer expert overlap between consecutive tokens. ExpertFlow saw 91.96% cache hit with predictive caching.

8. **I/O latency distribution** - Histogram of pread() latency for 2MB expert chunks. Cache hit vs cache miss. Sequential vs random access patterns.

### Phase 2: Prediction + Optimization

9. **Cross-layer expert prediction accuracy** - Fate paper measured 97.15% with learned predictor. What does the simple heuristic (reuse current gate logits for next layer) achieve? Is it enough, or do we need the learned transform?

10. **madvise prefetch effectiveness** - Does madvise(MADV_WILLNEED) actually improve cache hit rates for predicted experts? Or is the kernel readahead already sufficient?

11. **io_uring vs pthreads+pread on Linux** - For 8 concurrent 2MB reads, does io_uring provide meaningful speedup over a thread pool? flash-moe found single-threaded pread sufficient on Mac.

### Phase 3: Android

12. **UFS vs NVMe latency** - Android UFS storage may have different random read characteristics than desktop NVMe. Measure actual pread() latency distribution for 2MB chunks on target phone.

13. **Android page cache under memory pressure** - How aggressively does the low-memory killer reclaim mmap'd expert pages? Does mlock() on resident weights survive LMK pressure?

14. **AThermal API accuracy** - Compare AThermal_getCurrentThermalStatus() reporting against actual tok/s decline. If API lags or is unreliable, tok/s-based detection is the primary signal.

15. **UFS thermal interaction** - Does sustained expert streaming throttle UFS storage independently of GPU thermal state? Measure I/O latency over 5-minute inference sessions.

16. **Expert prefetching on phone** - flash-moe found prefetching didn't help on MacBook (OS cache was enough). With smaller phone cache and LMK pressure, does cross-layer prefetching help more on Android than desktop?
