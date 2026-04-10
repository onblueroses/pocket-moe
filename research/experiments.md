# Experiment Log

Following flash-moe's tradition of documenting what works AND what doesn't.

| # | Date | Experiment | Result | Notes |
|---|------|-----------|--------|-------|
| | | | | |

## Hypotheses to Test

1. **OS page cache on iOS** - does iOS's unified memory page cache behave as well as macOS for expert streaming? flash-moe found macOS page cache was optimal. iOS may differ due to memory pressure from other apps.

2. **UFS vs NVMe latency** - iPhone NVMe controller may have different random read characteristics than MacBook. Measure actual `pread()` latency distribution for 2MB chunks.

3. **Metal command buffer overlap** - flash-moe defers expert compute to overlap with next layer's I/O. Does this work as well on iPhone's GPU (fewer cores, shared thermal budget)?

4. **Expert locality across tokens** - if consecutive tokens activate similar experts, cache hit rate goes up. Measure expert activation overlap for typical prompts on Qwen3-30B-A3B.

5. **Thermal throttling onset** - at what point does sustained Metal + I/O cause frequency scaling? How much does it degrade tok/s?

6. **Expert prefetching** - flash-moe found prefetching didn't help on MacBook (OS cache was enough). With smaller phone cache, does prefetching the next layer's likely experts during this layer's compute help?

7. **NPU offload** - can the Neural Engine handle expert matmuls while Metal handles attention? Or is the data transfer overhead too high?
