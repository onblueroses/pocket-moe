# pocket-moe

Run large Mixture-of-Experts language models on phones by streaming expert weights from flash storage on demand.

## Concept

Large MoE models (15-30GB at 4-bit) don't fit in phone RAM. But phones only activate a small fraction of experts per token - the rest sit idle. pocket-moe keeps non-expert weights (embeddings, attention, norms) resident in memory and streams activated expert weights from UFS/NVMe storage through the OS page cache.

**Target model:** [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) - 30.5B total parameters, 3.3B active per token, 128 experts per layer (8 active). At Q4: ~17GB on disk, ~1.5-2GB resident RAM.

**Target hardware:** iPhone with Apple Silicon (A17 Pro+), 8GB+ RAM, iOS 17+.

**Estimated performance:** 2.5-5 tok/s (before thermal throttling).

## Architecture

```
┌─────────────────────────────┐
│  Resident RAM (~2GB)        │
│  Embeddings, attention,     │
│  layer norms, shared experts│
└──────────────┬──────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼────┐
│ Metal │ │ CPU   │ │ NVMe   │
│ GPU   │ │ Router│ │ Storage│
│ Attn  │ │ Top-K │ │ Expert │
│ FFN   │ │ Sched │ │ pool   │
└───┬───┘ └───┬───┘ └───┬────┘
    │         │         │
    │    ┌────▼────┐    │
    │    │ Page    │◄───┘
    │    │ cache   │ pread()
    │    │ (4-6GB) │
    │    └────┬────┘
    │         │
    └─────────┘
      Expert FMA
      dequant+matmul
```

**Per-layer pipeline:**
1. GPU runs previous layer's expert pass (deferred command buffer)
2. GPU computes attention projections (Metal + Accelerate BLAS)
3. CPU runs softmax + top-K routing (picks 8 of 128 experts)
4. Storage streams 8 expert chunks (~2MB each) via parallel `pread()` + GCD
5. GPU runs expert forward pass with FMA dequant kernel

**Key principle:** Trust the OS page cache. Flash-moe proved that custom caching (Metal LRU, malloc cache, LZ4 compression) was slower than letting the kernel manage it. We start with the same assumption and measure.

## Prior Art

- [flash-moe](https://github.com/danveloper/flash-moe) - Qwen3.5-397B on MacBook via NVMe expert streaming. The direct inspiration. Pure C/Metal.
- [PowerInfer-2](https://arxiv.org/abs/2406.06282) - 11.68 tok/s on TurboSparse-Mixtral-47B on smartphones via CPU/GPU/NPU heterogeneous scheduling.
- [Mixture of Cache-Conditional Experts](https://arxiv.org/abs/2412.00099) - 2x MoE speedup on mobile via cache-aware routing.

## What's Novel

1. **Thermal-adaptive scheduling** - monitor thermal state, dynamically reduce expert activation or throttle inference rate to sustain performance over minutes, not seconds.
2. **Fine-grained expert streaming on iOS** - flash-moe targets MacBook NVMe. Phone storage (UFS/NVMe) has different latency and bandwidth characteristics.
3. **Practical mobile package** - PowerInfer-2 is a research prototype. This ships as something you can build and run.

## Project Structure

```
src/
  main.m              App entry point
  engine/
    inference.c        Token-by-token inference loop
    attention.c        Multi-head / GQA / MLA attention
    expert.c           Expert streaming + FMA dequant
    router.c           Top-K expert routing
    thermal.c          Thermal monitoring + adaptive scheduling
    tokenizer.c        BPE tokenizer (C, no Python)
  metal/
    expert.metal       Expert matmul + dequant shaders
    attention.metal    Attention compute shaders
  io/
    storage.c          pread() expert loading + GCD dispatch
    cache.c            Page cache statistics + monitoring
  model/
    config.c           Model config parsing
    weights.c          Weight layout + mmap
scripts/
  extract_weights.py   Convert safetensors to pocket-moe format
  benchmark.py         Automated benchmark runner
research/
  experiments.md       Experiment log (flash-moe style)
  thermal-profile.md   Thermal measurement data
```

## Requirements

- Xcode 15+ (Metal, Accelerate framework)
- iPhone with A17 Pro or newer (8GB+ RAM)
- ~17GB free storage for Qwen3-30B-A3B Q4 weights
- Python 3.10+ (weight extraction only)

## Status

**Phase: Research + Scaffold.** No inference yet. Currently:
- [ ] Weight format design + extraction script
- [ ] BPE tokenizer (port from flash-moe)
- [ ] Metal expert dequant kernel
- [ ] Basic inference loop (single layer)
- [ ] Full model forward pass
- [ ] Thermal monitoring
- [ ] Benchmarks on real hardware

## License

MIT
