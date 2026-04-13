# pocket-moe

Run large Mixture-of-Experts language models on phones by streaming expert weights from flash storage on demand.

## Concept

Large MoE models (15-30GB at 4-bit) don't fit in phone RAM. But phones only activate a small fraction of experts per token - the rest sit idle. pocket-moe keeps non-expert weights (embeddings, attention, norms) resident in memory and streams activated expert weights from NVMe/UFS storage through the OS page cache.

**Target model:** [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) - 30.5B total parameters, 3.3B active per token, 128 experts per layer (8 active). At Q4: ~17GB on disk, ~1.5-2GB resident RAM.

**Target hardware:** Linux desktop (development), Android phones (deployment). Vulkan compute.

**Estimated performance:** 2-5 tok/s on phone (before thermal throttling).

## Architecture

```
┌──────────────────────────────────────────────────┐
│              pocket-moe                           │
│                                                   │
│  PORTABLE CORE                                    │
│  ┌──────────┐ ┌───────────┐ ┌──────────────────┐ │
│  │Tokenizer │ │ Inference │ │ Expert Router +  │ │
│  │ (BPE, C) │ │   Loop    │ │ Cross-Layer Pred │ │
│  └────┬─────┘ └─────┬─────┘ └────────┬─────────┘ │
│       │             │                │            │
│  ┌────▼─────────────▼────────────────▼─────────┐  │
│  │            Layer Pipeline                   │  │
│  │  1. Attention       (ggml/Vulkan)          │  │
│  │  2. Gate projection (ggml/Vulkan)          │  │
│  │  3. Top-K routing   (CPU)                  │  │
│  │  4. Expert I/O      (platform backend)     │  │
│  │  5. Expert FFN      (ggml/Vulkan)          │  │
│  │  6. Residual + Norm (ggml)                 │  │
│  │  7. Predict + prefetch next-layer experts  │  │
│  └─────────────────────┬───────────────────────┘  │
│                        │                          │
│  PLATFORM BACKENDS                                │
│  ┌──────────────────┐ ┌────────────────────────┐  │
│  │ Linux: io_uring  │ │ Android: pthreads +   │  │
│  │  + sysfs thermal │ │  pread + AThermal API │  │
│  └──────────────────┘ └────────────────────────┘  │
│                                                   │
│  ┌──────────────────────────────────────────────┐ │
│  │  ggml (Vulkan compute, quantization)         │ │
│  └──────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

**Per-layer pipeline:**
1. GPU computes attention (ggml Vulkan backend)
2. GPU computes gate projection, CPU runs top-K routing
3. Platform I/O backend loads cache-miss experts (~2MB each)
4. GPU runs expert FFN with quantized matmul (overlapping I/O)
5. CPU accumulates weighted expert outputs
6. Predict next-layer experts from current gate logits, prefetch via madvise

**Key principles:**
- Trust the OS page cache (71-73% hit rate confirmed, custom caching is slower)
- Cross-layer expert prediction for prefetching (97% accuracy, Fate paper)
- Thermal-adaptive expert count for sustained inference on phones

## Prior Art

- [flash-moe](https://github.com/danveloper/flash-moe) - Qwen3.5-397B on MacBook via NVMe expert streaming. Direct inspiration.
- [PowerInfer-2](https://arxiv.org/abs/2406.06282) - 11.68 tok/s on Mixtral-47B on smartphones via heterogeneous scheduling.
- [Mixture of Cache-Conditional Experts](https://arxiv.org/abs/2412.00099) - 2x MoE speedup on mobile via cache-aware routing.
- [Fate](https://arxiv.org/abs/2502.12224) - Cross-layer expert prediction with 97% prefetch accuracy.
- [FlashMoE](https://arxiv.org/abs/2601.17063) - ML-based cache replacement for MoE expert streaming.

## What's Novel

1. **Thermal-adaptive expert selection** - dynamically reduce active experts under thermal pressure to sustain inference over minutes, not seconds. No existing engine does this.
2. **Cross-layer expert prediction** - prefetch next-layer experts during current-layer compute using gate logit similarity (88-93% cosine between adjacent layers).
3. **Cross-platform C core** - portable inference loop + platform-specific backends for I/O and thermal monitoring. Linux desktop + Android from the same codebase.
4. **Custom weight format (.pmoe)** - separate resident/expert files optimized for streaming access patterns.

## Weight Format

Two files per model:
- `resident.pmoe` - non-expert weights, mmap'd and pinned (~2GB for Qwen3-30B-A3B Q4)
- `experts.pmoe` - expert weights, contiguous per (layer, expert), mmap'd with OS-managed paging (~15GB)

## Project Structure

```
src/
  main.c               Entry point + CLI
  engine/
    inference.c         Token-by-token inference loop
    attention.c         Attention (delegates to ggml)
    expert.c            Expert loading + FFN dispatch
    router.c            Top-K expert routing
    predictor.c         Cross-layer expert prediction + prefetching
    thermal.c           Thermal monitoring + adaptive scheduling
    tokenizer.c         BPE tokenizer (C)
  io/
    uring.c             io_uring expert reads (Linux only)
    storage.c           pthreads + pread() (primary) + mmap
    cache.c             Page cache monitoring (mincore/cachestat)
  model/
    config.c            Model config parsing
    weights.c           .pmoe format reader + mmap
scripts/
  extract_weights.py    Convert safetensors to .pmoe format
  benchmark.py          Automated benchmark runner
research/
  experiments.md        Experiment log
  thermal-profile.md    Thermal measurement data
```

## Requirements

**Development (Linux):**
- GCC/Clang with C17 support
- Vulkan SDK + drivers (Intel, AMD, or NVIDIA)
- Python 3.10+ (weight extraction only)
- ~17GB storage for model weights
- Optional: Linux kernel 5.1+ for io_uring optimization

**Deployment (Android):**
- Android NDK (cross-compilation from Linux)
- Android 11+ (API 30, Vulkan 1.2, thermal API)
- 8GB+ RAM, 17GB+ free storage
- adb for deployment

## Status

**Phase: Design complete (Rev 2), implementation starting.**

See `.claude/designs/architecture.md` for full design document.

- [ ] ggml integration + graph overhead measurement
- [ ] Expert reduction ablation (top-8/6/4 quality gate)
- [ ] .pmoe weight format + extraction script
- [ ] BPE tokenizer
- [ ] Platform I/O backends (pthreads+pread primary, io_uring Linux-only)
- [ ] Single-layer forward pass
- [ ] Full model inference + instrumentation
- [ ] Cross-layer expert prediction + prefetching
- [ ] Thermal adaptation
- [ ] Android cross-compilation + phone benchmarks

## License

MIT
