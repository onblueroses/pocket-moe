# Thermal Profiling

## Background

Phone thermal throttling is the primary constraint for sustained MoE inference. Published data:

- iPhone 16 Pro: 40 tok/s -> 22.6 tok/s (44% drop) within 2 LLM inferences
- Galaxy S24 Ultra: hard GPU frequency floor at iteration 6, terminates inference
- Hailo-10H NPU: sustains 7 tok/s at 1.87W with stable thermals

Source: "LLM Inference at the Edge: Thermal Analysis" (arXiv:2603.23640)

## Thermal-Adaptive Scheduling Design

### Signals
- `thermal_state` from `ProcessInfo` (nominal/fair/serious/critical)
- Metal GPU utilization from `MTLDevice`
- Inference timing drift (tok/s declining = throttling in progress)

### Response Levels

| Thermal State | Action |
|---------------|--------|
| nominal | Full speed, all 8 experts active |
| fair | Reduce to 6 experts (top-6 routing), skip prefetch |
| serious | Reduce to 4 experts, insert 100ms cooldown between tokens |
| critical | Pause inference, notify user, resume when nominal |

### Open Questions
- Does reducing active experts from 8 to 6 noticeably hurt output quality on Qwen3-30B-A3B?
- Is it better to throttle experts or insert cooldown pauses?
- Can we predict thermal trajectory and preemptively throttle before hitting "serious"?

## Measurements

| Device | Test | Duration | Start tok/s | End tok/s | Thermal state | Notes |
|--------|------|----------|-------------|-----------|---------------|-------|
| | | | | | | |
