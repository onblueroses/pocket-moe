# Thermal Profiling

## Background

Phone thermal throttling is the primary constraint for sustained MoE inference. Published data:

- iPhone 16 Pro: 40 tok/s -> 22.6 tok/s (44% drop) within 2 LLM inferences
- Galaxy S24 Ultra: hard GPU frequency floor at iteration 6, terminates inference
- Hailo-10H NPU: sustains 7 tok/s at 1.87W with stable thermals

Source: "LLM Inference at the Edge: Thermal Analysis" (arXiv:2603.23640)

## Thermal-Adaptive Scheduling Design

### Signals (platform-abstracted)

| Platform | Primary Signal | Backup Signal |
|----------|---------------|---------------|
| Linux desktop | sysfs thermal zones | tok/s decline |
| Android | AThermal API (API 30+) | tok/s decline |
| Fallback | tok/s decline only | N/A |

tok/s decline is the most reliable signal across platforms. AThermal reports
thresholds (not temperatures) and some devices return THERMAL_STATUS_NONE
regardless of actual state.

### Response Levels

| Thermal State | Expert Action | I/O Action | Trigger |
|---------------|--------------|------------|---------|
| nominal | 8 active (full) | Normal | Default |
| fair | 6 active (top-6) | Skip prefetch | AThermal FAIR or >10% tok/s drop |
| serious | 4 active (top-4)* | 50ms cooldown | AThermal SERIOUS or >30% tok/s drop |
| critical | Pause inference | Wait for nominal | AThermal CRITICAL or >50% tok/s drop |

*Top-4 routing quality must be validated in Phase 0.5 ablation. If quality is
unacceptable, "serious" falls back to top-6 with longer cooldown pauses instead.

### Important: Storage Thermal Coupling

Reducing GPU expert count reduces GPU heat but doesn't reduce storage I/O heat.
UFS controllers share the SoC thermal envelope and throttle independently.
The cooldown pauses at "serious" level address both GPU and storage thermal budgets.

### Open Questions
- Does reducing active experts from 8 to 6 noticeably hurt output quality on Qwen3-30B-A3B? (Phase 0.5 ablation)
- Is it better to throttle experts or insert cooldown pauses? (Measure both)
- Can we predict thermal trajectory and preemptively throttle before hitting "serious"?
- How does UFS controller thermal throttling interact with expert loading latency?

## Measurements

| Device | Test | Duration | Start tok/s | End tok/s | Thermal state | Notes |
|--------|------|----------|-------------|-----------|---------------|-------|
| | | | | | | |
