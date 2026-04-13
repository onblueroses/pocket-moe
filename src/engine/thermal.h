#ifndef POCKET_MOE_THERMAL_H
#define POCKET_MOE_THERMAL_H

#include "inference.h"

// Platform-abstracted thermal monitoring backend
typedef struct ThermalBackend ThermalBackend;

// Initialize thermal monitoring.
// Linux: reads /sys/class/thermal/thermal_zone*/temp
// Android: AThermal_getCurrentThermalStatus() (NDK API 30+)
// Fallback: tok/s decline detection (no OS API needed)
ThermalBackend *thermal_create(void);
void thermal_free(ThermalBackend *backend);

// Poll current thermal state
ThermalState thermal_get_state(ThermalBackend *backend);

// Update thermal config based on current state and performance trend.
// Called once per token. Adjusts active_experts_override and cooldown_ms.
// Uses tok/s decline as primary signal (most reliable across platforms).
void thermal_adapt(ThermalBackend *backend, ThermalConfig *config,
                   double current_tok_per_sec);

// Log thermal event for profiling
void thermal_log(ThermalState state, double tok_per_sec, int32_t active_experts);

#endif // POCKET_MOE_THERMAL_H
