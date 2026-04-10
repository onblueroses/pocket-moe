#ifndef POCKET_MOE_THERMAL_H
#define POCKET_MOE_THERMAL_H

#include "inference.h"

// Initialize thermal monitoring (registers for iOS thermal notifications)
void thermal_init(void);

// Poll current thermal state
ThermalState thermal_get_state(void);

// Update thermal config based on current state and performance trend.
// Called once per token. Adjusts active_experts_override and cooldown_ms.
void thermal_adapt(ThermalConfig *config, double current_tok_per_sec);

// Log thermal event for profiling
void thermal_log(ThermalState state, double tok_per_sec, int32_t active_experts);

#endif // POCKET_MOE_THERMAL_H
