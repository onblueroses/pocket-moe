#ifndef POCKET_MOE_ROUTER_H
#define POCKET_MOE_ROUTER_H

#include <stdint.h>
#include "inference.h"

// Router output
typedef struct {
    int32_t expert_indices[16];  // max 16 active experts
    float expert_weights[16];   // softmax routing weights
    int32_t num_active;
    double router_ms;
} RouterOutput;

// Compute top-K expert routing on CPU.
// gate_logits: [num_experts] float from gate projection
// config: model config for num_active_experts
// thermal: may reduce num_active under thermal pressure
void router_topk(const float *gate_logits, int32_t num_experts,
                 const ThermalConfig *thermal, const ModelConfig *config,
                 RouterOutput *output);

#endif // POCKET_MOE_ROUTER_H
