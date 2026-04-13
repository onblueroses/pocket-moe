#ifndef POCKET_MOE_INFERENCE_H
#define POCKET_MOE_INFERENCE_H

#include <stdint.h>

// Forward declarations
typedef struct PocketModel PocketModel;
typedef struct InferenceState InferenceState;

// Model configuration (parsed from config.json)
typedef struct {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_attention_heads;
    int32_t num_kv_heads;
    int32_t intermediate_size;
    int32_t max_seq_len;

    // MoE config
    int32_t num_experts;         // total experts per layer (e.g. 128)
    int32_t num_active_experts;  // experts activated per token (e.g. 8)
    int32_t num_shared_experts;  // always-active shared experts (e.g. 0 for Qwen3)
    int32_t expert_hidden_size;  // per-expert FFN hidden dim

    // Quantization
    int32_t quant_bits;          // 4 for Q4
    int32_t group_size;          // quantization group size (e.g. 128)
} ModelConfig;

// Thermal state (platform-abstracted: sysfs on Linux, AThermal on Android)
typedef enum {
    THERMAL_NOMINAL = 0,
    THERMAL_FAIR = 1,
    THERMAL_SERIOUS = 2,
    THERMAL_CRITICAL = 3
} ThermalState;

// Thermal-adaptive config
typedef struct {
    ThermalState current_state;
    int32_t active_experts_override;  // -1 = use default, else override
    int32_t cooldown_ms;              // inter-token cooldown
    double last_tok_per_sec;
} ThermalConfig;

// Inference timing for a single token
typedef struct {
    double attention_ms;
    double router_ms;
    double expert_io_ms;     // time reading experts from storage
    double expert_compute_ms;
    double total_ms;
    int32_t cache_hits;      // experts served from page cache
    int32_t cache_misses;    // experts read from storage
} TokenTiming;

// Core API
PocketModel *pocket_model_load(const char *model_dir, const ModelConfig *config);
void pocket_model_free(PocketModel *model);

InferenceState *pocket_inference_create(PocketModel *model);
void pocket_inference_free(InferenceState *state);

// Generate one token. Returns token ID.
int32_t pocket_inference_step(InferenceState *state, int32_t input_token,
                              ThermalConfig *thermal, TokenTiming *timing);

// Batch prompt encoding (prefill)
void pocket_inference_prefill(InferenceState *state, const int32_t *tokens,
                              int32_t num_tokens, ThermalConfig *thermal);

#endif // POCKET_MOE_INFERENCE_H
