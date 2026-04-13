#ifndef POCKET_MOE_PREDICTOR_H
#define POCKET_MOE_PREDICTOR_H

#include <stdint.h>
#include "inference.h"

// Cross-layer expert predictor.
// Uses current layer's gate logits to predict next layer's activated experts.
// Adjacent layers have 88-93% cosine similarity in gate inputs (Fate paper),
// enabling 97%+ prefetch accuracy.
//
// Prediction happens during current-layer GPU compute (steps 5-6).
// Predicted experts are prefetched via madvise(MADV_WILLNEED) so they're
// in page cache when the next layer needs them.

typedef struct ExpertPredictor ExpertPredictor;

// Create predictor for a model. Initially uses gate-logit-reuse heuristic
// (predict next layer = current layer experts). Can be upgraded with a
// learned linear transform trained offline.
ExpertPredictor *predictor_create(const ModelConfig *config);
void predictor_free(ExpertPredictor *pred);

// Predict next layer's top-K experts from current layer's gate logits.
// gate_logits: [num_experts] float (pre-softmax gate output from current layer)
// predicted_out: buffer for predicted expert indices (caller-allocated)
// max_predicted: capacity of predicted_out buffer (use runtime active expert count,
//                which may be reduced from model default under thermal pressure)
// Returns number of predicted experts written (<= max_predicted).
int32_t predictor_predict(ExpertPredictor *pred, int32_t current_layer,
                          const float *gate_logits,
                          int32_t *predicted_out, int32_t max_predicted);

// Record actual expert activations for accuracy tracking.
// Call after routing to log prediction accuracy per layer.
void predictor_record_actual(ExpertPredictor *pred, int32_t layer,
                             const int32_t *actual_indices, int32_t num_active);

// Get prediction accuracy stats (for profiling/experiments).
typedef struct {
    double accuracy;         // fraction of predicted experts that were correct
    int32_t total_predicted;
    int32_t total_correct;
} PredictorStats;

void predictor_get_stats(const ExpertPredictor *pred, PredictorStats *stats);

#endif // POCKET_MOE_PREDICTOR_H
