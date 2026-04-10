#ifndef POCKET_MOE_EXPERT_H
#define POCKET_MOE_EXPERT_H

#include <stdint.h>
#include "inference.h"

// Expert weight chunk descriptor
typedef struct {
    int32_t layer_idx;
    int32_t expert_idx;
    uint64_t file_offset;    // offset in expert weight file
    uint64_t chunk_size;     // bytes per expert (gate + up + down projections)
} ExpertChunk;

// Expert pool: manages mmap'd expert weights on storage
typedef struct ExpertPool ExpertPool;

ExpertPool *expert_pool_create(const char *expert_file, const ModelConfig *config);
void expert_pool_free(ExpertPool *pool);

// Load experts for one layer. Dispatches parallel pread() via GCD.
// expert_indices: array of num_active expert IDs selected by router
// Returns pointers to expert weight data (may be from page cache or fresh read).
int expert_pool_load(ExpertPool *pool, int32_t layer_idx,
                     const int32_t *expert_indices, int32_t num_active,
                     const void **expert_weights_out,
                     int32_t *cache_hits_out, int32_t *cache_misses_out);

// Expert forward pass on Metal GPU with FMA dequant
// input: [hidden_size] float16
// output: [hidden_size] float16 (accumulated from all active experts)
int expert_forward_metal(const void **expert_weights, int32_t num_active,
                         const void *input, void *output,
                         const ModelConfig *config);

#endif // POCKET_MOE_EXPERT_H
