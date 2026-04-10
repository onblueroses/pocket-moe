#ifndef POCKET_MOE_URING_H
#define POCKET_MOE_URING_H

#include <stdint.h>

// io_uring-based async expert weight loading.
// Submits multiple pread requests as a batch, waits for all completions.
// Falls back to pthreads + pread() if io_uring is unavailable.

typedef struct ExpertIO ExpertIO;

// Initialize the I/O subsystem. Probes for io_uring support.
// If io_uring is unavailable, returns a valid handle using pthreads + pread().
// queue_depth: max concurrent reads (typically 8-16)
// Returns NULL only on fatal errors (e.g. invalid fd). Use expert_io_has_uring()
// to check which backend is active.
ExpertIO *expert_io_create(int expert_fd, int32_t queue_depth);
void expert_io_free(ExpertIO *io);

// Returns 1 if using io_uring, 0 if using pread fallback.
int expert_io_has_uring(const ExpertIO *io);

// Submit batch of expert chunk reads.
// offsets: file offsets for each expert chunk
// sizes: byte sizes per chunk
// buffers: pre-allocated destination buffers
// count: number of chunks (typically num_active_experts, e.g. 8)
//
// This function submits all reads and returns immediately.
// Call expert_io_wait() to block until all complete.
int expert_io_submit(ExpertIO *io, const uint64_t *offsets,
                     const uint64_t *sizes, void **buffers,
                     int32_t count);

// Wait for all submitted reads to complete.
// Returns 0 on success, -1 if any read failed.
int expert_io_wait(ExpertIO *io);

// Check page cache residency for expert chunks before submitting I/O.
// Uses mincore() on the mmap'd expert file.
// resident_out: array of bools, set to 1 if chunk is in page cache.
// Only checks chunks, doesn't load them.
int expert_io_check_cache(ExpertIO *io, const uint64_t *offsets,
                          const uint64_t *sizes, int32_t count,
                          int *resident_out);

#endif // POCKET_MOE_URING_H
