#ifndef POCKET_MOE_STORAGE_H
#define POCKET_MOE_STORAGE_H

#include <stdint.h>

// Parallel pread() via thread pool for expert chunk loading.
// Reads multiple expert chunks concurrently from the expert weight file.
//
// fd: file descriptor for the expert weight file (opened with O_RDONLY)
// offsets: array of file offsets for each expert chunk
// sizes: array of byte sizes for each chunk
// buffers: array of destination pointers (pre-allocated)
// count: number of chunks to read
//
// Returns 0 on success, -1 on any read failure.
int storage_parallel_pread(int fd, const uint64_t *offsets,
                           const uint64_t *sizes, void **buffers,
                           int32_t count);

// Measure page cache residency for a memory region.
// Uses mincore() to check which pages are in cache.
// Returns fraction [0.0, 1.0] of pages resident.
double storage_cache_residency(const void *addr, uint64_t length);

// Prefetch a memory region into page cache.
// Uses madvise(MADV_WILLNEED) to trigger async kernel readahead.
// Non-blocking: returns immediately, kernel fetches in background.
int storage_prefetch(const void *addr, uint64_t length);

#endif // POCKET_MOE_STORAGE_H
