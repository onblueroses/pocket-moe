#include <metal_stdlib>
using namespace metal;

// FMA dequant kernel for 4-bit quantized expert weights.
// Reformulates dequant+matmul as fma(nibble, scale*x, bias*x)
// so the GPU's fused multiply-add unit does dequant+multiply in one instruction.
//
// Based on flash-moe's approach, adapted for iPhone Metal GPU.
//
// Weight layout per expert (Q4, group_size=128):
//   packed_weights: uint8 array, 2 nibbles per byte
//   scales: float16 array, one per group
//   zeros: float16 array, one per group (or biases)

kernel void expert_fma_dequant_matmul(
    device const uint8_t *packed_weights [[buffer(0)]],
    device const half *scales            [[buffer(1)]],
    device const half *zeros             [[buffer(2)]],
    device const half *input             [[buffer(3)]],
    device half *output                  [[buffer(4)]],
    constant uint &K                     [[buffer(5)]],  // input dim
    constant uint &N                     [[buffer(6)]],  // output dim
    constant uint &group_size            [[buffer(7)]],
    uint2 gid                            [[thread_position_in_grid]])
{
    uint row = gid.y;  // output element
    uint col = gid.x;  // reduction (processed in groups)

    if (row >= N) return;

    half acc = 0.0h;

    for (uint k = col; k < K; k += 2) {
        uint byte_idx = row * (K / 2) + k / 2;
        uint8_t packed = packed_weights[byte_idx];

        uint8_t lo = packed & 0x0F;
        uint8_t hi = (packed >> 4) & 0x0F;

        uint group_idx = k / group_size;
        half scale = scales[row * (K / group_size) + group_idx];
        half zero = zeros[row * (K / group_size) + group_idx];

        // FMA: weight = scale * nibble + zero
        // output += weight * input[k]
        // Combined: fma(nibble, scale * input[k], zero * input[k])
        half sx0 = scale * input[k];
        half zx0 = zero * input[k];
        acc = fma(half(lo), sx0, acc + zx0);

        if (k + 1 < K) {
            half sx1 = scale * input[k + 1];
            half zx1 = zero * input[k + 1];
            acc = fma(half(hi), sx1, acc + zx1);
        }
    }

    output[row] = acc;
}
