#include <metal_stdlib>
using namespace metal;

// Test 1: streaming read. Each thread sums a strided slice of the buffer.
// Consecutive threads read consecutive float4 values, so reads are coalesced.
// One float per thread is written so the compiler cannot remove the loads.
kernel void stream_read(device const float4 *src [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        constant uint &n4 [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint gsize [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n4; i += gsize) {
        acc += src[i];
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}

// Test 2: Q4_0 matrix-vector product, y = W x.
// Layout is llama.cpp's block_q4_0: 1 float16 scale + 16 bytes holding 32 4-bit
// weights, 18 bytes in total. Byte j holds weight j in the low nibble and
// weight j+16 in the high nibble. Each weight is (nibble - 8) * scale.
struct block_q4_0 {
    half d;
    uchar qs[16];
};

constant uint ROWS_PER_TG = 4; // one simdgroup (32 lanes) per output row

kernel void q4_0_matvec(device const block_q4_0 *w [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant uint &rows [[buffer(3)]],
                        constant uint &blocks_per_row [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * ROWS_PER_TG + sg;
    if (row >= rows) {
        return;
    }
    device const block_q4_0 *wr = w + (ulong)row * blocks_per_row;
    float sum = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32) {
        device const block_q4_0 &blk = wr[b];
        device const float *xb = x + b * 32;
        float d = (float)blk.d;
        float s = 0.0f;
        for (uint j = 0; j < 16; j++) {
            uchar q = blk.qs[j];
            s += (float)((int)(q & 0x0F) - 8) * xb[j];
            s += (float)((int)(q >> 4) - 8) * xb[j + 16];
        }
        sum += d * s;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}
