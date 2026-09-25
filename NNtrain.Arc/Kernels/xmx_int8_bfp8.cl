// Native signed INT8 XMX forward projection for block32 BFP8 operands.
// Each K32 dot has an exact INT32 accumulator, then its two block scales are
// applied in FP32. A single scale after the full K reduction would be wrong.
#if defined(ARC_XMX) && ARC_SG == 16 && defined(ARC_INT8_LINEAR)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

__kernel void xmx_i8_bfp8_pack_a(
    __global const char* source, __global ushort* panels,
    int rows, int k, int firstRow)
{
    int i = get_global_id(0);
    int rowGroups = (rows + 7) / 8;
    int kBlocks = k / 32;
    int total = rowGroups * kBlocks * 128;
    if (i >= total) return;
    int lane = i % 16;
    int localRow = (i / 16) % 8;
    int rowGroup = (i / 128) % rowGroups;
    int kBlock = i / (128 * rowGroups);
    int row = rowGroup * 8 + localRow;
    ushort packed = 0;
    if (row < rows) {
        int offset = (firstRow + row) * k + kBlock * 32 + lane * 2;
        packed = (ushort)(uchar)source[offset]
            | ((ushort)(uchar)source[offset + 1] << 8);
    }
    panels[i] = packed;
}

// Coalesce weight reads over a 32x32 tile, then transpose in local memory
// to the K32 x N16 INT8 DPAS panel ABI (one uint per lane and K4 packet).
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void xmx_i8_bfp8_pack_b(
    __global const char* source, __global uint* panels, int n, int k)
{
    int tid = get_local_id(0);
    int colBase = get_group_id(0) * 32;
    int kBase = get_group_id(1) * 32;
    int colGroups = n / 16;
    __local char values[32][33];
    for (int i = tid; i < 1024; i += 256) {
        int nn = i / 32, kk = i % 32;
        values[nn][kk] = source[(colBase + nn) * k + kBase + kk];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = tid; i < 256; i += 256) {
        int nn = (i / 128) * 16 + i % 16;
        int q4 = (i / 16) % 8;
        uint packed = (uint)(uchar)values[nn][q4 * 4]
            | ((uint)(uchar)values[nn][q4 * 4 + 1] << 8)
            | ((uint)(uchar)values[nn][q4 * 4 + 2] << 16)
            | ((uint)(uchar)values[nn][q4 * 4 + 3] << 24);
        int colGroup = (colBase + nn) / 16;
        panels[((kBase / 32) * colGroups + colGroup) * 128
            + q4 * 16 + nn % 16] = packed;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16,16,1)))
__kernel void xmx_i8_bfp8_linear_16x32(
    __global const ushort* aPanels, __global const uint* bPanels,
    __global const float* aScales, __global const float* bScales,
    __global char* output, __global float* outputScales,
    __global int* status, __global const float* bias,
    int rows, int n, int k, int relu, int firstRow)
{
    int lane = get_local_id(0), sg = get_local_id(1);
    int row = get_group_id(1) * 256 + sg * 16;
    int colBase = get_group_id(0) * 32;
    int rowGroups = (rows + 7) / 8, colGroups = n / 16;
    float8 sums[2][2];
    for (int rb = 0; rb < 2; rb++)
        for (int tile = 0; tile < 2; tile++)
            sums[rb][tile] = (float8)(bias[colBase + tile * 16 + lane]);

    for (int base = 0; base < k; base += 32) {
        short8 av[2];
        float8 aScale[2];
        for (int rb = 0; rb < 2; rb++) {
            av[rb] = (short8)(0);
            aScale[rb] = (float8)(0);
            int firstLocalRow = row + rb * 8;
            if (firstLocalRow < rows) {
                av[rb] = as_short8(intel_sub_group_block_read_us8(
                    aPanels + ((base / 32) * rowGroups + firstLocalRow / 8) * 128));
                for (int r = 0; r < 8; r++)
                    if (firstLocalRow + r < rows)
                        aScale[rb][r] = aScales[
                            ((firstRow + firstLocalRow + r) * k + base) / 32];
            }
        }
        for (int tile = 0; tile < 2; tile++) {
            int col = colBase + tile * 16 + lane;
            int8 bv = as_int8(intel_sub_group_block_read8(
                bPanels + ((base / 32) * colGroups + colBase / 16 + tile) * 128));
            float bScale = bScales[(col * k + base) / 32];
            for (int rb = 0; rb < 2; rb++) {
                int8 dot = intel_sub_group_i8_i8_matrix_mad_k32(
                    av[rb], bv, (int8)(0));
                sums[rb][tile] = fma(convert_float8(dot),
                    aScale[rb] * bScale, sums[rb][tile]);
            }
        }
    }

    for (int rb = 0; rb < 2; rb++) {
        for (int r = 0; r < 8; r++) {
            int localRow = row + rb * 8 + r;
            if (localRow >= rows) continue;
            float x = sums[rb][0][r], y = sums[rb][1][r];
            int valid = isfinite(x) && isfinite(y);
            if (relu) { x = fmax(0.0f, x); y = fmax(0.0f, y); }
            float maximum = fmax(fabs(x), fabs(y));
            for (int stride = 8; stride; stride /= 2) {
                maximum = fmax(maximum,
                    intel_sub_group_shuffle(maximum, lane ^ stride));
                valid &= intel_sub_group_shuffle(valid, lane ^ stride);
            }
            float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
            valid &= scale > 0.0f && isfinite(scale);
            int offset = (firstRow + localRow) * n + colBase;
            if (lane == 0) {
                outputScales[offset / 32] = valid ? scale : NAN;
                if (!valid) atomic_or(status, 1);
            }
            output[offset + lane] = valid
                ? convert_char_sat_rte(clamp(x / scale, -127.0f, 127.0f)) : 0;
            output[offset + lane + 16] = valid
                ? convert_char_sat_rte(clamp(y / scale, -127.0f, 127.0f)) : 0;
        }
    }
}
#endif
