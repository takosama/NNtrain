// Loss-head logits are already rounded to BF16 in the GEMM epilogue. Keep
// those values in two bytes until backward, then restore exact FP32 values.
inline ushort loss_head_bf16_bits(float value)
{
    uint bits = as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return (ushort)((bits >> 16) | 0x0040u);
    return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__kernel void loss_head_pack_bf16_at(
    __global const float* source, __global ushort* target, int count, int offset)
{
    int i = get_global_id(0);
    if (i < count) target[offset + i] = loss_head_bf16_bits(source[i]);
}

__kernel void loss_head_unpack_bf16_at(
    __global const ushort* source, __global float* target, int count, int offset)
{
    int i = get_global_id(0);
    if (i < count) target[i] = as_float(((uint)source[offset + i]) << 16);
}
