#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>

#define NNTRAIN_EXPORT extern "C" __declspec(dllexport)

namespace
{
bool HasF16cAndOsAvxState()
{
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    constexpr int OsXsave = 1 << 27;
    constexpr int Avx = 1 << 28;
    constexpr int F16c = 1 << 29;
    if ((registers[2] & (OsXsave | Avx | F16c))
        != (OsXsave | Avx | F16c))
    {
        return false;
    }

    if ((_xgetbv(0) & 0x6) != 0x6)
        return false;

    __cpuidex(registers, 7, 0);
    constexpr int Avx2 = 1 << 5;
    return (registers[1] & Avx2) != 0;
}

float HalfToFloat(std::uint16_t half)
{
    const std::uint32_t sign = (half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1fu;
    const std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
            float result;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }

        // binary16 subnormal: mantissa * 2^-24.
        float result = static_cast<float>(mantissa) * 0x1.0p-24f;
        return sign == 0 ? result : -result;
    }

    if (exponent == 0x1fu)
    {
        // Preserve IEEE-754 infinities and NaNs for scalar tails and bias.
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float DotF16(const std::uint16_t* left, const std::uint16_t* right, int length)
{
    __m256 sum = _mm256_setzero_ps();
    int index = 0;
    for (; index + 8 <= length; index += 8)
    {
        const __m128i leftHalf = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(left + index));
        const __m128i rightHalf = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(right + index));
        sum = _mm256_add_ps(
            sum,
            _mm256_mul_ps(
                _mm256_cvtph_ps(leftHalf),
                _mm256_cvtph_ps(rightHalf)));
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3]
        + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; index < length; ++index)
        result += HalfToFloat(left[index]) * HalfToFloat(right[index]);
    return result;
}

inline float HorizontalSum(__m256 values)
{
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3]
        + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

inline std::uint16_t FloatToHalf(float value)
{
    const __m128 input = _mm_set_ss(value);
    const __m128i packed = _mm_cvtps_ph(input, _MM_FROUND_TO_NEAREST_INT);
    return static_cast<std::uint16_t>(_mm_cvtsi128_si32(packed));
}

void ForwardFourColumns(
    const std::uint16_t* input,
    const std::uint16_t* weight0,
    const std::uint16_t* weight1,
    const std::uint16_t* weight2,
    const std::uint16_t* weight3,
    float bias0,
    float bias1,
    float bias2,
    float bias3,
    std::uint16_t* output,
    int inputWidth,
    int applyRelu)
{
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    int index = 0;
    for (; index + 8 <= inputWidth; index += 8)
    {
        const __m256 inputValues = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(input + index)));
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(inputValues,
            _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight0 + index)))));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(inputValues,
            _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight1 + index)))));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(inputValues,
            _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight2 + index)))));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(inputValues,
            _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight3 + index)))));
    }

    float value0 = bias0 + HorizontalSum(sum0);
    float value1 = bias1 + HorizontalSum(sum1);
    float value2 = bias2 + HorizontalSum(sum2);
    float value3 = bias3 + HorizontalSum(sum3);
    for (; index < inputWidth; ++index)
    {
        const float inputValue = HalfToFloat(input[index]);
        value0 += inputValue * HalfToFloat(weight0[index]);
        value1 += inputValue * HalfToFloat(weight1[index]);
        value2 += inputValue * HalfToFloat(weight2[index]);
        value3 += inputValue * HalfToFloat(weight3[index]);
    }
    const __m128i packed = _mm_cvtps_ph(
        _mm_setr_ps(
            applyRelu && value0 <= 0.0f ? 0.0f : value0,
            applyRelu && value1 <= 0.0f ? 0.0f : value1,
            applyRelu && value2 <= 0.0f ? 0.0f : value2,
            applyRelu && value3 <= 0.0f ? 0.0f : value3),
        _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(output), packed);
}

// Processes two independent activation rows against four shared weight rows.
// Each weight vector is widened once and reused for both output rows.  The
// add/multiply order within either row stays identical to ForwardFourColumns,
// which keeps the non-FMA numerical contract unchanged while reducing F16C
// conversion and weight-load traffic for training batches.
void ForwardTwoRowsFourColumns(
    const std::uint16_t* input0,
    const std::uint16_t* input1,
    const std::uint16_t* weight0,
    const std::uint16_t* weight1,
    const std::uint16_t* weight2,
    const std::uint16_t* weight3,
    float bias0,
    float bias1,
    float bias2,
    float bias3,
    std::uint16_t* output0,
    std::uint16_t* output1,
    int inputWidth,
    int applyRelu)
{
    __m256 sum00 = _mm256_setzero_ps();
    __m256 sum01 = _mm256_setzero_ps();
    __m256 sum02 = _mm256_setzero_ps();
    __m256 sum03 = _mm256_setzero_ps();
    __m256 sum10 = _mm256_setzero_ps();
    __m256 sum11 = _mm256_setzero_ps();
    __m256 sum12 = _mm256_setzero_ps();
    __m256 sum13 = _mm256_setzero_ps();
    int index = 0;
    for (; index + 8 <= inputWidth; index += 8)
    {
        const __m256 inputValues0 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(input0 + index)));
        const __m256 inputValues1 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(input1 + index)));
        const __m256 weightValues0 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight0 + index)));
        const __m256 weightValues1 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight1 + index)));
        const __m256 weightValues2 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight2 + index)));
        const __m256 weightValues3 = _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(weight3 + index)));
        sum00 = _mm256_add_ps(sum00, _mm256_mul_ps(inputValues0, weightValues0));
        sum01 = _mm256_add_ps(sum01, _mm256_mul_ps(inputValues0, weightValues1));
        sum02 = _mm256_add_ps(sum02, _mm256_mul_ps(inputValues0, weightValues2));
        sum03 = _mm256_add_ps(sum03, _mm256_mul_ps(inputValues0, weightValues3));
        sum10 = _mm256_add_ps(sum10, _mm256_mul_ps(inputValues1, weightValues0));
        sum11 = _mm256_add_ps(sum11, _mm256_mul_ps(inputValues1, weightValues1));
        sum12 = _mm256_add_ps(sum12, _mm256_mul_ps(inputValues1, weightValues2));
        sum13 = _mm256_add_ps(sum13, _mm256_mul_ps(inputValues1, weightValues3));
    }

    float value00 = bias0 + HorizontalSum(sum00);
    float value01 = bias1 + HorizontalSum(sum01);
    float value02 = bias2 + HorizontalSum(sum02);
    float value03 = bias3 + HorizontalSum(sum03);
    float value10 = bias0 + HorizontalSum(sum10);
    float value11 = bias1 + HorizontalSum(sum11);
    float value12 = bias2 + HorizontalSum(sum12);
    float value13 = bias3 + HorizontalSum(sum13);
    for (; index < inputWidth; ++index)
    {
        const float inputValue0 = HalfToFloat(input0[index]);
        const float inputValue1 = HalfToFloat(input1[index]);
        const float weightValue0 = HalfToFloat(weight0[index]);
        const float weightValue1 = HalfToFloat(weight1[index]);
        const float weightValue2 = HalfToFloat(weight2[index]);
        const float weightValue3 = HalfToFloat(weight3[index]);
        value00 += inputValue0 * weightValue0;
        value01 += inputValue0 * weightValue1;
        value02 += inputValue0 * weightValue2;
        value03 += inputValue0 * weightValue3;
        value10 += inputValue1 * weightValue0;
        value11 += inputValue1 * weightValue1;
        value12 += inputValue1 * weightValue2;
        value13 += inputValue1 * weightValue3;
    }
    const __m128i packed0 = _mm_cvtps_ph(
        _mm_setr_ps(
            applyRelu && value00 <= 0.0f ? 0.0f : value00,
            applyRelu && value01 <= 0.0f ? 0.0f : value01,
            applyRelu && value02 <= 0.0f ? 0.0f : value02,
            applyRelu && value03 <= 0.0f ? 0.0f : value03),
        _MM_FROUND_TO_NEAREST_INT);
    const __m128i packed1 = _mm_cvtps_ph(
        _mm_setr_ps(
            applyRelu && value10 <= 0.0f ? 0.0f : value10,
            applyRelu && value11 <= 0.0f ? 0.0f : value11,
            applyRelu && value12 <= 0.0f ? 0.0f : value12,
            applyRelu && value13 <= 0.0f ? 0.0f : value13),
        _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(output0), packed0);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(output1), packed1);
}

float GradientValue(
    const float* outputGradient,
    const std::uint16_t* output,
    int index,
    int applyRelu)
{
    // Match the managed ReLU backward condition exactly: NaN is not <= 0,
    // so it remains observable rather than being silently masked away.
    return applyRelu && HalfToFloat(output[index]) <= 0.0f
        ? 0.0f
        : outputGradient[index];
}
}

NNTRAIN_EXPORT int nntrain_f16c_available()
{
    return HasF16cAndOsAvxState() ? 1 : 0;
}

NNTRAIN_EXPORT void nntrain_f16_linear_forward(
    const std::uint16_t* input,
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    std::uint16_t* output,
    int rowStart,
    int rowCount,
    int inputWidth,
    int outputWidth,
    int applyRelu)
{
    const int rowEnd = rowStart + rowCount;
    int row = rowStart;
    for (; row + 1 < rowEnd; row += 2)
    {
        const std::uint16_t* inputRow0 = input + row * inputWidth;
        const std::uint16_t* inputRow1 = inputRow0 + inputWidth;
        std::uint16_t* outputRow0 = output + row * outputWidth;
        std::uint16_t* outputRow1 = outputRow0 + outputWidth;
        int column = 0;
        for (; column + 4 <= outputWidth; column += 4)
        {
            ForwardTwoRowsFourColumns(
                inputRow0,
                inputRow1,
                weight + column * inputWidth,
                weight + (column + 1) * inputWidth,
                weight + (column + 2) * inputWidth,
                weight + (column + 3) * inputWidth,
                HalfToFloat(bias[column]),
                HalfToFloat(bias[column + 1]),
                HalfToFloat(bias[column + 2]),
                HalfToFloat(bias[column + 3]),
                outputRow0 + column,
                outputRow1 + column,
                inputWidth,
                applyRelu);
        }
        for (; column < outputWidth; ++column)
        {
            float value = HalfToFloat(bias[column])
                + DotF16(inputRow0, weight + column * inputWidth, inputWidth);
            outputRow0[column] = FloatToHalf(
                applyRelu && value <= 0.0f ? 0.0f : value);
            value = HalfToFloat(bias[column])
                + DotF16(inputRow1, weight + column * inputWidth, inputWidth);
            outputRow1[column] = FloatToHalf(
                applyRelu && value <= 0.0f ? 0.0f : value);
        }
    }

    if (row < rowEnd)
    {
        const std::uint16_t* inputRow = input + row * inputWidth;
        std::uint16_t* outputRow = output + row * outputWidth;
        int column = 0;
        for (; column + 4 <= outputWidth; column += 4)
        {
            ForwardFourColumns(
                inputRow,
                weight + column * inputWidth,
                weight + (column + 1) * inputWidth,
                weight + (column + 2) * inputWidth,
                weight + (column + 3) * inputWidth,
                HalfToFloat(bias[column]),
                HalfToFloat(bias[column + 1]),
                HalfToFloat(bias[column + 2]),
                HalfToFloat(bias[column + 3]),
                outputRow + column,
                inputWidth,
                applyRelu);
        }
        for (; column < outputWidth; ++column)
        {
            float value = HalfToFloat(bias[column])
                + DotF16(inputRow, weight + column * inputWidth, inputWidth);
            outputRow[column] = FloatToHalf(
                applyRelu && value <= 0.0f ? 0.0f : value);
        }
    }
}

NNTRAIN_EXPORT void nntrain_f16_linear_backward_input(
    const float* outputGradient,
    const std::uint16_t* output,
    const std::uint16_t* weight,
    float* inputGradient,
    int rowStart,
    int rowCount,
    int inputWidth,
    int outputWidth,
    int applyRelu)
{
    const int rowEnd = rowStart + rowCount;
    for (int row = rowStart; row < rowEnd; ++row)
    {
        const float* gradientRow = outputGradient + row * outputWidth;
        const std::uint16_t* outputRow = output + row * outputWidth;
        float* inputGradientRow = inputGradient + row * inputWidth;
        for (int column = 0; column < outputWidth; ++column)
        {
            const float gradient = GradientValue(
                gradientRow,
                outputRow,
                column,
                applyRelu);
            if (gradient == 0.0f)
                continue;

            const std::uint16_t* weightRow = weight + column * inputWidth;
            const __m256 gradientVector = _mm256_set1_ps(gradient);
            int inputIndex = 0;
            for (; inputIndex + 8 <= inputWidth; inputIndex += 8)
            {
                const __m256 contribution = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(weightRow + inputIndex))),
                    gradientVector);
                const __m256 existing = _mm256_loadu_ps(
                    inputGradientRow + inputIndex);
                _mm256_storeu_ps(
                    inputGradientRow + inputIndex,
                    _mm256_add_ps(existing, contribution));
            }
            for (; inputIndex < inputWidth; ++inputIndex)
            {
                inputGradientRow[inputIndex] +=
                    HalfToFloat(weightRow[inputIndex]) * gradient;
            }
        }
    }
}

NNTRAIN_EXPORT void nntrain_f16_linear_backward_weight(
    const std::uint16_t* input,
    const float* outputGradient,
    const std::uint16_t* output,
    float* weightGradient,
    float* biasGradient,
    int columnStart,
    int columnCount,
    int rows,
    int inputWidth,
    int outputWidth,
    int applyRelu)
{
    const int columnEnd = columnStart + columnCount;
    for (int column = columnStart; column < columnEnd; ++column)
    {
        float* weightGradientRow = weightGradient + column * inputWidth;
        const float* gradientColumn = outputGradient + column;
        const std::uint16_t* outputColumn = output + column;
        float biasSum = 0.0f;

        for (int row = 0; row < rows; ++row)
        {
            const float gradient = GradientValue(
                gradientColumn,
                outputColumn,
                0,
                applyRelu);
            if (gradient != 0.0f)
            {
                const std::uint16_t* inputRow = input + row * inputWidth;
                const __m256 gradientVector = _mm256_set1_ps(gradient);
                int inputIndex = 0;
                for (; inputIndex + 8 <= inputWidth; inputIndex += 8)
                {
                    const __m256 contribution = _mm256_mul_ps(
                        _mm256_cvtph_ps(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(inputRow + inputIndex))),
                        gradientVector);
                    const __m256 existing = _mm256_loadu_ps(
                        weightGradientRow + inputIndex);
                    _mm256_storeu_ps(
                        weightGradientRow + inputIndex,
                        _mm256_add_ps(existing, contribution));
                }
                for (; inputIndex < inputWidth; ++inputIndex)
                {
                    weightGradientRow[inputIndex] +=
                        HalfToFloat(inputRow[inputIndex]) * gradient;
                }
            }
            biasSum += gradient;
            gradientColumn += outputWidth;
            outputColumn += outputWidth;
        }
        biasGradient[column] += biasSum;
    }
}
