using System.Globalization;

namespace NNtrain.Gui;

internal sealed record GenerationInput(int MaxNewTokens, GenerationSampling Sampling)
{
    internal static bool TryParse(string maxTokensText, string temperatureText, string topPText,
        string topKText, out GenerationInput? input, out string? error)
    {
        input = null;
        if (!int.TryParse(maxTokensText, out int maxTokens) || maxTokens is < 1 or > 8192)
        {
            error = "最大生成数は 1 ～ 8192 の整数で指定してください。";
            return false;
        }
        if (!float.TryParse(temperatureText, NumberStyles.Float, CultureInfo.InvariantCulture,
            out float temperature) || !float.IsFinite(temperature) || temperature is < 0f or > 2f)
        {
            error = "temperature は 0 ～ 2 の数値で指定してください。";
            return false;
        }
        if (!float.TryParse(topPText, NumberStyles.Float, CultureInfo.InvariantCulture,
            out float topP) || !float.IsFinite(topP) || topP is <= 0f or > 1f)
        {
            error = "top-p は 0 より大きく 1 以下の数値で指定してください。";
            return false;
        }
        if (!int.TryParse(topKText, out int topK) || topK is < 1 or > 256)
        {
            error = "top-k は 1 ～ 256 の整数で指定してください。";
            return false;
        }
        input = new GenerationInput(maxTokens, new GenerationSampling(temperature, topP, topK));
        error = null;
        return true;
    }
}
