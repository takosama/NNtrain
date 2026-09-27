using System.Security.Cryptography;
using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35LoraModelTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ZeroAdapterFullSequenceLossMatchesIncrementalTeacherForcingAndResponseMask(bool tiedOutput)
    {
        RequireArc();
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput);
        using Qwen35QuantizedModel model = Load(fixture.Path);
        int[] tokens = [1, 2, 3, 0, 2];
        float[][] baseline = Logits(model, tokens[..^1]);
        model.AttachLora(new() { Rank = 2, Alpha = 4, IncludeOutput = true });
        float[][] attached = Logits(model, tokens[..^1]);
        for (int t = 0; t < baseline.Length; ++t) Assert.Equal(baseline[t], attached[t]);
        foreach (int responseStart in new[] { 1, 3 })
        {
            double expected = Enumerable.Range(responseStart - 1, tokens.Length - responseStart)
                .Average(t => CrossEntropy(baseline[t], tokens[t + 1]));
            double actual = model.EvaluateLoraLoss(tokens, responseStart);
            Close(expected, actual, 3e-5);
            Qwen35LoraStepResult gradients = model.ComputeLoraGradients(tokens, responseStart);
            Close(actual, gradients.Loss, 1e-6);
            Assert.Equal(tokens.Length - responseStart, gradients.SupervisedTokens);
            Assert.True(double.IsFinite(gradients.GradientNorm) && gradients.GradientNorm > 0);
            Assert.Equal(0, model.LoraStep);
        }
    }

    [Fact]
    public void HybridAttentionAndMlpAdapterGradientsMatchFullModelFiniteDifferences()
    {
        RequireArc();
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using Qwen35QuantizedModel model = Load(fixture.Path);
        model.AttachLora(new() { Rank = 2, Alpha = 4 });
        var random = new Random(719);
        foreach (Qwen35LoraMatrix adapter in model.LoraMatrices.Values)
        {
            float[][] state = adapter.ReadState();
            for (int i = 0; i < state[0].Length; ++i) state[0][i] = (float)(random.NextDouble() * .16 - .08);
            for (int i = 0; i < state[1].Length; ++i) state[1][i] = (float)(random.NextDouble() * .2 - .1);
            adapter.RestoreState(state, training: true);
        }
        int[] tokens = [1, 2, 3, 0, 1];
        Qwen35LoraStepResult result = model.ComputeLoraGradients(tokens, responseStartIndex: 2);
        Assert.True(double.IsFinite(result.GradientNorm) && result.GradientNorm > 0);
        string[] names = ["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
            "blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight", "blk.1.attn_output.weight"];
        foreach (string name in names)
        {
            Qwen35LoraMatrix adapter = model.LoraMatrices[name];
            float[][] analytic = adapter.ReadGradients(), state = adapter.ReadState();
            for (int parameter = 0; parameter < 2; ++parameter)
            {
                int index = Enumerable.Range(0, analytic[parameter].Length)
                    .MaxBy(i => Math.Abs(analytic[parameter][i]));
                float gradient = analytic[parameter][index];
                Assert.True(float.IsFinite(gradient) && Math.Abs(gradient) > 1e-7,
                    $"Disconnected or non-finite gradient: {name}, parameter {parameter}, gradient {gradient:R}.");
                float original = state[parameter][index];
                // CE is reduced to FP32 on device. A 0.01 perturbation keeps
                // weak early-layer derivatives above the loss readback noise.
                const float epsilon = .01f;
                double plus, minus;
                try
                {
                    state[parameter][index] = original + epsilon;
                    adapter.RestoreState(state, training: true);
                    plus = model.EvaluateLoraLoss(tokens, 2);
                    state[parameter][index] = original - epsilon;
                    adapter.RestoreState(state, training: true);
                    minus = model.EvaluateLoraLoss(tokens, 2);
                }
                finally
                {
                    state[parameter][index] = original;
                    adapter.RestoreState(state, training: true);
                }
                double numeric = (plus - minus) / (2 * epsilon);
                double tolerance = 3e-5 + .06 * Math.Max(Math.Abs(numeric), Math.Abs(gradient));
                Assert.True(Math.Abs(numeric - gradient) <= tolerance,
                    $"{name} {(parameter == 0 ? "A" : "B")}[{index}]: analytic={gradient:R}, finite difference={numeric:R}, tolerance={tolerance:R}.");
            }
        }
    }

    [Fact]
    public void TrainingUpdatesAttentionAndMlpLowersLossAndKeepsBaseFrozen()
    {
        RequireArc();
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        byte[] fileHash = Hash(fixture.Path);
        using Qwen35QuantizedModel model = Load(fixture.Path);
        model.AttachLora(new() { Rank = 2, Alpha = 4, LearningRate = .003f, GradientClip = 5 });
        int[] tokens = [1, 2, 3, 0, 2];
        long[] resident = model.ResidentWeightBytes.ToArray();
        float[][] baseLogits = Logits(model, tokens[..^1]);
        var original = model.LoraMatrices.ToDictionary(pair => pair.Key, pair => pair.Value.ReadState());
        double initialLoss = model.EvaluateLoraLoss(tokens, 2);
        for (int step = 1; step <= 6; ++step)
        {
            long uploaded = model.UploadedBytes.Sum();
            Qwen35LoraStepResult result = model.TrainLora(tokens, 2);
            Assert.Equal(step, model.LoraStep);
            Assert.Equal(step, result.Step);
            Assert.True(double.IsFinite(result.Loss) && double.IsFinite(result.GradientNorm));
            Assert.Equal(tokens.Length - 2, result.SupervisedTokens);
            Assert.Equal(resident, model.ResidentWeightBytes);
            // Only input IDs and target IDs are uploaded during a resident step.
            Assert.Equal((long)sizeof(int) * ((tokens.Length - 1) + (tokens.Length - 2)), model.UploadedBytes.Sum() - uploaded);
        }
        double finalLoss = model.EvaluateLoraLoss(tokens, 2);
        Assert.True(finalLoss < initialLoss - 1e-5, $"Training did not reduce loss: {initialLoss:R} -> {finalLoss:R}.");
        foreach (string name in new[] { "blk.0.attn_qkv.weight", "blk.1.attn_q.weight", "blk.0.ffn_down.weight" })
        {
            float[][] trained = model.LoraMatrices[name].ReadState();
            Assert.Contains(trained[0].Zip(original[name][0], (a, b) => Math.Abs(a - b)), difference => difference > 1e-7);
            Assert.Contains(trained[1].Zip(original[name][1], (a, b) => Math.Abs(a - b)), difference => difference > 1e-7);
        }
        // Restoring only the adapters recovers base logits, detecting accidental
        // mutation of frozen GPU weights as well as checking the unchanged file.
        foreach (var pair in original) model.LoraMatrices[pair.Key].RestoreState(pair.Value, training: true);
        float[][] restored = Logits(model, tokens[..^1]);
        for (int t = 0; t < baseLogits.Length; ++t) Close(baseLogits[t], restored[t], 1e-6);
        Assert.Equal(fileHash, Hash(fixture.Path));
    }

    [Fact]
    public void CheckpointRestoresOptimizerAndRejectsInvalidInputsBeforeAttaching()
    {
        RequireArc();
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using TemporaryQwenGguf otherBase = Qwen35ResidentModelTests.CreateFixture(tiedOutput: true);
        string path = Path.Combine(Path.GetTempPath(), "nntrain-q35-lora-" + Guid.NewGuid().ToString("N") + ".adapter");
        string corrupt = path + ".corrupt";
        try
        {
            using Qwen35QuantizedModel original = Load(fixture.Path);
            original.AttachLora(new() { Rank = 2, Alpha = 4, LearningRate = .003f, GradientClip = 5 });
            int[] tokens = [1, 2, 3, 0, 2];
            original.TrainLora(tokens, 2);
            original.TrainLora(tokens, 2);
            original.SaveLora(path, "fixture-sft-v1");
            byte[] bytes = File.ReadAllBytes(path);
            bytes[bytes.Length / 2] ^= 0x40;
            File.WriteAllBytes(corrupt, bytes);
            using Qwen35QuantizedModel resumed = Load(fixture.Path);
            Assert.Throws<InvalidDataException>(() => resumed.LoadLora(corrupt, "fixture-sft-v1"));
            Assert.Empty(resumed.LoraTargets);
            Assert.Equal(0, resumed.LoraStep);
            Assert.Throws<InvalidDataException>(() => resumed.LoadLora(path, "different-data"));
            Assert.Empty(resumed.LoraTargets);
            Assert.Equal(0, resumed.LoraStep);
            using (Qwen35QuantizedModel wrongBase = Load(otherBase.Path))
            {
                Assert.Throws<InvalidDataException>(() => wrongBase.LoadLora(path, "fixture-sft-v1"));
                Assert.Empty(wrongBase.LoraTargets);
                Assert.Equal(0, wrongBase.LoraStep);
            }
            resumed.LoadLora(path, "fixture-sft-v1");
            Assert.Equal(original.LoraStep, resumed.LoraStep);
            Assert.Equal(original.LoraTargets, resumed.LoraTargets);
            AssertSameState(original, resumed);
            float[][] expectedLogits = Logits(original, tokens[..^1]), actualLogits = Logits(resumed, tokens[..^1]);
            for (int t = 0; t < expectedLogits.Length; ++t) Close(expectedLogits[t], actualLogits[t], 1e-6);
            Qwen35LoraStepResult expected = original.TrainLora(tokens, 2), actual = resumed.TrainLora(tokens, 2);
            Assert.Equal(expected.Step, actual.Step);
            Close(expected.Loss, actual.Loss, 1e-6);
            Close(expected.GradientNorm, actual.GradientNorm, 1e-6);
            AssertSameState(original, resumed);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(corrupt)) File.Delete(corrupt);
        }
    }

    private static Qwen35QuantizedModel Load(string path)
        => Qwen35QuantizedModel.Load(path, [0], options: new() { LoraTraining = true });

    private static void RequireArc()
        => Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");

    private static float[][] Logits(Qwen35QuantizedModel model, int[] tokens)
    {
        model.Reset();
        return tokens.Select(token => model.ForwardToken(token)).ToArray();
    }

    private static double CrossEntropy(float[] logits, int target)
    {
        double maximum = logits.Max();
        return Math.Log(logits.Sum(logit => Math.Exp(logit - maximum))) + maximum - logits[target];
    }

    private static void AssertSameState(Qwen35QuantizedModel expected, Qwen35QuantizedModel actual)
    {
        foreach (var pair in expected.LoraMatrices)
        {
            float[][] left = pair.Value.ReadState(), right = actual.LoraMatrices[pair.Key].ReadState();
            for (int i = 0; i < left.Length; ++i) Assert.Equal(left[i], right[i]);
        }
    }

    private static byte[] Hash(string path)
    {
        using var file = File.OpenRead(path);
        return SHA256.HashData(file);
    }

    private static void Close(float[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Close(expected[i], actual[i], tolerance);
    }

    private static void Close(double expected, double actual, double tolerance)
        => Assert.True(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}.");
}
