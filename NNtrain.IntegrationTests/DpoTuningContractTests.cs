using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class DpoTuningContractTests
{
    [Fact]
    public void LegacyPrecisionContractDefaultsAreCompatibleAndConversionIsExplicit()
    {
        string Contract(string mode, int? block = null, int rank = 8) => JsonSerializer.Serialize(new {
            batchSize = 2, gradientAccumulationSteps = 8, dpoBeta = .1f,
            lora = new { precisionMode = mode, bfp8BlockSize = block, rank }
        }, new JsonSerializerOptions { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull });
        Assert.True(DpoCommand.CompatibleContract(Contract("mix16_32"), Contract("mix16_32", 32)));
        Assert.True(DpoCommand.CompatibleContract(Contract("mix8_32"), Contract("mix8_32", 32)));
        Assert.False(DpoCommand.CompatibleContract(Contract("mix16_32"), Contract("mix8_32", 32)));
        Assert.True(DpoCommand.CompatibleContract(Contract("mix16_32"), Contract("mix8_32", 32), true));
        Assert.False(DpoCommand.CompatibleContract(Contract("mix16_32"), Contract("mix8_32", 32, 4), true));
        Assert.False(DpoCommand.CompatibleContract(Contract("mix8_32", 32), Contract("mix8_32", 128)));
        Assert.False(DpoCommand.CompatibleContract(Contract("unknown"), Contract("mix8_32", 32), true));
    }

    [Fact]
    public void TuningKeepsObjectiveAndEffectiveBatchButAllowsBufferingAndPartitionChanges()
    {
        string Contract(int batch, int accumulation, int queue, int slots, float beta = .1f) =>
            JsonSerializer.Serialize(new { batchSize = batch, gradientAccumulationSteps = accumulation,
                completedQueueCapacity = queue, generationSlots = slots, dpoBeta = beta });
        string original = Contract(2, 8, 2, 2);
        Assert.True(DpoCommand.CompatibleContract(original, Contract(1, 16, 8, 4)));
        Assert.False(DpoCommand.CompatibleContract(original, Contract(2, 4, 8, 4)));
        Assert.False(DpoCommand.CompatibleContract(original, Contract(2, 8, 8, 4, .2f)));
    }

    [Fact]
    public void EffectiveBatchMismatchReportsValuesAndPreservingAccumulationWithoutPrecisionHint()
    {
        var differences = DpoCommand.ContractMismatches(Contract(2, 8), Contract(4, 8));
        string difference = Assert.Single(differences);
        Assert.Contains("effective batch: saved=16", difference);
        Assert.Contains("current=32", difference);
        Assert.Contains("batchSize=2, gradientAccumulationSteps=8", difference);
        Assert.Contains("with batchSize=4, set gradientAccumulationSteps=4", difference);
        Assert.DoesNotContain("allowPrecisionConversion", difference);
        Assert.False(DpoCommand.CompatibleContract(Contract(2, 8), Contract(4, 8), true));
        Assert.True(DpoCommand.CompatibleContract(Contract(2, 8), Contract(4, 4)));
    }

    [Fact]
    public void EffectiveBatchMismatchDoesNotSuggestFractionalAccumulation()
    {
        string difference = Assert.Single(DpoCommand.ContractMismatches(Contract(2, 8), Contract(3, 8)));
        Assert.DoesNotContain("set gradientAccumulationSteps", difference);
    }

    [Fact]
    public void ObjectiveAndNestedAdapterChangesReportExactFieldValues()
    {
        var differences = DpoCommand.ContractMismatches(Contract(2, 8), Contract(2, 8, beta: .2f, rank: 4));
        Assert.Equal(2, differences.Count);
        Assert.Contains("dpoBeta: saved=0.1, current=0.2", differences);
        Assert.Contains("lora.rank: saved=8, current=4", differences);
        Assert.All(differences, difference => Assert.DoesNotContain("allowPrecisionConversion", difference));
    }

    [Theory]
    [InlineData("mix16_32", 32, "mix8_32", 32, "lora.precisionMode")]
    [InlineData("mix8_32", 32, "mix8_32", 128, "lora.bfp8BlockSize")]
    public void PrecisionHintAppearsOnlyForBlockedPrecisionConversion(string savedMode, int savedBlock,
        string currentMode, int currentBlock, string field)
    {
        string saved = Contract(2, 8, mode: savedMode, block: savedBlock);
        string current = Contract(2, 8, mode: currentMode, block: currentBlock);
        var differences = DpoCommand.ContractMismatches(saved, current);
        Assert.Contains(differences, difference => difference.StartsWith(field + ": saved=", StringComparison.Ordinal) &&
            difference.Contains("allowPrecisionConversionOnResume=true", StringComparison.Ordinal));
        Assert.Empty(DpoCommand.ContractMismatches(saved, current, true));
    }

    [Fact]
    public void UnsupportedPrecisionRemainsRejectedEvenIfConversionIsAllowed()
    {
        var differences = DpoCommand.ContractMismatches(Contract(2, 8, mode: "unknown"), Contract(2, 8), true);
        Assert.Contains("unsupported lora.precisionMode: saved=\"unknown\", current=\"mix8_32\"", differences);
        Assert.All(differences, difference => Assert.DoesNotContain("requires allowPrecisionConversion", difference));
    }

    private static string Contract(int batch, int accumulation, float beta = .1f, int rank = 8,
        string mode = "mix8_32", int block = 32) => JsonSerializer.Serialize(new {
            batchSize = batch, gradientAccumulationSteps = accumulation, dpoBeta = beta,
            lora = new { precisionMode = mode, bfp8BlockSize = block, rank }
        });
}
