namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private static ArcDataParallelEngine? CreateArcDataParallelEngine(
        WikiTrainingConfiguration config,
        LanguageModel primary,
        int vocabularySize,
        TensorPrecisionMode precisionMode,
        TensorDType storageDType,
        int bfp8BlockSize)
    {
        int[] devices = config.DeviceIndices ?? [config.DeviceIndex];
        if (config.GetExecutionDevice() != TensorDevice.Arc || devices.Length != 2)
            return null;
        LanguageModel secondary;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, devices[1])))
        {
            secondary = CreateModel(config, vocabularySize,
                precisionMode, storageDType, bfp8BlockSize);
        }
        return new ArcDataParallelEngine(primary, secondary, devices);
    }
}
