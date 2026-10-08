namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    /// <summary>Whether this lane can execute the tiled FP32 by IQ2_S projection.</summary>
    public bool SupportsIq2TiledProjection => !Options.AsrKernelsOnly && !Options.Qwen35VisionKernelsOnly
        && Options.XmxMatrices && Device.SupportsXmx
        && Device.MinimumSubgroupSize == 16
        && Device.Extensions.Split(' ').Contains("cl_khr_fp16")
        && Device.Extensions.Split(' ').Contains("cl_intel_subgroups_short");

    private static int Iq2PanelColumns(int inputWidth, int outputWidth, int requested)
    {
        // Temporary exact-integer panels, not a persistent dequantized model.
        long bytesPerColumn = 2L * inputWidth + 4L * (inputWidth / 256);
        int limit = Math.Max(16, (int)Math.Min(int.MaxValue, (64L * 1024 * 1024 / bytesPerColumn / 16) * 16));
        return checked((Math.Min(outputWidth, requested > 0 ? requested : limit) + 15) / 16 * 16);
    }

    public long Iq2TiledForwardWorkspaceBytes(int rows, int inputWidth, int outputWidth)
    {
        if (rows <= 0 || inputWidth <= 0 || inputWidth % 256 != 0 || outputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int columns = Iq2PanelColumns(inputWidth, outputWidth, 0);
        int chunkRows = Iq2ForwardChunkRows(rows, inputWidth, 0);
        long staging = chunkRows < rows ? checked(4L * chunkRows * (inputWidth + (long)outputWidth)) : 0;
        return checked(((chunkRows + 7L) / 8 * 8) * inputWidth * 4 + staging
            + (long)columns * inputWidth * 2 + (long)columns * (inputWidth / 256) * 4 + 4L * chunkRows + 4);
    }

    private static int Iq2ForwardChunkRows(int rows, int inputWidth, int requested)
    {
        // K-major A panels become costly to traverse when the packed working
        // set exceeds the pool/cache-friendly range. Keep long reductions in
        // bounded row chunks; staging and copies stay entirely on the device.
        long packedBytes = checked(((rows + 7L) / 8 * 8) * inputWidth * 4);
        return Math.Min(rows, requested > 0 ? requested : packedBytes > 384L * 1024 * 1024 ? 2048 : rows);
    }

    /// <summary>
    /// FP32 activations times original IQ2_S weights, with FP32 output and bias.
    /// Includes transient activation/weight packing on this lane's queue. High/low
    /// halves retain activation residuals; IQ2 integer panels are exact. The device
    /// falls back to decoded FP32 arithmetic for out-of-range activations.
    /// </summary>
    public void Iq2TiledForward(ArcBuffer input, ArcBuffer weight, ArcBuffer bias, ArcBuffer output,
        int rows, int inputWidth, int outputWidth, int tileRows = 16, int tileColumns = 32,
        int panelColumns = 0, bool unrollTwo = true, int rowChunkRows = 0, bool rowMajor = false)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!SupportsIq2TiledProjection) throw new NotSupportedException("IQ2 tiles require SG16, XMX and FP16.");
            if (rows <= 0 || inputWidth <= 0 || inputWidth % 256 != 0 || outputWidth <= 0
                || panelColumns < 0 || (panelColumns != 0 && panelColumns % 16 != 0)
                || rowChunkRows < 0 || (rowChunkRows != 0 && rowChunkRows % 8 != 0))
                throw new ArgumentOutOfRangeException(nameof(rows));
            if ((tileRows, tileColumns) is not ((8, 16) or (16, 16) or (16, 32) or (32, 32) or (128, 64)))
                throw new ArgumentOutOfRangeException(nameof(tileRows));
            if (rowMajor && (tileRows != 16 || tileColumns != 32 || !unrollTwo))
                throw new ArgumentException("The experimental row-major A layout requires a 16x32 tile with unrollTwo enabled.", nameof(rowMajor));
            ValidateIq2Buffer(input, checked((long)rows * inputWidth * 4));
            ValidateIq2Buffer(weight, checked((long)outputWidth * (inputWidth / 256) * 82));
            ValidateIq2Buffer(bias, checked((long)outputWidth * 4));
            ValidateIq2Buffer(output, checked((long)rows * outputWidth * 4));
            if (output.Handle == input.Handle || output.Handle == weight.Handle || output.Handle == bias.Handle)
                throw new ArgumentException("IQ2 projection output must not alias an input.");
            // The alternative layout removes the long K-major row stride,
            // so its benchmark uses the full input unless explicitly chunked.
            int chunkRows = rowMajor && rowChunkRows == 0
                ? rows : Iq2ForwardChunkRows(rows, inputWidth, rowChunkRows);
            if (chunkRows < rows)
            {
                using ArcBuffer inputSlice = Allocate(checked(chunkRows * inputWidth));
                using ArcBuffer outputSlice = Allocate(checked(chunkRows * outputWidth));
                for (int first = 0; first < rows; first += chunkRows)
                {
                    int count = Math.Min(chunkRows, rows - first);
                    CopyBytes(input, inputSlice, checked(first * inputWidth * 4), 0, checked(count * inputWidth * 4));
                    Iq2TiledForward(inputSlice, weight, bias, outputSlice, count, inputWidth, outputWidth,
                        tileRows, tileColumns, panelColumns, unrollTwo, rowChunkRows: 0, rowMajor: rowMajor);
                    CopyBytes(outputSlice, output, 0, checked(first * outputWidth * 4), checked(count * outputWidth * 4));
                }
                return;
            }
            int aElements = checked((rows + 7) / 8 * 8 * inputWidth);
            int columns = Iq2PanelColumns(inputWidth, outputWidth, panelColumns);
            using ArcBuffer high = AllocateBytes(checked(aElements * 2));
            using ArcBuffer low = AllocateBytes(checked(aElements * 2));
            using ArcBuffer status = Allocate(1);
            using ArcBuffer rowScales = Allocate(rows);
            Run("q35a_zero", 1, 0, status, 1);
            Run("q35s_input_scales", (long)rows * 128, 128, input, rowScales, status, rows, inputWidth);
            Run(rowMajor ? "q35s_pack_input_f16x2_rowmajor" : "q35s_pack_input_f16x2",
                aElements / 8, 0, input, high, low, status, rows, inputWidth, rowScales);
            if (tileRows == 128)
            {
                Run2D("q35s_forward_gguf_bslm", ((long)outputWidth + 63) / 64 * 16,
                    ((long)rows + 127) / 128 * 16, 16, 16,
                    high, low, weight, bias, output, rows, inputWidth, outputWidth, input, status, rowScales);
                return;
            }
            using ArcBuffer packed = AllocateBytes(checked(columns * inputWidth * 2));
            using ArcBuffer scales = Allocate(checked(columns * (inputWidth / 256)));
            string kernel = rowMajor ? "q35s_forward_u2_rowmajor_m16n32"
                : "q35s_forward_" + (unrollTwo && tileColumns == 32 ? "u2_" : "") + $"m{tileRows}n{tileColumns}";
            for (int first = 0; first < outputWidth; first += columns)
            {
                int count = Math.Min(columns, outputWidth - first);
                int padded = checked((count + 15) / 16 * 16);
                Run("q35s_pack_iq2_b", (long)(inputWidth / 16) * (padded / 16) * 32, 0,
                    weight, packed, scales, status, outputWidth, inputWidth, first, count);
                Run(kernel, ((long)rows + tileRows - 1) / tileRows
                    * ((count + (long)tileColumns - 1) / tileColumns) * 16, 16,
                    high, low, packed, scales, bias, output, input, weight, status,
                    rows, outputWidth, inputWidth, first, count, rowScales);
            }
        }
    }

    private static int Iq2TransposePanelColumns(int reduction, int destinations, int requested)
    {
        int limit = Math.Max(16, (int)Math.Min(int.MaxValue, (64L * 1024 * 1024 / (4L * reduction) / 16) * 16));
        return checked((Math.Min(destinations, requested > 0 ? requested : limit) + 15) / 16 * 16);
    }

    public long Iq2TiledBackwardWorkspaceBytes(int rows, int inputWidth, int outputWidth)
    {
        if (rows <= 0 || inputWidth <= 0 || inputWidth % 256 != 0 || outputWidth <= 0 || outputWidth % 16 != 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int columns = Iq2TransposePanelColumns(outputWidth, inputWidth, 0);
        return checked(((rows + 7L) / 8 * 8) * outputWidth * 4 + (long)columns * outputWidth * 4
            + 4L * rows + inputWidth / 256L * 4 + 4);
    }

    /// <summary>Accumulates dY times the original IQ2_S weight into dX, using bounded transient weight panels.</summary>
    public void Iq2TiledBackward(ArcBuffer input, ArcBuffer weight, ArcBuffer output,
        int rows, int inputWidth, int outputWidth, int tileRows = 16, int tileColumns = 32,
        int panelColumns = 0, bool addToOutput = true, long workspaceBudgetBytes = 256L * 1024 * 1024)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!SupportsIq2TiledProjection) throw new NotSupportedException("IQ2 tiles require SG16, XMX and FP16.");
            if (rows <= 0 || inputWidth <= 0 || inputWidth % 256 != 0 || outputWidth <= 0 || outputWidth % 16 != 0
                || panelColumns < 0 || (panelColumns != 0 && panelColumns % 16 != 0))
                throw new ArgumentOutOfRangeException(nameof(rows));
            if ((tileRows, tileColumns) is not ((8, 16) or (16, 16) or (16, 32) or (32, 32)))
                throw new ArgumentOutOfRangeException(nameof(tileRows));
            ValidateIq2Buffer(input, checked((long)rows * outputWidth * 4));
            ValidateIq2Buffer(weight, checked((long)outputWidth * (inputWidth / 256) * 82));
            ValidateIq2Buffer(output, checked((long)rows * inputWidth * 4));
            if (output.Handle == input.Handle || output.Handle == weight.Handle)
                throw new ArgumentException("IQ2 gradient output must not alias an input.");
            int columns = Iq2TransposePanelColumns(outputWidth, inputWidth, panelColumns);
            long weightBytes = checked((long)columns * outputWidth * 4);
            long fullWorkspace = checked(((rows + 7L) / 8 * 8) * outputWidth * 4 + weightBytes
                + 4L * rows + inputWidth / 256L * 4 + 4);
            if (workspaceBudgetBytes < 0) throw new ArgumentOutOfRangeException(nameof(workspaceBudgetBytes));
            if (workspaceBudgetBytes > 0 && fullWorkspace > workspaceBudgetBytes)
            {
                // Include both device staging buffers in the same scratch cap.
                long bytesPerRow = checked(8L * outputWidth + 4L * inputWidth + 4);
                long capacity = (workspaceBudgetBytes - weightBytes - inputWidth / 256L * 4 - 4) / bytesPerRow / 8 * 8;
                if (capacity < 8) throw new ArgumentOutOfRangeException(nameof(workspaceBudgetBytes), "IQ2 transpose scratch cap is too small.");
                int chunkRows = checked((int)Math.Min(rows, capacity));
                using ArcBuffer inputSlice = Allocate(checked(chunkRows * outputWidth));
                using ArcBuffer outputSlice = Allocate(checked(chunkRows * inputWidth));
                for (int first = 0; first < rows; first += chunkRows)
                {
                    int count = Math.Min(chunkRows, rows - first);
                    CopyBytes(input, inputSlice, checked(first * outputWidth * 4), 0, checked(count * outputWidth * 4));
                    if (addToOutput)
                        CopyBytes(output, outputSlice, checked(first * inputWidth * 4), 0, checked(count * inputWidth * 4));
                    Iq2TiledBackward(inputSlice, weight, outputSlice, count, inputWidth, outputWidth,
                        tileRows, tileColumns, columns, addToOutput, workspaceBudgetBytes: 0);
                    CopyBytes(outputSlice, output, 0, checked(first * inputWidth * 4), checked(count * inputWidth * 4));
                }
                return;
            }
            int elements = checked((rows + 7) / 8 * 8 * outputWidth);
            using ArcBuffer high = AllocateBytes(checked(elements * 2));
            using ArcBuffer low = AllocateBytes(checked(elements * 2));
            using ArcBuffer status = Allocate(1);
            using ArcBuffer rowScales = Allocate(rows);
            using ArcBuffer weightScales = Allocate(inputWidth / 256);
            using ArcBuffer bHigh = AllocateBytes(checked(columns * outputWidth * 2));
            using ArcBuffer bLow = AllocateBytes(checked(columns * outputWidth * 2));
            Run("q35a_zero", 1, 0, status, 1);
            Run("q35s_input_scales", (long)rows * 128, 128, input, rowScales, status, rows, outputWidth);
            Run("q35s_pack_input_f16x2", elements / 8, 0, input, high, low, status, rows, outputWidth, rowScales);
            Run("q35s_transpose_scales", (long)(inputWidth / 256) * 128, 128,
                weight, weightScales, status, inputWidth, outputWidth);
            for (int first = 0; first < inputWidth; first += columns)
            {
                int count = Math.Min(columns, inputWidth - first);
                int padded = checked((count + 15) / 16 * 16);
                Run("q35s_pack_iq2_bt", (long)(outputWidth / 16) * (padded / 16) * 128, 0,
                    weight, bHigh, bLow, status, inputWidth, outputWidth, first, count, weightScales);
                Run($"q35s_transpose_m{tileRows}n{tileColumns}", ((long)rows + tileRows - 1) / tileRows
                    * ((count + (long)tileColumns - 1) / tileColumns) * 16, 16,
                    high, low, bHigh, bLow, output, input, weight, status,
                    rows, inputWidth, outputWidth, first, count, addToOutput ? 1 : 0, rowScales, weightScales);
            }
        }
    }

    private void ValidateIq2Buffer(ArcBuffer buffer, long bytes)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.Owner != this || !buffer.IsAlive || buffer.ByteLength < bytes)
            throw new ArgumentException("IQ2 buffer is too short, disposed, or belongs to another lane.");
    }
}
