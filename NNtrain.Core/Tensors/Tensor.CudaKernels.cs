using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

/// <summary>CUDA kernels shared by the ForgetMemory training graph.</summary>
internal static class TensorCudaKernels
{
    internal static float ClipGradientNormResident(
        IReadOnlyList<Parameter> parameters,
        float maxNorm)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var squaredSumBuffer = accelerator.Allocate1D<double>(1);
        squaredSumBuffer.MemSetToZero();
        var normKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<double>>(
                GradientSquaredSumKernel);
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        foreach (Parameter parameter in parameters)
        {
            Tensor tensor = parameter.T;
            if (!tensor.HasGradientBuffer)
                continue;
            var gradient = tensor.EnsureCudaGradientBuffer();
            normKernel(tensor.Numel, gradient.View, squaredSumBuffer.View);
        }
        accelerator.Synchronize();
        var squaredSum = new double[1];
        squaredSumBuffer.CopyToCPU(squaredSum);
        float totalNorm = (float)Math.Sqrt(squaredSum[0]);
        if (totalNorm <= maxNorm)
            return totalNorm;

        float scale = maxNorm / (totalNorm + 1e-6f);
        foreach (Parameter parameter in parameters)
        {
            Tensor tensor = parameter.T;
            if (!tensor.HasGradientBuffer)
                continue;
            foreach (int deviceIndex in devices)
            {
                CudaAccelerator device =
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                var deviceScaleKernel = device.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, float>(ScaleGradientKernel);
                var gradient = tensor.EnsureCudaGradientBuffer(deviceIndex);
                deviceScaleKernel(tensor.Numel, gradient.View, scale);
                device.Synchronize();
            }
            tensor.MarkCudaGradientsSynchronized(devices);
        }
        return totalNorm;
    }

    internal static void AllReduceGradientResident(
        Tensor tensor,
        IReadOnlyList<int> deviceIndices)
    {
        if (deviceIndices.Count < 2)
            return;
        int primaryIndex = deviceIndices[0];
        CudaAccelerator primary =
            ForgetMemoryV2Cuda.GetAccelerator(primaryIndex);
        MemoryBuffer1D<float, Stride1D.Dense> primaryGradient =
            tensor.EnsureCudaGradientBuffer(primaryIndex);
        using var staging = primary.Allocate1D<float>(tensor.Numel);
        var addKernel = primary.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(AccumulateKernel);
        for (int index = 1; index < deviceIndices.Count; index++)
        {
            int secondaryIndex = deviceIndices[index];
            CudaAccelerator secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            MemoryBuffer1D<float, Stride1D.Dense> secondaryGradient =
                tensor.EnsureCudaGradientBuffer(secondaryIndex);
            secondary.Synchronize();
            secondaryGradient.View.CopyTo(staging.View);
            primary.Synchronize();
            addKernel(tensor.Numel, staging.View, primaryGradient.View);
        }
        primary.Synchronize();
        for (int index = 1; index < deviceIndices.Count; index++)
        {
            int secondaryIndex = deviceIndices[index];
            CudaAccelerator secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            MemoryBuffer1D<float, Stride1D.Dense> secondaryGradient =
                tensor.EnsureCudaGradientBuffer(secondaryIndex);
            primaryGradient.View.CopyTo(secondaryGradient.View);
            secondary.Synchronize();
        }
        tensor.MarkCudaGradientsSynchronized(deviceIndices);
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense> CopyForwardResident(
        Tensor input)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var outputBuffer = accelerator.Allocate1D<float>(input.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(CopyKernel);
        kernel(input.Numel, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void AccumulateGradientResident(
        Tensor source,
        Tensor destination)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var sourceBuffer = source.EnsureCudaGradientBuffer();
        var destinationBuffer = destination.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(AccumulateKernel);
        kernel(source.Numel, sourceBuffer.View, destinationBuffer.View);
        accelerator.Synchronize();
        destination.MarkCudaGradientMutated();
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense> AddForwardResident(
        Tensor left,
        Tensor right,
        bool bfloat16Compute)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var leftBuffer = left.EnsureCudaFloat32Buffer();
        var rightBuffer = right.EnsureCudaFloat32Buffer();
        var outputBuffer = accelerator.Allocate1D<float>(left.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                AddForwardKernel);
        kernel(
            left.Numel,
            leftBuffer.View,
            rightBuffer.View,
            outputBuffer.View,
            bfloat16Compute ? 1 : 0);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void AddBackwardResident(
        Tensor output,
        Tensor left,
        Tensor right)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradient = output.EnsureCudaGradientBuffer();
        var leftGradient = left.EnsureCudaGradientBuffer();
        var rightGradient = ReferenceEquals(left, right)
            ? leftGradient
            : right.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                AddBackwardKernel);
        kernel(
            output.Numel,
            outputGradient.View,
            leftGradient.View,
            rightGradient.View,
            ReferenceEquals(left, right) ? 1 : 0);
        accelerator.Synchronize();
        left.MarkCudaGradientMutated();
        if (!ReferenceEquals(left, right))
            right.MarkCudaGradientMutated();
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense>
        EmbeddingForwardResident(
            Tensor table,
            int[] indices,
            int width)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var tableBuffer = table.EnsureCudaFloat32Buffer();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        var outputBuffer = accelerator.Allocate1D<float>(
            checked(indices.Length * width));
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(
                EmbeddingForwardKernel);
        kernel(
            checked((int)outputBuffer.Length),
            tableBuffer.View,
            indicesBuffer.View,
            outputBuffer.View,
            width);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void EmbeddingBackwardResident(
        Tensor output,
        Tensor table,
        int[] indices,
        int width)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var tableGradientBuffer = table.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, int>(
                EmbeddingBackwardKernel);
        kernel(
            output.Numel,
            indicesBuffer.View,
            outputGradientBuffer.View,
            tableGradientBuffer.View,
            width);
        accelerator.Synchronize();
        table.MarkCudaGradientMutated();
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense>
        DropoutForwardResident(
            Tensor input,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var outputBuffer = accelerator.Allocate1D<float>(input.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, uint, uint, float>(
                DropoutForwardKernel);
        kernel(
            input.Numel,
            inputBuffer.View,
            outputBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void DropoutBackwardResident(
        Tensor output,
        Tensor input,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var inputGradientBuffer = input.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, uint, uint, float>(
                DropoutBackwardKernel);
        kernel(
            output.Numel,
            outputGradientBuffer.View,
            inputGradientBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        input.MarkCudaGradientMutated();
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense>
        AddDropoutForwardResident(
            Tensor residual,
            Tensor branch,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var residualBuffer = residual.EnsureCudaFloat32Buffer();
        var branchBuffer = branch.EnsureCudaFloat32Buffer();
        var outputBuffer = accelerator.Allocate1D<float>(residual.Numel);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            uint, uint, float>(AddDropoutForwardKernel);
        kernel(
            residual.Numel,
            residualBuffer.View,
            branchBuffer.View,
            outputBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void AddDropoutBackwardResident(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var residualGradientBuffer = residual.EnsureCudaGradientBuffer();
        MemoryBuffer1D<float, Stride1D.Dense> branchGradientBuffer = sameParent
            ? residualGradientBuffer
            : branch.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int,
            uint, uint, float>(AddDropoutBackwardKernel);
        kernel(
            output.Numel,
            outputGradientBuffer.View,
            residualGradientBuffer.View,
            branchGradientBuffer.View,
            sameParent ? 1 : 0,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        residual.MarkCudaGradientMutated();
        if (!sameParent)
            branch.MarkCudaGradientMutated();
    }

    internal static float[] EmbeddingForward(
        Tensor table,
        int[] indices,
        int width)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[checked(indices.Length * width)];
        var tableBuffer = table.EnsureCudaFloat32Buffer();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(
                EmbeddingForwardKernel);
        kernel(
            output.Length,
            tableBuffer.View,
            indicesBuffer.View,
            outputBuffer.View,
            width);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void EmbeddingBackward(
        int[] indices,
        float[] outputGradient,
        float[] tableGradient,
        int width)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var tableGradientBuffer = accelerator.Allocate1D(tableGradient);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, int>(
                EmbeddingBackwardKernel);
        kernel(
            outputGradient.Length,
            indicesBuffer.View,
            outputGradientBuffer.View,
            tableGradientBuffer.View,
            width);
        accelerator.Synchronize();
        tableGradientBuffer.CopyToCPU(tableGradient);
    }

    internal static float[] DropoutForward(
        float[] input,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[input.Length];
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, uint, uint, float>(
                DropoutForwardKernel);
        kernel(
            output.Length,
            inputBuffer.View,
            outputBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void DropoutBackward(
        float[] outputGradient,
        float[] inputGradient,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, uint, uint, float>(
                DropoutBackwardKernel);
        kernel(
            outputGradient.Length,
            outputGradientBuffer.View,
            inputGradientBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
    }

    internal static float[] AddDropoutForward(
        float[] residual,
        float[] branch,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[residual.Length];
        var residualBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, residual);
        var branchBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, branch);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            uint, uint, float>(AddDropoutForwardKernel);
        kernel(
            output.Length,
            residualBuffer.View,
            branchBuffer.View,
            outputBuffer.View,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void AddDropoutBackward(
        float[] outputGradient,
        float[] residualGradient,
        float[] branchGradient,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var residualGradientBuffer = accelerator.Allocate1D(residualGradient);
        using var branchGradientBuffer = sameParent
            ? null
            : accelerator.Allocate1D(branchGradient);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int,
            uint, uint, float>(AddDropoutBackwardKernel);
        kernel(
            outputGradient.Length,
            outputGradientBuffer.View,
            residualGradientBuffer.View,
            sameParent
                ? residualGradientBuffer.View
                : branchGradientBuffer!.View,
            sameParent ? 1 : 0,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        residualGradientBuffer.CopyToCPU(residualGradient);
        if (!sameParent)
            branchGradientBuffer!.CopyToCPU(branchGradient);
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense>
        LinearForwardResident(
            Tensor input,
            Tensor weight,
            Tensor bias,
            int rows,
            int inputWidth,
            int outputWidth,
            bool applyRelu,
            bool bfloat16Compute)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaFloat32Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaFloat32Buffer(deviceIndex);
        var outputBuffer = accelerator.Allocate1D<float>(
            checked(rows * outputWidth));
        CudaBlas.LinearForward(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            outputBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(
                LinearBiasActivationKernel);
        kernel(
            checked(rows * outputWidth),
            biasBuffer.View,
            outputBuffer.View,
            outputWidth,
            applyRelu ? 1 : 0,
            bfloat16Compute ? 1 : 0);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void LinearBackwardResident(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Tensor output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaFloat32Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var outputBuffer = output.EnsureCudaFloat32Buffer(deviceIndex);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradientBuffer = input.EnsureCudaGradientBuffer(deviceIndex);
        var weightGradientBuffer = weight.EnsureCudaGradientBuffer(deviceIndex);
        var biasGradientBuffer = bias.EnsureCudaGradientBuffer(deviceIndex);
        var maskKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(
                LinearApplyActivationGradientKernel);
        maskKernel(
            output.Numel,
            outputBuffer.View,
            outputGradientBuffer.View,
            applyRelu ? 1 : 0);
        CudaBlas.LinearBackwardInput(
            accelerator,
            deviceIndex,
            outputGradientBuffer,
            weightBuffer,
            inputGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaBlas.LinearBackwardWeight(
            accelerator,
            deviceIndex,
            inputBuffer,
            outputGradientBuffer,
            weightGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        var biasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                LinearBackwardBiasKernel);
        biasKernel(
            outputWidth,
            outputGradientBuffer.View,
            biasGradientBuffer.View,
            rows,
            outputWidth);
        accelerator.Synchronize();
        input.MarkCudaGradientMutated(deviceIndex);
        weight.MarkCudaGradientMutated(deviceIndex);
        bias.MarkCudaGradientMutated(deviceIndex);
    }

    internal static float[] LinearForward(
        float[] input,
        Tensor weight,
        Tensor bias,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute = false)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(rows, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            return LinearForwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]), devices[0],
                input, weight, bias, rows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute,
                cacheInput: true);
        }

        var output = new float[checked(rows * outputWidth)];
        Parallel.For(0, devices.Length, shard =>
        {
            int start = rows * shard / devices.Length;
            int end = rows * (shard + 1) / devices.Length;
            float[] shardOutput = LinearForwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]),
                devices[shard],
                input.AsSpan(start * inputWidth, (end - start) * inputWidth)
                    .ToArray(),
                weight, bias, end - start, inputWidth, outputWidth, applyRelu,
                bfloat16Compute,
                cacheInput: false);
            shardOutput.CopyTo(output, start * outputWidth);
        });
        return output;
    }

    private static float[] LinearForwardSingle(
        CudaAccelerator accelerator,
        int deviceIndex,
        float[] input,
        Tensor weight,
        Tensor bias,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute,
        bool cacheInput)
    {
        var output = new float[checked(rows * outputWidth)];
        using MemoryBuffer1D<float, Stride1D.Dense>? temporaryInputBuffer =
            cacheInput ? null : accelerator.Allocate1D(input);
        MemoryBuffer1D<float, Stride1D.Dense> inputBuffer = cacheInput
            ? CudaResidentArrayCache.GetOrUpload(accelerator, input)
            : temporaryInputBuffer!;
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaFloat32Buffer(deviceIndex);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        CudaBlas.LinearForward(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            outputBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(
                LinearBiasActivationKernel);
        kernel(
            output.Length,
            biasBuffer.View,
            outputBuffer.View,
            outputWidth,
            applyRelu ? 1 : 0,
            bfloat16Compute ? 1 : 0);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void LinearBackward(
        float[] input,
        float[] weight,
        float[] storedOutput,
        float[] outputGradient,
        float[] inputGradient,
        float[] weightGradient,
        float[] biasGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(rows, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1
            || ReferenceEquals(inputGradient, weightGradient))
        {
            LinearBackwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]), devices[0], input, weight,
                storedOutput, outputGradient, inputGradient, weightGradient,
                biasGradient, rows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute);
            return;
        }

        var shardInputGradients = new float[devices.Length][];
        var shardWeightGradients = new float[devices.Length][];
        var shardBiasGradients = new float[devices.Length][];
        Parallel.For(0, devices.Length, shard =>
        {
            int start = rows * shard / devices.Length;
            int end = rows * (shard + 1) / devices.Length;
            int shardRows = end - start;
            var localInputGradient = new float[shardRows * inputWidth];
            var localWeightGradient = new float[weightGradient.Length];
            var localBiasGradient = new float[biasGradient.Length];
            LinearBackwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]),
                devices[shard],
                input.AsSpan(start * inputWidth, shardRows * inputWidth).ToArray(),
                weight,
                storedOutput.AsSpan(start * outputWidth, shardRows * outputWidth).ToArray(),
                outputGradient.AsSpan(start * outputWidth, shardRows * outputWidth).ToArray(),
                localInputGradient, localWeightGradient, localBiasGradient,
                shardRows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute);
            shardInputGradients[shard] = localInputGradient;
            shardWeightGradients[shard] = localWeightGradient;
            shardBiasGradients[shard] = localBiasGradient;
        });
        for (int shard = 0; shard < devices.Length; shard++)
        {
            int start = rows * shard / devices.Length;
            float[] localInput = shardInputGradients[shard];
            for (int index = 0; index < localInput.Length; index++)
                inputGradient[start * inputWidth + index] += localInput[index];
            float[] localWeight = shardWeightGradients[shard];
            for (int index = 0; index < localWeight.Length; index++)
                weightGradient[index] += localWeight[index];
            float[] localBias = shardBiasGradients[shard];
            for (int index = 0; index < localBias.Length; index++)
                biasGradient[index] += localBias[index];
        }
    }

    private static void LinearBackwardSingle(
        CudaAccelerator accelerator,
        int deviceIndex,
        float[] input,
        float[] weight,
        float[] storedOutput,
        float[] outputGradient,
        float[] inputGradient,
        float[] weightGradient,
        float[] biasGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        var weightBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, weight);
        var outputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, storedOutput);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        using var weightGradientBuffer = accelerator.Allocate1D(weightGradient);
        using var biasGradientBuffer = accelerator.Allocate1D(biasGradient);
        var maskKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(
                LinearApplyActivationGradientKernel);
        maskKernel(
            outputGradient.Length,
            outputBuffer.View,
            outputGradientBuffer.View,
            applyRelu ? 1 : 0);
        CudaBlas.LinearBackwardInput(
            accelerator,
            deviceIndex,
            outputGradientBuffer,
            weightBuffer,
            inputGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaBlas.LinearBackwardWeight(
            accelerator,
            deviceIndex,
            inputBuffer,
            outputGradientBuffer,
            weightGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        var biasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(
                LinearBackwardBiasKernel);
        biasKernel(
            outputWidth,
            outputGradientBuffer.View,
            biasGradientBuffer.View,
            rows,
            outputWidth);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
        weightGradientBuffer.CopyToCPU(weightGradient);
        biasGradientBuffer.CopyToCPU(biasGradient);
    }

    internal static LayerNormResidentContext LayerNormForwardResident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        float epsilon)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var betaBuffer = beta.EnsureCudaFloat32Buffer();
        var outputBuffer = accelerator.Allocate1D<float>(input.Numel);
        var normalizedBuffer = accelerator.Allocate1D<float>(input.Numel);
        var inverseBuffer = accelerator.Allocate1D<float>(rows);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(
                LayerNormForwardKernel);
        kernel(
            rows,
            inputBuffer.View,
            gammaBuffer.View,
            betaBuffer.View,
            outputBuffer.View,
            normalizedBuffer.View,
            inverseBuffer.View,
            columns,
            epsilon);
        accelerator.Synchronize();
        return new LayerNormResidentContext(
            outputBuffer,
            normalizedBuffer,
            inverseBuffer);
    }

    internal static void LayerNormBackwardResident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        LayerNormResidentContext context,
        int rows,
        int columns)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var inputGradientBuffer = input.EnsureCudaGradientBuffer();
        var gammaGradientBuffer = gamma.EnsureCudaGradientBuffer();
        var betaGradientBuffer = beta.EnsureCudaGradientBuffer();
        var inputKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int>(LayerNormBackwardInputKernel);
        inputKernel(
            rows,
            gammaBuffer.View,
            context.Normalized.View,
            context.Inverses.View,
            outputGradientBuffer.View,
            inputGradientBuffer.View,
            columns);
        var parameterKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int>(LayerNormBackwardParameterKernel);
        parameterKernel(
            columns,
            context.Normalized.View,
            outputGradientBuffer.View,
            gammaGradientBuffer.View,
            betaGradientBuffer.View,
            rows,
            columns);
        accelerator.Synchronize();
        input.MarkCudaGradientMutated();
        gamma.MarkCudaGradientMutated();
        beta.MarkCudaGradientMutated();
    }

    internal sealed class LayerNormResidentContext(
        MemoryBuffer1D<float, Stride1D.Dense> output,
        MemoryBuffer1D<float, Stride1D.Dense> normalized,
        MemoryBuffer1D<float, Stride1D.Dense> inverses)
    {
        internal MemoryBuffer1D<float, Stride1D.Dense> Output { get; } = output;
        internal MemoryBuffer1D<float, Stride1D.Dense> Normalized { get; } = normalized;
        internal MemoryBuffer1D<float, Stride1D.Dense> Inverses { get; } = inverses;

        ~LayerNormResidentContext()
        {
            Normalized.Dispose();
            Inverses.Dispose();
        }
    }

    internal static (
        float[] Output,
        float[] Normalized,
        float[] InverseStandardDeviations) LayerNormForward(
        float[] input,
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        float epsilon)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[input.Length];
        var normalized = new float[input.Length];
        var inverses = new float[rows];
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var betaBuffer = beta.EnsureCudaFloat32Buffer();
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        using var normalizedBuffer =
            accelerator.Allocate1D<float>(normalized.Length);
        using var inverseBuffer = accelerator.Allocate1D<float>(rows);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(
                LayerNormForwardKernel);
        kernel(
            rows,
            inputBuffer.View,
            gammaBuffer.View,
            betaBuffer.View,
            outputBuffer.View,
            normalizedBuffer.View,
            inverseBuffer.View,
            columns,
            epsilon);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        normalizedBuffer.CopyToCPU(normalized);
        inverseBuffer.CopyToCPU(inverses);
        return (output, normalized, inverses);
    }

    internal static void LayerNormBackward(
        Tensor gamma,
        float[] normalized,
        float[] inverses,
        float[] outputGradient,
        float[] inputGradient,
        float[] gammaGradient,
        float[] betaGradient,
        int rows,
        int columns)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var normalizedBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, normalized);
        var inverseBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, inverses);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        using var gammaGradientBuffer = accelerator.Allocate1D(gammaGradient);
        using var betaGradientBuffer = accelerator.Allocate1D(betaGradient);
        var inputKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int>(LayerNormBackwardInputKernel);
        inputKernel(
            rows,
            gammaBuffer.View,
            normalizedBuffer.View,
            inverseBuffer.View,
            outputGradientBuffer.View,
            inputGradientBuffer.View,
            columns);
        var parameterKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, int, int>(LayerNormBackwardParameterKernel);
        parameterKernel(
            columns,
            normalizedBuffer.View,
            outputGradientBuffer.View,
            gammaGradientBuffer.View,
            betaGradientBuffer.View,
            rows,
            columns);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
        gammaGradientBuffer.CopyToCPU(gammaGradient);
        betaGradientBuffer.CopyToCPU(betaGradient);
    }

    internal static CrossEntropyResidentContext CrossEntropyForwardResident(
        Tensor logits,
        int[] labels,
        int rows,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var logitsBuffer = logits.EnsureCudaFloat32Buffer();
        var labelsBuffer = accelerator.Allocate1D(labels);
        var probabilitiesBuffer = accelerator.Allocate1D<float>(logits.Numel);
        var lossBuffer = accelerator.Allocate1D<float>(1);
        lossBuffer.MemSetToZero();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, int, int, int, float>(CrossEntropyForwardKernel);
        kernel(
            rows,
            logitsBuffer.View,
            labelsBuffer.View,
            probabilitiesBuffer.View,
            lossBuffer.View,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing);
        accelerator.Synchronize();
        return new CrossEntropyResidentContext(
            lossBuffer,
            probabilitiesBuffer,
            labelsBuffer);
    }

    internal static void CrossEntropyBackwardResident(
        Tensor logits,
        Tensor loss,
        CrossEntropyResidentContext context,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var lossGradientBuffer = loss.EnsureCudaGradientBuffer();
        var logitsGradientBuffer = logits.EnsureCudaGradientBuffer();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, int, int, int, float>(
                CrossEntropyBackwardResidentKernel);
        kernel(
            logits.Numel,
            context.Probabilities.View,
            context.Labels.View,
            logitsGradientBuffer.View,
            lossGradientBuffer.View,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing);
        accelerator.Synchronize();
        logits.MarkCudaGradientMutated();
    }

    internal sealed class CrossEntropyResidentContext(
        MemoryBuffer1D<float, Stride1D.Dense> loss,
        MemoryBuffer1D<float, Stride1D.Dense> probabilities,
        MemoryBuffer1D<int, Stride1D.Dense> labels)
    {
        internal MemoryBuffer1D<float, Stride1D.Dense> Loss { get; } = loss;
        internal MemoryBuffer1D<float, Stride1D.Dense> Probabilities { get; } = probabilities;
        internal MemoryBuffer1D<int, Stride1D.Dense> Labels { get; } = labels;

        ~CrossEntropyResidentContext()
        {
            Probabilities.Dispose();
            Labels.Dispose();
        }
    }

    internal static (float Loss, float[] Probabilities) CrossEntropyForward(
        float[] logits,
        int[] labels,
        int rows,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var probabilities = new float[logits.Length];
        var loss = new float[1];
        var logitsBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, logits);
        using var labelsBuffer = accelerator.Allocate1D(labels);
        using var probabilitiesBuffer =
            accelerator.Allocate1D<float>(probabilities.Length);
        using var lossBuffer = accelerator.Allocate1D<float>(1);
        lossBuffer.MemSetToZero();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, int, int, int, float>(CrossEntropyForwardKernel);
        kernel(
            rows,
            logitsBuffer.View,
            labelsBuffer.View,
            probabilitiesBuffer.View,
            lossBuffer.View,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing);
        accelerator.Synchronize();
        probabilitiesBuffer.CopyToCPU(probabilities);
        lossBuffer.CopyToCPU(loss);
        return (loss[0], probabilities);
    }

    internal static void CrossEntropyBackward(
        float[] probabilities,
        int[] labels,
        float[] logitsGradient,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing,
        float upstreamGradient)
    {
        CudaAccelerator accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var probabilitiesBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, probabilities);
        using var labelsBuffer = accelerator.Allocate1D(labels);
        using var gradientBuffer = accelerator.Allocate1D(logitsGradient);
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int,
            int, int, float, float>(CrossEntropyBackwardKernel);
        kernel(
            probabilities.Length,
            probabilitiesBuffer.View,
            labelsBuffer.View,
            gradientBuffer.View,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            upstreamGradient);
        accelerator.Synchronize();
        gradientBuffer.CopyToCPU(logitsGradient);
    }

    private static void GradientSquaredSumKernel(
        Index1D index,
        ArrayView<float> gradient,
        ArrayView<double> squaredSum)
    {
        double value = gradient[index];
        Atomic.Add(ref squaredSum[0], value * value);
    }

    private static void ScaleGradientKernel(
        Index1D index,
        ArrayView<float> gradient,
        float scale)
        => gradient[index] *= scale;

    private static void CopyKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output)
        => output[index] = input[index];

    private static void AccumulateKernel(
        Index1D index,
        ArrayView<float> source,
        ArrayView<float> destination)
        => destination[index] += source[index];

    private static void AddForwardKernel(
        Index1D index,
        ArrayView<float> left,
        ArrayView<float> right,
        ArrayView<float> output,
        int bfloat16Compute)
    {
        float value = left[index] + right[index];
        output[index] = bfloat16Compute != 0
            ? RoundBFloat16(value)
            : value;
    }

    private static void AddBackwardKernel(
        Index1D index,
        ArrayView<float> outputGradient,
        ArrayView<float> leftGradient,
        ArrayView<float> rightGradient,
        int sameParent)
    {
        float value = outputGradient[index];
        if (sameParent != 0)
            leftGradient[index] += 2f * value;
        else
        {
            leftGradient[index] += value;
            rightGradient[index] += value;
        }
    }

    private static void LinearBiasActivationKernel(
        Index1D index,
        ArrayView<float> bias,
        ArrayView<float> output,
        int outputWidth,
        int applyRelu,
        int bfloat16Compute)
    {
        int linear = index;
        int column = linear % outputWidth;
        float sum = output[linear] + bias[column];
        float result = applyRelu != 0 && sum <= 0f ? 0f : sum;
        output[linear] = bfloat16Compute != 0
            ? RoundBFloat16(result)
            : result;
    }

    private static float RoundBFloat16(float value)
    {
        uint bits = Interop.FloatAsInt(value);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        return Interop.IntAsFloat((bits + roundingBias) & 0xFFFF0000u);
    }

    private static void EmbeddingForwardKernel(
        Index1D index,
        ArrayView<float> table,
        ArrayView<int> indices,
        ArrayView<float> output,
        int width)
    {
        int linear = index;
        int position = linear / width;
        int column = linear - position * width;
        output[linear] = table[indices[position] * width + column];
    }

    private static void EmbeddingBackwardKernel(
        Index1D index,
        ArrayView<int> indices,
        ArrayView<float> outputGradient,
        ArrayView<float> tableGradient,
        int width)
    {
        int linear = index;
        int position = linear / width;
        int column = linear - position * width;
        Atomic.Add(
            ref tableGradient[indices[position] * width + column],
            outputGradient[linear]);
    }

    private static void DropoutForwardKernel(
        Index1D index,
        ArrayView<float> input,
        ArrayView<float> output,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        output[i] = input[i] * DropoutMultiplier(
            seed,
            i,
            dropThreshold,
            scale);
    }

    private static void DropoutBackwardKernel(
        Index1D index,
        ArrayView<float> outputGradient,
        ArrayView<float> inputGradient,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        inputGradient[i] += outputGradient[i] * DropoutMultiplier(
            seed,
            i,
            dropThreshold,
            scale);
    }

    private static void AddDropoutForwardKernel(
        Index1D index,
        ArrayView<float> residual,
        ArrayView<float> branch,
        ArrayView<float> output,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        output[i] = residual[i] + branch[i] * DropoutMultiplier(
            seed,
            i,
            dropThreshold,
            scale);
    }

    private static void AddDropoutBackwardKernel(
        Index1D index,
        ArrayView<float> outputGradient,
        ArrayView<float> residualGradient,
        ArrayView<float> branchGradient,
        int sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int i = index;
        float gradient = outputGradient[i];
        float multiplier = DropoutMultiplier(seed, i, dropThreshold, scale);
        if (sameParent != 0)
            residualGradient[i] += gradient * (1f + multiplier);
        else
        {
            residualGradient[i] += gradient;
            branchGradient[i] += gradient * multiplier;
        }
    }

    private static float DropoutMultiplier(
        uint seed,
        int index,
        uint dropThreshold,
        float scale)
    {
        uint counter = unchecked((uint)(index + 1));
        uint bits = unchecked(seed + 0x9E3779B9u * counter);
        bits ^= bits >> 16;
        bits *= 0x7FEB352Du;
        bits ^= bits >> 15;
        bits *= 0x846CA68Bu;
        bits ^= bits >> 16;
        return bits < dropThreshold ? 0f : scale;
    }

    private static void LinearApplyActivationGradientKernel(
        Index1D index,
        ArrayView<float> output,
        ArrayView<float> outputGradient,
        int applyRelu)
    {
        int i = index;
        if (applyRelu != 0 && output[i] <= 0f)
            outputGradient[i] = 0f;
    }

    private static void LinearBackwardBiasKernel(
        Index1D columnIndex,
        ArrayView<float> outputGradient,
        ArrayView<float> biasGradient,
        int rows,
        int outputWidth)
    {
        int column = columnIndex;
        float biasSum = 0f;
        for (int row = 0; row < rows; row++)
            biasSum += outputGradient[row * outputWidth + column];
        biasGradient[column] += biasSum;
    }

    private static void LayerNormForwardKernel(
        Index1D rowIndex,
        ArrayView<float> input,
        ArrayView<float> gamma,
        ArrayView<float> beta,
        ArrayView<float> output,
        ArrayView<float> normalized,
        ArrayView<float> inverses,
        int columns,
        float epsilon)
    {
        int row = rowIndex;
        int offset = row * columns;
        float mean = 0f;
        for (int column = 0; column < columns; column++)
            mean += input[offset + column];
        mean /= columns;
        float variance = 0f;
        for (int column = 0; column < columns; column++)
        {
            float difference = input[offset + column] - mean;
            variance += difference * difference;
        }
        float inverse = 1f / XMath.Sqrt(variance / columns + epsilon);
        inverses[row] = inverse;
        for (int column = 0; column < columns; column++)
        {
            float value = (input[offset + column] - mean) * inverse;
            normalized[offset + column] = value;
            output[offset + column] = value * gamma[column] + beta[column];
        }
    }

    private static void LayerNormBackwardInputKernel(
        Index1D rowIndex,
        ArrayView<float> gamma,
        ArrayView<float> normalized,
        ArrayView<float> inverses,
        ArrayView<float> outputGradient,
        ArrayView<float> inputGradient,
        int columns)
    {
        int row = rowIndex;
        int offset = row * columns;
        float sum = 0f;
        float normalizedSum = 0f;
        for (int column = 0; column < columns; column++)
        {
            float dxhat = outputGradient[offset + column] * gamma[column];
            sum += dxhat;
            normalizedSum += dxhat * normalized[offset + column];
        }
        float scale = inverses[row] / columns;
        for (int column = 0; column < columns; column++)
        {
            float dxhat = outputGradient[offset + column] * gamma[column];
            inputGradient[offset + column] += scale *
                (columns * dxhat - sum -
                    normalized[offset + column] * normalizedSum);
        }
    }

    private static void LayerNormBackwardParameterKernel(
        Index1D columnIndex,
        ArrayView<float> normalized,
        ArrayView<float> outputGradient,
        ArrayView<float> gammaGradient,
        ArrayView<float> betaGradient,
        int rows,
        int columns)
    {
        int column = columnIndex;
        float gammaSum = 0f;
        float betaSum = 0f;
        for (int row = 0; row < rows; row++)
        {
            int index = row * columns + column;
            float gradient = outputGradient[index];
            gammaSum += gradient * normalized[index];
            betaSum += gradient;
        }
        gammaGradient[column] += gammaSum;
        betaGradient[column] += betaSum;
    }

    private static void CrossEntropyForwardKernel(
        Index1D rowIndex,
        ArrayView<float> logits,
        ArrayView<int> labels,
        ArrayView<float> probabilities,
        ArrayView<float> loss,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int row = rowIndex;
        int label = labels[row];
        if (label == ignoreIndex)
            return;
        int offset = row * columns;
        float maximum = logits[offset];
        float logitSum = 0f;
        for (int column = 0; column < columns; column++)
        {
            float value = logits[offset + column];
            maximum = XMath.Max(maximum, value);
            logitSum += value;
        }
        float exponentialSum = 0f;
        for (int column = 0; column < columns; column++)
        {
            float exponential = XMath.Exp(logits[offset + column] - maximum);
            probabilities[offset + column] = exponential;
            exponentialSum += exponential;
        }
        float inverse = 1f / exponentialSum;
        for (int column = 0; column < columns; column++)
            probabilities[offset + column] *= inverse;
        float normalizer = maximum + XMath.Log(exponentialSum);
        float negativeLogLikelihood = normalizer - logits[offset + label];
        float uniformLoss = normalizer - logitSum / columns;
        float rowLoss = (1f - labelSmoothing) * negativeLogLikelihood +
            labelSmoothing * uniformLoss;
        Atomic.Add(ref loss[0], rowLoss / validRows);
    }

    private static void CrossEntropyBackwardKernel(
        Index1D index,
        ArrayView<float> probabilities,
        ArrayView<int> labels,
        ArrayView<float> gradient,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing,
        float upstreamGradient)
    {
        int linear = index;
        int row = linear / columns;
        int column = linear - row * columns;
        int label = labels[row];
        if (label == ignoreIndex)
            return;
        float scale = upstreamGradient / validRows;
        float target = labelSmoothing / columns;
        if (column == label)
            target += 1f - labelSmoothing;
        gradient[linear] += scale * (probabilities[linear] - target);
    }

    private static void CrossEntropyBackwardResidentKernel(
        Index1D index,
        ArrayView<float> probabilities,
        ArrayView<int> labels,
        ArrayView<float> gradient,
        ArrayView<float> upstreamGradient,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        int linear = index;
        int row = linear / columns;
        int column = linear - row * columns;
        int label = labels[row];
        if (label == ignoreIndex)
            return;
        float scale = upstreamGradient[0] / validRows;
        float target = labelSmoothing / columns;
        if (column == label)
            target += 1f - labelSmoothing;
        gradient[linear] += scale * (probabilities[linear] - target);
    }
}
