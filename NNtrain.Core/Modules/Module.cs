namespace NNtrain;

public abstract class Module
{
    private readonly List<RegisteredMember> _members = [];
    private readonly HashSet<Parameter> _directParameters =
        new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<Module> _directModules =
        new(ReferenceEqualityComparer.Instance);

    protected Module(TensorDType dtype = TensorDType.Float32)
    {
        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));
        DType = dtype;
        PrecisionMode = dtype.ToPrecisionMode();
    }

    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Gets the physical storage dtype selected for this module's parameters.
    /// Stateless modules propagate this contract to their inputs and children.
    /// </summary>
    public TensorDType DType { get; private set; }

    /// <summary>Gets the model-level numeric contract for this module.</summary>
    public TensorPrecisionMode PrecisionMode { get; private set; }

    /// <summary>
    /// Applies a model-level numeric contract to this module and all of its
    /// registered children. The selected mode must use the module's physical
    /// parameter storage dtype.
    /// </summary>
    public void SetPrecisionMode(TensorPrecisionMode precisionMode)
    {
        TensorDType expectedStorage = precisionMode.ToStorageDType();
        bool legacyMixedStorage = precisionMode == TensorPrecisionMode.Mix16_32
            && DType == TensorDType.Float16;
        if (DType != expectedStorage
            && !legacyMixedStorage
            && _directParameters.Count != 0)
        {
            throw new InvalidOperationException(
                $"Precision mode '{TensorPrecisionModeNames.Format(precisionMode)}' " +
                $"requires storage dtype '{expectedStorage}', but module " +
                $"'{GetType().Name}' uses '{DType}'.");
        }
        if (expectedStorage == TensorDType.Bfp8)
        {
            Bfp8ScaleGranularity expectedGranularity =
                precisionMode == TensorPrecisionMode.Bfp8
                    ? Bfp8ScaleGranularity.Tensor
                    : Bfp8ScaleGranularity.Block;
            Parameter? incompatible = _directParameters.FirstOrDefault(
                parameter => parameter.T.Bfp8Quantization?.Granularity
                    != expectedGranularity);
            if (incompatible is not null)
            {
                throw new InvalidOperationException(
                    $"Precision mode " +
                    $"'{TensorPrecisionModeNames.Format(precisionMode)}' " +
                    $"requires {expectedGranularity.ToString().ToLowerInvariant()} " +
                    $"BFP8 scaling, but parameter '{incompatible.Name}' in " +
                    $"module '{GetType().Name}' uses " +
                    $"'{FormatBfp8Descriptor(incompatible.T.Bfp8Quantization)}'. " +
                    "Use model.to(...) to convert the scaling contract.");
            }
        }
        PrecisionMode = precisionMode;
        foreach (Module module in _directModules)
            module.SetPrecisionMode(precisionMode);
    }

    private static string FormatBfp8Descriptor(
        Bfp8QuantizationDescriptor? descriptor)
        => descriptor is null
            ? "none"
            : descriptor.Granularity == Bfp8ScaleGranularity.Tensor
                ? "tensor"
                : $"block:{descriptor.BlockSize}";

    protected Parameter RegisterParameter(Parameter parameter)
    {
        ArgumentNullException.ThrowIfNull(parameter);
        parameter.AttachOwner(this);

        if (!_directParameters.Add(parameter))
        {
            throw new InvalidOperationException(
                $"Parameter '{parameter.Name}' is already registered in " +
                $"module '{GetType().Name}'.");
        }

        _members.Add(RegisteredMember.ForParameter(parameter));
        return parameter;
    }

    protected T RegisterModule<T>(T module)
        where T : Module
    {
        ArgumentNullException.ThrowIfNull(module);

        if (ReferenceEquals(module, this))
        {
            throw new InvalidOperationException(
                $"Module '{GetType().Name}' cannot register itself.");
        }

        if (!_directModules.Add(module))
        {
            throw new InvalidOperationException(
                $"Module '{module.GetType().Name}' is already registered in " +
                $"module '{GetType().Name}'.");
        }

        module.SetTraining(IsTraining);

        _members.Add(RegisteredMember.ForModule(module));
        return module;
    }

    internal void Train() => SetTraining(true);

    internal void Eval() => SetTraining(false);

    // PyTorch-style aliases. The PascalCase API remains available for
    // existing callers while new training code can use the familiar surface.
    public Module train()
    {
        Train();
        return this;
    }

    public Module eval()
    {
        Eval();
        return this;
    }

    internal Module To(TensorDevice device)
    {
        return device switch
        {
            TensorDevice.Cpu => MoveToCpu(),
            TensorDevice.Cuda => MoveToConfiguredCudaDevices(),
            _ => throw new ArgumentOutOfRangeException(
                nameof(device), device, "Unknown tensor device."),
        };
    }

    public Module to(TensorDevice device) => To(device);

    public Module to(TorchDevice device)
    {
        if (device.Type == TensorDevice.Cpu)
            return MoveToCpu();

        if (!Tensor.IsCudaAvailable(device.Index))
        {
            throw new InvalidOperationException(
                $"CUDA device {device.Index} is not available.");
        }
        foreach (Parameter parameter in Parameters())
            parameter.T.to(device);
        TensorExecutionContext.Device = device;
        return this;
    }

    /// <summary>
    /// Converts parameter storage in place while preserving Parameter,
    /// Tensor, autograd, and optimizer references.
    /// </summary>
    public Module to(TensorPrecisionMode precisionMode)
        => ToPrecision(precisionMode, Bfp8QuantizationDescriptor.DefaultBlockSize);

    /// <summary>
    /// Converts parameter storage in place. The block size is used only by
    /// <see cref="TensorPrecisionMode.Mix8_32"/>.
    /// </summary>
    public Module to(
        TensorPrecisionMode precisionMode,
        int bfp8_block_size)
        => ToPrecision(precisionMode, bfp8_block_size);

    /// <summary>
    /// Converts a module to a pure storage dtype. Mixed policies use the
    /// <see cref="TensorPrecisionMode"/> overload instead.
    /// </summary>
    public Module to(TensorDType dtype)
        => dtype switch
        {
            TensorDType.Float32 => ToPrecision(
                TensorPrecisionMode.Float32,
                Bfp8QuantizationDescriptor.DefaultBlockSize),
            TensorDType.BFloat16 => ToPrecision(
                TensorPrecisionMode.BFloat16,
                Bfp8QuantizationDescriptor.DefaultBlockSize),
            TensorDType.Bfp8 => ToPrecision(
                TensorPrecisionMode.Bfp8,
                Bfp8QuantizationDescriptor.DefaultBlockSize),
            _ => throw new NotSupportedException(
                $"Module conversion to dtype '{dtype}' is not supported. " +
                "Use TensorPrecisionMode.Mix16_32 or Mix8_32 for mixed precision."),
        };

    /// <summary>
    /// PyTorch-style string conversion for device and precision targets.
    /// Supported aliases include fp16_32 for mix16_32.
    /// </summary>
    public Module to(string target)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(target);
        string normalized = target.Trim().ToLowerInvariant();
        return normalized switch
        {
            "cpu" => MoveToCpu(),
            "cuda" => MoveToConfiguredCudaDevices(),
            "auto" => MoveToAutomaticallySelectedDevice(),
            "float32" => to(TensorPrecisionMode.Float32),
            "bfloat16" => to(TensorPrecisionMode.BFloat16),
            "mix16_32" or "fp16_32" => to(TensorPrecisionMode.Mix16_32),
            "bfp8" => to(TensorPrecisionMode.Bfp8),
            "mix8_32" => to(TensorPrecisionMode.Mix8_32),
            _ when normalized.StartsWith("cuda:", StringComparison.Ordinal)
                => to(TorchDevice.Parse(normalized)),
            _ => throw new ArgumentException(
                $"Unsupported module conversion target '{target}'. " +
                "Supported targets are cpu, cuda, cuda:N, auto, float32, " +
                "bfloat16, mix16_32 (fp16_32), bfp8, and mix8_32.",
                nameof(target)),
        };
    }

    private Module ToPrecision(
        TensorPrecisionMode precisionMode,
        int bfp8BlockSize)
    {
        if (!Enum.IsDefined(precisionMode))
            throw new ArgumentOutOfRangeException(nameof(precisionMode));
        if (precisionMode == TensorPrecisionMode.Mix8_32
            && bfp8BlockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(bfp8BlockSize),
                "Mix8_32 requires a positive BFP8 block size.");
        }

        TensorDType storageDType = precisionMode.ToStorageDType();
        Bfp8QuantizationDescriptor? quantization = precisionMode switch
        {
            TensorPrecisionMode.Bfp8 =>
                Bfp8QuantizationDescriptor.TensorWide,
            TensorPrecisionMode.Mix8_32 =>
                Bfp8QuantizationDescriptor.Block(bfp8BlockSize),
            _ => null,
        };
        bool preserveMaster = precisionMode is TensorPrecisionMode.Mix16_32
            or TensorPrecisionMode.Mix8_32;

        foreach (Parameter parameter in Parameters())
        {
            parameter.T.ConvertStorageInPlace(
                storageDType,
                quantization,
                preserveMaster);
        }
        SetNumericContractRecursively(
            precisionMode,
            storageDType,
            new HashSet<Module>(ReferenceEqualityComparer.Instance));
        return this;
    }

    private Module MoveToConfiguredCudaDevices()
    {
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        if (devices.Length == 0)
            throw new InvalidOperationException("No CUDA devices are configured.");
        foreach (int deviceIndex in devices)
        {
            if (!Tensor.IsCudaAvailable(deviceIndex))
            {
                throw new InvalidOperationException(
                    $"CUDA device {deviceIndex} is not available.");
            }
        }

        Parameter[] parameters = Parameters().ToArray();
        foreach (Parameter parameter in parameters)
        {
            foreach (int deviceIndex in devices)
            {
                parameter.T.to(
                    new TorchDevice(TensorDevice.Cuda, deviceIndex));
            }
            // The first configured adapter is authoritative for unsharded
            // operations; the remaining resident replicas stay allocated.
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        }
        TensorExecutionContext.Device = new TorchDevice(
            TensorDevice.Cuda,
            devices[0]);
        return this;
    }

    private Module MoveToAutomaticallySelectedDevice()
    {
        int[] available = Tensor.CudaDeviceIndices
            .Where(Tensor.IsCudaAvailable)
            .ToArray();
        if (available.Length == 0)
            return MoveToCpu();
        if (!available.SequenceEqual(Tensor.CudaDeviceIndices))
            Tensor.CudaDeviceIndices = available;
        return MoveToConfiguredCudaDevices();
    }

    private Module MoveToCpu()
    {
        foreach (Parameter parameter in Parameters())
        {
            if (parameter.T.HasGradientBuffer)
                parameter.T.EnsureHostGradientStorage();
            parameter.T.to(new TorchDevice(TensorDevice.Cpu));
            // An explicit model.to(cpu) is a move, so release accelerator
            // replicas after their authoritative values have been copied.
            parameter.T.InvalidateCudaBuffers();
        }
        TensorExecutionContext.Device = new TorchDevice(TensorDevice.Cpu);
        return this;
    }

    private void SetNumericContractRecursively(
        TensorPrecisionMode precisionMode,
        TensorDType dtype,
        HashSet<Module> visited)
    {
        if (!visited.Add(this))
        {
            throw new InvalidOperationException(
                $"Module '{GetType().Name}' is registered through multiple paths.");
        }
        PrecisionMode = precisionMode;
        DType = dtype;
        foreach (Module child in _directModules)
            child.SetNumericContractRecursively(precisionMode, dtype, visited);
    }

    internal ModuleState CaptureState()
    {
        Parameter[] parameters = Parameters().ToArray();
        var states = new ModuleParameterState[parameters.Length];
        for (int index = 0; index < parameters.Length; index++)
        {
            Parameter parameter = parameters[index];
            states[index] = new ModuleParameterState(
                index,
                parameter.Name,
                parameter.T.Shape.ToArray(),
                parameter.T.CaptureData(preferMaster: true),
                parameter.T.DType);
        }

        return new ModuleState(ModuleState.CurrentFormatVersion, states);
    }

    internal IReadOnlyList<IReadOnlyList<Parameter>>
        MakeGainShareParameterGroups(int blockDepth = 1)
    {
        if (blockDepth < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockDepth),
                blockDepth,
                "GainShare block depth must be non-negative.");
        }

        var groups = new List<IReadOnlyList<Parameter>>();
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);
        var seenModules =
            new HashSet<Module>(ReferenceEqualityComparer.Instance);
        CollectParameterGroups(
            this,
            blockDepth,
            groups,
            seenParameters,
            seenModules);

        return groups.AsReadOnly();
    }

    public IReadOnlyList<IReadOnlyList<Parameter>>
        make_gainshare_parameter_groups(int block_depth = 1)
        => MakeGainShareParameterGroups(block_depth);

    internal void RestoreState(ModuleState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        Parameter[] parameters = Parameters().ToArray();
        ValidateState(state, parameters);

        for (int index = 0; index < parameters.Length; index++)
        {
            ModuleParameterState parameterState = state.Parameters[index];
            if (parameters[index].T.DType == TensorDType.Bfp8)
            {
                parameters[index].T.RestoreBfp8ValuesInPlace(
                    parameterState.Values,
                    preserveFloat32Master:
                        PrecisionMode == TensorPrecisionMode.Mix8_32);
            }
            else
            {
                using Tensor.DataMutation mutation =
                    parameters[index].BeginUpdate();
                parameterState.Values.AsSpan().CopyTo(mutation.Values);
            }
        }
    }

    internal IEnumerable<Parameter> Parameters()
    {
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);
        var seenModules =
            new HashSet<Module>(ReferenceEqualityComparer.Instance);

        foreach (Parameter parameter in EnumerateParameters(
            this,
            seenParameters,
            seenModules))
        {
            yield return parameter;
        }
    }

    public IEnumerable<Parameter> parameters() => Parameters();

    public ModuleState state_dict() => CaptureState();

    public void load_state_dict(ModuleState state) => RestoreState(state);

    private static IEnumerable<Parameter> EnumerateParameters(
        Module module,
        HashSet<Parameter> seenParameters,
        HashSet<Module> seenModules)
    {
        if (!seenModules.Add(module))
        {
            throw new InvalidOperationException(
                $"Module '{module.GetType().Name}' is registered through " +
                "multiple paths.");
        }

        foreach (RegisteredMember member in module._members)
        {
            if (member.Parameter is { } parameter)
            {
                if (!seenParameters.Add(parameter))
                {
                    throw new InvalidOperationException(
                        $"Parameter '{parameter.Name}' is registered through " +
                        "multiple module paths.");
                }

                yield return parameter;
                continue;
            }

            if (member.ChildModule is { } childModule)
            {
                foreach (Parameter childParameter in EnumerateParameters(
                    childModule,
                    seenParameters,
                    seenModules))
                {
                    yield return childParameter;
                }
            }
        }
    }

    internal void ZeroGrad()
    {
        Parameter[] parameters = Parameters().ToArray();

        foreach (Parameter parameter in parameters)
            parameter.ZeroGrad();
    }

    public void zero_grad() => ZeroGrad();

    private void SetTraining(bool isTraining)
    {
        SetTraining(
            isTraining,
            new HashSet<Module>(ReferenceEqualityComparer.Instance));
    }

    private void SetTraining(
        bool isTraining,
        HashSet<Module> visited)
    {
        if (!visited.Add(this))
        {
            throw new InvalidOperationException(
                $"Module '{GetType().Name}' is registered through " +
                "multiple paths.");
        }

        IsTraining = isTraining;
        foreach (RegisteredMember member in _members)
        {
            member.ChildModule?.SetTraining(isTraining, visited);
        }
    }

    private static void ValidateState(
        ModuleState state,
        IReadOnlyList<Parameter> parameters)
    {
        if (state.FormatVersion != ModuleState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported module state format version " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{ModuleState.CurrentFormatVersion}'.",
                nameof(state));
        }

        if (state.Parameters is null
            || state.Parameters.Length != parameters.Count)
        {
            throw new ArgumentException(
                "Module state parameter count does not match the model.",
                nameof(state));
        }

        for (int index = 0; index < parameters.Count; index++)
        {
            Parameter parameter = parameters[index];
            ModuleParameterState parameterState = state.Parameters[index];
            if (parameterState is null
                || parameterState.Index != index
                || !string.Equals(
                    parameterState.Name,
                    parameter.Name,
                    StringComparison.Ordinal)
                || parameterState.Shape is null
                || !parameterState.Shape.SequenceEqual(parameter.T.Shape)
                || parameterState.Values is null
                || parameterState.Values.Length != parameter.T.Numel
                || parameterState.DType is not TensorDType.Float32
                    and not TensorDType.Float16
                    and not TensorDType.BFloat16
                    and not TensorDType.Bfp8
                || parameterState.StorageMetadata is { IsRaw: false })
            {
                throw new ArgumentException(
                    $"Module parameter state for slot {index} is " +
                    "incompatible.",
                    nameof(state));
            }

            if (parameterState.Values.Any(value => !float.IsFinite(value)))
            {
                throw new ArgumentException(
                    $"Module parameter state for slot {index} contains " +
                    "a non-finite value.",
                    nameof(state));
            }

            if ((parameterState.DType == TensorDType.Float16
                    || parameter.T.DType == TensorDType.Float16)
                && parameterState.Values.Any(
                    value => !Half.IsFinite((Half)value)))
            {
                throw new ArgumentException(
                    $"Module parameter state for slot {index} contains " +
                    "a value outside the finite Float16 range.",
                    nameof(state));
            }
        }
    }

    private static void CollectParameterGroups(
        Module module,
        int remainingDepth,
        List<IReadOnlyList<Parameter>> groups,
        HashSet<Parameter> seenParameters,
        HashSet<Module> seenModules)
    {
        if (!seenModules.Add(module))
        {
            throw new InvalidOperationException(
                $"Module '{module.GetType().Name}' is registered through " +
                "multiple paths.");
        }

        if (remainingDepth == 0)
        {
            var group = new List<Parameter>();
            CollectSubtreeParameters(
                module,
                group,
                seenParameters,
                seenModules,
                moduleAlreadyVisited: true);
            if (group.Count > 0)
                groups.Add(group.AsReadOnly());
            return;
        }

        var directGroup = new List<Parameter>();
        foreach (RegisteredMember member in module._members)
        {
            if (member.Parameter is not { } parameter)
                continue;
            if (!seenParameters.Add(parameter))
            {
                throw new InvalidOperationException(
                    $"Parameter '{parameter.Name}' is registered through " +
                    "multiple module paths.");
            }

            directGroup.Add(parameter);
        }

        if (directGroup.Count > 0)
            groups.Add(directGroup.AsReadOnly());

        foreach (RegisteredMember member in module._members)
        {
            if (member.ChildModule is { } childModule)
            {
                CollectParameterGroups(
                    childModule,
                    remainingDepth - 1,
                    groups,
                    seenParameters,
                    seenModules);
            }
        }
    }

    private static void CollectSubtreeParameters(
        Module module,
        List<Parameter> destination,
        HashSet<Parameter> seenParameters,
        HashSet<Module> seenModules,
        bool moduleAlreadyVisited = false)
    {
        if (!moduleAlreadyVisited && !seenModules.Add(module))
        {
            throw new InvalidOperationException(
                $"Module '{module.GetType().Name}' is registered through " +
                "multiple paths.");
        }

        foreach (RegisteredMember member in module._members)
        {
            if (member.Parameter is { } parameter)
            {
                if (!seenParameters.Add(parameter))
                {
                    throw new InvalidOperationException(
                        $"Parameter '{parameter.Name}' is registered " +
                        "through multiple module paths.");
                }

                destination.Add(parameter);
            }
            else if (member.ChildModule is { } childModule)
            {
                CollectSubtreeParameters(
                    childModule,
                    destination,
                    seenParameters,
                    seenModules);
            }
        }
    }

    private readonly record struct RegisteredMember(
        Parameter? Parameter,
        Module? ChildModule)
    {
        internal static RegisteredMember ForParameter(Parameter parameter)
            => new(parameter, null);

        internal static RegisteredMember ForModule(Module module)
            => new(null, module);
    }
}
