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
    }

    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Gets the physical storage dtype selected for this module's parameters.
    /// Stateless modules propagate this contract to their inputs and children.
    /// </summary>
    public TensorDType DType { get; }

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
        foreach (Parameter parameter in Parameters())
            parameter.T.To(device);
        return this;
    }

    public Module to(TensorDevice device) => To(device);

    public Module to(TorchDevice device)
    {
        foreach (Parameter parameter in Parameters())
            parameter.T.to(device);
        return this;
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
            using Tensor.DataMutation mutation =
                parameters[index].BeginUpdate();
            ModuleParameterState parameterState = state.Parameters[index];
            parameterState.Values.AsSpan().CopyTo(mutation.Values);
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
