namespace NNtrain;

public abstract class Module
{
    private readonly List<RegisteredMember> _members = [];
    private readonly HashSet<Parameter> _directParameters =
        new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<Module> _directModules =
        new(ReferenceEqualityComparer.Instance);

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

        _members.Add(RegisteredMember.ForModule(module));
        return module;
    }

    public IEnumerable<Parameter> Parameters()
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

    public void ZeroGrad()
    {
        Parameter[] parameters = Parameters().ToArray();

        foreach (Parameter parameter in parameters)
            parameter.ZeroGrad();
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
