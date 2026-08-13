namespace NNtrain;

public class Parameter
{
    public Parameter(
        float[] data,
        int[] shape,
        string name,
        WeightDecayPolicy weightDecay,
        TensorDType dtype = TensorDType.Float32)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException(
                "A parameter name cannot be null, empty, or whitespace.",
                nameof(name));
        }

        if (!Enum.IsDefined(weightDecay))
        {
            throw new ArgumentOutOfRangeException(
                nameof(weightDecay),
                weightDecay,
                "Unknown weight-decay policy.");
        }

        Name = name;
        WeightDecay = weightDecay;
        T = new Tensor(data, shape, name, dtype);
        T.EnableMasterData();
        if (dtype != TensorDType.Float32)
            data.AsSpan().CopyTo(T.DataBuffer);
    }

    public Tensor T { get; }
    public string Name { get; }
    public Module? Owner { get; private set; }
    public WeightDecayPolicy WeightDecay { get; }

    internal void AttachOwner(Module owner)
    {
        ArgumentNullException.ThrowIfNull(owner);

        if (Owner is null)
        {
            Owner = owner;
            return;
        }

        if (!ReferenceEquals(Owner, owner))
        {
            throw new InvalidOperationException(
                $"Parameter '{Name}' is already owned by module " +
                $"'{Owner.GetType().Name}' and cannot also be owned by " +
                $"'{owner.GetType().Name}'.");
        }
    }

    public void ZeroGrad()
    {
        T.ZeroGrad();
    }

    internal Tensor.DataMutation BeginUpdate()
        => T.BeginDataMutation();

    internal float[] DataBuffer => T.DataBuffer;

    internal void CompleteUpdate() => T.SynchronizeStorageFromMaster();
}
