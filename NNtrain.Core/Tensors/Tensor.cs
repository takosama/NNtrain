namespace NNtrain;

/// <summary>
/// A dense, row-major tensor with reverse-mode automatic differentiation.
/// </summary>
/// <remarks>
/// Shapes always contain at least one positive dimension. A scalar is represented
/// by shape <c>[1]</c>. Element-wise operators support equal shapes or broadcasting
/// when exactly one operand contains a single element.
/// </remarks>
public partial class Tensor
{
    private readonly float[] _data;
    private readonly float[] _grad;
    private readonly int[] _shape;
    private long _dataVersion;

    internal AutogradNode Node { get; }

    public IReadOnlyList<float> Data { get; }
    public IReadOnlyList<float> Grad { get; }
    public IReadOnlyList<int> Shape { get; }
    public string Name { get; }

    public int Rank => _shape.Length;
    public int Numel => _data.Length;

    internal Span<float> MutableGrad => _grad;
    internal long DataVersion => _dataVersion;

    public Tensor(float[] data, int[] shape, string name = "")
    {
        ArgumentNullException.ThrowIfNull(data);
        ValidateShape(shape, data.Length);

        _data = (float[])data.Clone();
        _grad = new float[data.Length];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Data = Array.AsReadOnly(_data);
        Grad = Array.AsReadOnly(_grad);
        Shape = Array.AsReadOnly(_shape);

        Node = new AutogradNode();
    }

    private Tensor(float[] data, int[] shape, Tensor[] prev, string name = "")
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(prev);
        ValidateShape(shape, data.Length);

        _data = data;
        _grad = new float[data.Length];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Data = Array.AsReadOnly(_data);
        Grad = Array.AsReadOnly(_grad);
        Shape = Array.AsReadOnly(_shape);

        Node = AutogradContext.IsRecordingEnabled
            ? new AutogradNode(prev)
            : AutogradNode.Detached();
    }

    public static Tensor Scalar(float value, string name = "")
        => new([value], [1], name);

    public static Tensor Zeros(params int[] shape)
    {
        int length = NumelOf(shape);
        return new Tensor(new float[length], shape);
    }

    public static Tensor From1D(float[] values, string name = "")
    {
        ArgumentNullException.ThrowIfNull(values);
        return new Tensor(values, [values.Length], name);
    }

    public static Tensor From2D(float[,] values, string name = "")
    {
        ArgumentNullException.ThrowIfNull(values);

        int rows = values.GetLength(0);
        int columns = values.GetLength(1);
        float[] data = new float[checked(rows * columns)];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                data[row * columns + column] = values[row, column];

        return new Tensor(data, [rows, columns], name);
    }

    public float this[int index]
    {
        get
        {
            CheckRank(1);
            return _data[index];
        }
    }

    public float this[int row, int column]
    {
        get
        {
            CheckRank(2);
            return _data[row * _shape[1] + column];
        }
    }

    /// <summary>
    /// Clears this tensor's gradient buffer without changing any other tensor.
    /// </summary>
    public void ZeroGrad() => ClearGradient();

    internal void ClearGradient() => _grad.AsSpan().Clear();

    internal DataMutation BeginDataMutation() => new(this);

    internal ref struct DataMutation
    {
        private Tensor? _owner;

        internal DataMutation(Tensor owner)
        {
            _owner = owner;
        }

        internal Span<float> Values
            => _owner?._data
                ?? throw new ObjectDisposedException(nameof(DataMutation));

        public void Dispose()
        {
            if (_owner is null)
                return;

            unchecked
            {
                _owner._dataVersion++;
            }

            _owner = null;
        }
    }

    private static int NumelOf(int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);

        if (shape.Length == 0)
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(shape));

        int elementCount = 1;
        for (int index = 0; index < shape.Length; index++)
        {
            int dimension = shape[index];
            if (dimension <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(shape),
                    dimension,
                    $"Shape dimension at index {index} must be positive.");
            }

            try
            {
                elementCount = checked(elementCount * dimension);
            }
            catch (OverflowException exception)
            {
                throw new ArgumentException(
                    "The product of the shape dimensions exceeds Int32.MaxValue.",
                    nameof(shape),
                    exception);
            }
        }

        return elementCount;
    }

    private static void ValidateShape(int[] shape, int dataLength)
    {
        int expectedLength = NumelOf(shape);
        if (expectedLength != dataLength)
        {
            throw new ArgumentException(
                $"Data length {dataLength} does not match shape [{string.Join(", ", shape)}] " +
                $"with {expectedLength} elements.",
                nameof(shape));
        }
    }

    private static ArgumentException ShapeMismatch(
        Tensor left,
        Tensor right,
        string operation)
        => new(
            $"{operation} cannot combine shapes {ShapeText(left)} and {ShapeText(right)}.");

    private static string ShapeText(Tensor tensor)
        => $"[{string.Join(", ", tensor._shape)}]";

    private void CheckRank(int expectedRank)
    {
        if (Rank != expectedRank)
        {
            throw new InvalidOperationException(
                $"Tensor rank is {Rank}; this operation requires rank {expectedRank}.");
        }
    }
}
