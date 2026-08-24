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
    /// <summary>
    /// Gets or selects the execution device for kernels with a GPU backend.
    /// Kernels not yet ported continue to use their CPU implementation.
    /// </summary>
    public static TensorDevice ExecutionDevice
    {
        get => TensorExecutionContext.Device.Type;
        set
        {
            if (!Enum.IsDefined(value))
                throw new ArgumentOutOfRangeException(nameof(value));
            TorchDevice current = TensorExecutionContext.Device;
            TensorExecutionContext.Device = new TorchDevice(
                value,
                value == TensorDevice.Cuda ? current.Index : 0);
        }
    }

    /// <summary>Gets or selects the zero-based CUDA adapter index.</summary>
    public static int CudaDeviceIndex
    {
        get => TensorExecutionContext.Device.IsCuda
            ? TensorExecutionContext.Device.Index
            : TensorExecutionContext.CudaDevices[0];
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            TensorExecutionContext.CudaDevices = [value];
        }
    }

    /// <summary>
    /// Gets or selects the CUDA adapters used by data-parallel kernels.
    /// The first adapter is used by kernels that cannot be partitioned.
    /// </summary>
    public static IReadOnlyList<int> CudaDeviceIndices
    {
        get => TensorExecutionContext.CudaDevices;
        set => TensorExecutionContext.CudaDevices = value;
    }

    /// <summary>Gets the active adapter name and initializes CUDA on demand.</summary>
    public static string ExecutionDeviceName
        => TensorBackends.Get(ExecutionDevice).GetName(CudaDeviceIndex);

    /// <summary>Reports whether the requested CUDA adapter can be initialized.</summary>
    public static bool IsCudaAvailable(int deviceIndex = 0)
        => TensorBackends.Get(TensorDevice.Cuda).IsAvailable(deviceIndex);

    /// <summary>Gets the number of CUDA adapters visible to the runtime.</summary>
    public static int CudaDeviceCount => NNtrain.ForgetMemoryV2Cuda.DeviceCount;

    private readonly TensorStorage _data;
    private float[]? _masterData;
    private float[]? _physicalFloat32Cache;
    private long _physicalFloat32CacheDataVersion = -1;
    private float[] _grad;
    private readonly int[] _shape;
    private long _dataVersion;
    private readonly object _transposeCacheLock = new();
    private float[]? _transposedDataCache;
    private long _transposedDataVersion = -1;

    internal AutogradNode Node { get; }

    public IReadOnlyList<float> Data
    {
        get
        {
            EnsureHostDataCurrent();
            return _data;
        }
    }
    public IReadOnlyList<float> Grad { get; }
    public IReadOnlyList<int> Shape { get; }
    public string Name { get; }

    /// <summary>Gets the physical storage dtype.</summary>
    public TensorDType DType => _data.DType;

    /// <summary>Gets the dtype used for tensor operation results.</summary>
    public TensorDType ComputeDType
        => DType == TensorDType.BFloat16
            ? TensorDType.BFloat16
            : TensorDType.Float32;

    /// <summary>Gets the dtype used by reductions and gradient accumulation.</summary>
    public TensorDType AccumulationDType
        => DType == TensorDType.BFloat16
            ? TensorDType.BFloat16
            : TensorDType.Float32;

    public int Rank => _shape.Length;
    public int Numel => _data.Count;

    public IReadOnlyList<float> data => Data;
    public IReadOnlyList<float> grad => Grad;
    public IReadOnlyList<int> shape => Shape;
    public int ndim => Rank;
    public int numel() => Numel;

    public float item()
    {
        if (Numel != 1)
        {
            throw new InvalidOperationException(
                "item() requires a tensor containing exactly one value.");
        }
        EnsureHostDataCurrent();
        return _data[0];
    }

    internal Span<float> MutableGrad => EnsureGradientBuffer();
    internal float[] GradientBuffer
    {
        get
        {
            EnsureHostGradientCurrent();
            return _grad;
        }
    }
    internal long DataVersion => _dataVersion;
    internal bool HasGradientBuffer
        => _grad.Length != 0 || _cudaGradientBuffers.Count != 0;

    public Tensor(
        float[] data,
        int[] shape,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
    {
        ArgumentNullException.ThrowIfNull(data);
        ValidateShape(shape, data.Length);
        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));

        _data = TensorStorage.Create(data, dtype);
        _grad = [];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Grad = new GradientView(this);
        Shape = Array.AsReadOnly(_shape);

        Node = new AutogradNode();
    }

    /// <summary>
    /// Creates a leaf tensor without copying a newly allocated data array.
    /// The caller must not mutate the array after ownership is transferred.
    /// </summary>
    public static Tensor FromOwnedData(
        float[] data,
        int[] shape,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
    {
        ArgumentNullException.ThrowIfNull(data);
        ValidateShape(shape, data.Length);

        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));
        return new Tensor(data, shape, name, takeOwnership: true, dtype);
    }

    private Tensor(
        float[] data,
        int[] shape,
        string name,
        bool takeOwnership,
        TensorDType dtype)
    {
        _data = dtype == TensorDType.Float32 && takeOwnership
            ? TensorStorage.FromOwnedFloat32(data)
            : TensorStorage.Create(data, dtype);
        _grad = [];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Grad = new GradientView(this);
        Shape = Array.AsReadOnly(_shape);
        Node = new AutogradNode();
    }

    private Tensor(
        float[] data,
        int[] shape,
        Tensor[] prev,
        string name = "",
        TensorDType? dtype = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(prev);
        ValidateShape(shape, data.Length);

        TensorDType resultDType = dtype
            ?? TensorDTypeContract.Promote(prev);
        TensorDTypeContract.ValidateImplemented(
            resultDType,
            nameof(dtype));
        _data = resultDType == TensorDType.Float32
            ? TensorStorage.FromOwnedFloat32(data)
            : TensorStorage.Create(data, resultDType);
        bool isRecording = AutogradContext.IsRecordingEnabled;
        _grad = isRecording && ExecutionDevice != TensorDevice.Cuda
            ? new float[data.Length]
            : [];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Grad = new GradientView(this);
        Shape = Array.AsReadOnly(_shape);

        if (isRecording)
        {
            foreach (Tensor parent in prev)
            {
                if (ExecutionDevice != TensorDevice.Cuda
                    || parent.Device != TensorDevice.Cuda)
                    parent.EnsureGradientBuffer();
            }
            Node = new AutogradNode(prev);
        }
        else
        {
            Node = AutogradNode.Detached();
        }
    }

    private Tensor(
        TensorStorage data,
        int[] shape,
        Tensor[] prev,
        string name = "")
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(prev);
        ValidateShape(shape, data.Count);

        _data = data;
        bool isRecording = AutogradContext.IsRecordingEnabled;
        _grad = isRecording && ExecutionDevice != TensorDevice.Cuda
            ? new float[data.Count]
            : [];
        _shape = (int[])shape.Clone();
        Name = name ?? throw new ArgumentNullException(nameof(name));

        Grad = new GradientView(this);
        Shape = Array.AsReadOnly(_shape);

        if (isRecording)
        {
            foreach (Tensor parent in prev)
            {
                if (ExecutionDevice != TensorDevice.Cuda
                    || parent.Device != TensorDevice.Cuda)
                    parent.EnsureGradientBuffer();
            }
            Node = new AutogradNode(prev);
        }
        else
        {
            Node = AutogradNode.Detached();
        }
    }

    private static Tensor FromStorageResult(
        TensorStorage data,
        int[] shape,
        Tensor[] prev,
        string name = "")
        => new(data, shape, prev, name);

    private static Tensor FromFloat16Result(
        Half[] data,
        int[] shape,
        Tensor[] prev,
        string name = "")
        => FromStorageResult(
            TensorStorage.FromOwnedFloat16(data),
            shape,
            prev,
            name);

    public static Tensor Scalar(
        float value,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
        => new([value], [1], name, dtype);

    public static Tensor tensor(
        float[] data,
        int[] shape,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
        => new(data, shape, name, dtype);

    public static Tensor Zeros(params int[] shape)
        => Zeros(TensorDType.Float32, shape);

    public static Tensor Zeros(
        TensorDType dtype,
        params int[] shape)
    {
        int length = NumelOf(shape);
        return new Tensor(new float[length], shape, dtype: dtype);
    }

    public static Tensor From1D(
        float[] values,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new Tensor(values, [values.Length], name, dtype);
    }

    public static Tensor From2D(
        float[,] values,
        string name = "",
        TensorDType dtype = TensorDType.Float32)
    {
        ArgumentNullException.ThrowIfNull(values);

        int rows = values.GetLength(0);
        int columns = values.GetLength(1);
        float[] data = new float[checked(rows * columns)];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                data[row * columns + column] = values[row, column];

        return new Tensor(data, [rows, columns], name, dtype);
    }

    /// <summary>Converts physical storage and operation result dtype.</summary>
    public Tensor To(TensorDType dtype)
    {
        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));
        if (dtype == DType)
            return this;

        var result = new Tensor(
            _data.ToFloat32Array(),
            _shape,
            [this],
            dtype: dtype);
        result.Node.BackwardAction = () => AddScaledValues(
            EnsureGradientBuffer(),
            0,
            result._grad,
            0,
            1f,
            Numel);
        return result;
    }

    public Tensor to(TensorDType dtype) => To(dtype);

    public Tensor Half() => To(TensorDType.Float16);

    public Tensor half() => Half();

    public Tensor ToFloat32() => To(TensorDType.Float32);

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

    public void zero_grad() => ZeroGrad();

    internal void ClearGradient()
    {
        _grad.AsSpan().Clear();
        ClearCudaGradients();
    }

    internal void ClearGradientRange(int start, int length)
    {
        if (_grad.Length != 0)
            _grad.AsSpan(start, length).Clear();
    }

    private float[] EnsureGradientBuffer()
    {
        EnsureHostGradientCurrent();
        if (_grad.Length == 0)
            _grad = new float[Numel];
        MarkHostGradientMutable();
        return _grad;
    }

    internal DataMutation BeginDataMutation() => new(this);

    internal float[] DataBuffer
        => DType == TensorDType.Float32
            ? _data.GetMutableFloat32Buffer()
            : _masterData ??= _data.ToFloat32Array();

    internal int StorageByteLength => _data.ByteLength;

    internal void EnableMasterData()
    {
        if (DType != TensorDType.Float32)
            _masterData ??= _data.ToFloat32Array();
    }

    internal float[] CaptureData(bool preferMaster)
    {
        EnsureHostDataCurrent();
        return preferMaster && _masterData is not null
            ? (float[])_masterData.Clone()
            : _data.ToFloat32Array();
    }

    /// <summary>
    /// Gets a versioned Float32 decoding of a physical Float16 payload.
    /// This is intentionally not the optimizer's Float32 master buffer.
    /// Dense parameter tensors can reuse it across many activation rows.
    /// </summary>
    internal float[] GetPhysicalFloat32ComputeCache()
    {
        EnsureHostDataCurrent();
        if (DType == TensorDType.Float32)
            return _data.GetMutableFloat32Buffer();

        long version = _dataVersion;
        if (_physicalFloat32Cache is not null
            && _physicalFloat32CacheDataVersion == version)
        {
            return _physicalFloat32Cache;
        }

        lock (_transposeCacheLock)
        {
            if (_physicalFloat32Cache is null
                || _physicalFloat32Cache.Length != Numel)
            {
                _physicalFloat32Cache = new float[Numel];
                _physicalFloat32CacheDataVersion = -1;
            }
            if (_physicalFloat32CacheDataVersion != version)
            {
                _data.CopyTo(_physicalFloat32Cache);
                _physicalFloat32CacheDataVersion = version;
            }
            return _physicalFloat32Cache;
        }
    }

    internal void SynchronizeStorageFromMaster()
    {
        if (_masterData is not null)
            _data.CopyFrom(_masterData);
        MarkDataMutated();
    }

    internal void MarkDataMutated()
    {
        InvalidateCudaBuffers();
        CudaResidentArrayCache.Invalidate(_physicalFloat32Cache);
        if (DType == TensorDType.Float32
            && _data.TryGetFloat32Buffer(out float[] values))
        {
            CudaResidentArrayCache.Invalidate(values);
        }
        unchecked
        {
            _dataVersion++;
        }
        _physicalFloat32CacheDataVersion = -1;
    }

    internal float[] GetTransposedData2D()
    {
        CheckRank(2);
        long version = _dataVersion;
        float[]? cached = _transposedDataCache;
        if (cached is not null && _transposedDataVersion == version)
            return cached;

        lock (_transposeCacheLock)
        {
            if (_transposedDataCache is not null
                && _transposedDataVersion == version)
            {
                return _transposedDataCache;
            }

            int rows = _shape[0];
            int columns = _shape[1];
            var transposed = new float[_data.Count];
            const int BlockSize = 32;
            for (int columnBlock = 0;
                columnBlock < columns;
                columnBlock += BlockSize)
            {
                int columnEnd = Math.Min(columns, columnBlock + BlockSize);
                for (int rowBlock = 0;
                    rowBlock < rows;
                    rowBlock += BlockSize)
                {
                    int rowEnd = Math.Min(rows, rowBlock + BlockSize);
                    for (int column = columnBlock;
                        column < columnEnd;
                        column++)
                    {
                        int destination = column * rows + rowBlock;
                        for (int row = rowBlock; row < rowEnd; row++)
                        {
                            transposed[destination++] =
                                _data[row * columns + column];
                        }
                    }
                }
            }

            _transposedDataCache = transposed;
            _transposedDataVersion = version;
            return transposed;
        }
    }

    internal ref struct DataMutation
    {
        private Tensor? _owner;

        internal DataMutation(Tensor owner)
        {
            _owner = owner;
        }

        internal Span<float> Values
            => _owner?.DataBuffer
                ?? throw new ObjectDisposedException(nameof(DataMutation));

        public void Dispose()
        {
            if (_owner is null)
                return;

            _owner.SynchronizeStorageFromMaster();

            _owner = null;
        }
    }

    private sealed class GradientView(Tensor owner)
        : IList<float>, IReadOnlyList<float>
    {
        public int Count => owner.Numel;
        public bool IsReadOnly => true;

        public float this[int index]
        {
            get
            {
                ArgumentOutOfRangeException.ThrowIfNegative(index);
                if (index >= Count)
                    throw new ArgumentOutOfRangeException(nameof(index));
                owner.EnsureHostGradientCurrent();
                return owner._grad.Length == 0 ? 0f : owner._grad[index];
            }
            set => throw new NotSupportedException();
        }

        public IEnumerator<float> GetEnumerator()
        {
            for (int index = 0; index < Count; index++)
                yield return this[index];
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
            => GetEnumerator();

        public int IndexOf(float item)
        {
            for (int index = 0; index < Count; index++)
            {
                if (this[index].Equals(item))
                    return index;
            }

            return -1;
        }

        public bool Contains(float item) => IndexOf(item) >= 0;

        public void CopyTo(float[] array, int arrayIndex)
        {
            ArgumentNullException.ThrowIfNull(array);
            for (int index = 0; index < Count; index++)
                array[arrayIndex + index] = this[index];
        }

        public void Add(float item) => throw new NotSupportedException();
        public void Clear() => throw new NotSupportedException();
        public void Insert(int index, float item) => throw new NotSupportedException();
        public bool Remove(float item) => throw new NotSupportedException();
        public void RemoveAt(int index) => throw new NotSupportedException();
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
