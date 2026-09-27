using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Small explicit GPU reverse-mode graph, retaining activations rather than dense weights.</summary>
internal sealed class Qwen35TrainingTape : IDisposable
{
    internal sealed class Value(ArcExecutionLane lane, ArcBuffer data, int rows, int width, bool differentiable, bool ownsData) : IDisposable
    {
        internal readonly ArcExecutionLane Lane = lane;
        internal readonly ArcBuffer Data = data;
        internal readonly int Rows = rows, Width = width;
        internal readonly bool Differentiable = differentiable;
        internal ArcBuffer? Gradient;
        internal ArcBuffer Grad()
        {
            if (Gradient is null)
            {
                Gradient = Lane.Allocate(checked(Rows * Width));
                Lane.Run("q35a_zero", (long)Rows * Width, 0, Gradient, Rows * Width);
            }
            return Gradient;
        }
        public void Dispose() { Gradient?.Dispose(); if (ownsData) Data.Dispose(); }
    }
    private readonly List<IDisposable> _owned = [];
    private readonly List<Action> _backward = [];
    internal Value Add(ArcExecutionLane lane, ArcBuffer data, int rows, int width, bool differentiable, bool ownsData = true)
    {
        var value = new Value(lane, data, rows, width, differentiable, ownsData); _owned.Add(value); return value;
    }
    internal T Own<T>(T resource) where T : IDisposable { _owned.Add(resource); return resource; }
    internal void Record(Action action) => _backward.Add(action);
    internal void Backward() { for (int i = _backward.Count - 1; i >= 0; i--) _backward[i](); }

    internal Value SliceRows(Value input, int start, int count)
    {
        if (start < 0 || count < 1 || start + count > input.Rows) throw new ArgumentOutOfRangeException(nameof(start));
        if (start == 0 && count == input.Rows) return input;
        int elements = checked(count * input.Width), offset = checked(start * input.Width);
        Value output = Add(input.Lane, input.Lane.Allocate(elements), count, input.Width, input.Differentiable);
        input.Lane.CopyBytes(input.Data, output.Data, checked(offset * 4), 0, checked(elements * 4));
        if (input.Differentiable) Record(() =>
        {
            if (output.Gradient is not null)
                input.Lane.Run("q35t_add_offset", elements, 0, output.Gradient, input.Grad(), elements, offset);
        });
        return output;
    }

    internal Value Move(Value input, ArcExecutionLane destination)
    {
        if (ReferenceEquals(input.Lane, destination)) return input;
        var staging = new float[checked(input.Rows * input.Width)];
        input.Lane.Read(input.Data, staging);
        Value output = Add(destination, destination.Upload(staging), input.Rows, input.Width, input.Differentiable);
        if (input.Differentiable) Record(() =>
        {
            if (output.Gradient is null) return;
            destination.Read(output.Gradient, staging);
            using ArcBuffer moved = input.Lane.Upload(staging);
            Qwen35Gpu.AddInPlace(input.Lane, input.Grad(), moved, staging.Length);
        });
        return output;
    }
    internal Value Sum(Value a, Value b)
    {
        if (!ReferenceEquals(a.Lane, b.Lane) || a.Rows != b.Rows || a.Width != b.Width)
            throw new ArgumentException("Training residual shape/lane mismatch.");
        int n = checked(a.Rows * a.Width);
        var output = Add(a.Lane, a.Lane.Allocate(n), a.Rows, a.Width, a.Differentiable || b.Differentiable);
        a.Lane.Run("q35t_add", n, 0, a.Data, b.Data, output.Data, n);
        Record(() =>
        {
            if (output.Gradient is null) return;
            if (a.Differentiable) Qwen35Gpu.AddInPlace(a.Lane, a.Grad(), output.Gradient, n);
            if (b.Differentiable) Qwen35Gpu.AddInPlace(b.Lane, b.Grad(), output.Gradient, n);
        });
        return output;
    }
    internal Value Norm(Value input, ArcBuffer weight, float epsilon)
    {
        Value output = Add(input.Lane, Qwen35Gpu.RmsNorm(input.Lane, input.Data, weight,
            input.Rows, input.Width, epsilon), input.Rows, input.Width, input.Differentiable);
        if (input.Differentiable) Record(() =>
        {
            if (output.Gradient is not null) input.Lane.Run("q35t_rms_dx", (long)input.Rows * 128, 128,
                input.Data, weight, output.Gradient, input.Grad(), input.Rows, input.Width, epsilon);
        });
        return output;
    }
    internal Value Silu(Value gate, Value up)
    {
        int n = checked(gate.Rows * gate.Width);
        Value output = Add(gate.Lane, Qwen35Gpu.SiluMultiply(gate.Lane, gate.Data, up.Data, n),
            gate.Rows, gate.Width, gate.Differentiable || up.Differentiable);
        Record(() =>
        {
            if (output.Gradient is null) return;
            gate.Lane.Run("q35t_silu_dx", n, 0, gate.Data, up.Data, output.Gradient, gate.Grad(), up.Grad(), n);
        });
        return output;
    }
    public void Dispose() { for (int i = _owned.Count - 1; i >= 0; i--) _owned[i].Dispose(); }
}
