namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Adds a [rows, columns] tensor to every item of a
    /// [batch, rows, columns] tensor.
    /// </summary>
    public Tensor AddBatchWise(Tensor matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        CheckRank(3);
        matrix.CheckRank(2);

        int batch = _shape[0];
        int rows = _shape[1];
        int columns = _shape[2];
        if (matrix._shape[0] != rows || matrix._shape[1] != columns)
            throw ShapeMismatch(this, matrix, "Batch-wise addition");

        int matrixLength = rows * columns;
        float[] output = (float[])_data.Clone();
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            AddScaledValues(
                output,
                batchIndex * matrixLength,
                matrix._data,
                0,
                1f,
                matrixLength);
        }

        var result = new Tensor(output, _shape, [this, matrix]);
        result.Node.BackwardAction = () =>
        {
            AddScaledValues(
                _grad,
                0,
                result._grad,
                0,
                1f,
                Numel);

            for (int batchIndex = 0; batchIndex < batch; batchIndex++)
            {
                AddScaledValues(
                    matrix._grad,
                    0,
                    result._grad,
                    batchIndex * matrixLength,
                    1f,
                    matrixLength);
            }
        };

        return result;
    }
}
