namespace NNtrain;

partial class Tensor
{
    public override string ToString() => DataString();

    public string DataString()
    {
        if (Rank == 1)
            return "[" + string.Join(", ", _data.Select(v => v.ToString("F4"))) + "]";

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];
            string[] lines = new string[rows];
            for (int r = 0; r < rows; r++)
            {
                string[] vals = new string[cols];
                for (int c = 0; c < cols; c++)
                    vals[c] = _data[r * cols + c].ToString("F4");
                lines[r] = "[" + string.Join(", ", vals) + "]";
            }
            return string.Join(Environment.NewLine, lines);
        }

        return "[" + string.Join(", ", _data.Select(v => v.ToString("F4"))) + "]";
    }

    public string GradString()
    {
        if (Rank == 1)
            return "[" + string.Join(", ", Grad.Select(v => v.ToString("F4"))) + "]";

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];
            string[] lines = new string[rows];
            for (int r = 0; r < rows; r++)
            {
                string[] vals = new string[cols];
                for (int c = 0; c < cols; c++)
                    vals[c] = Grad[r * cols + c].ToString("F4");
                lines[r] = "[" + string.Join(", ", vals) + "]";
            }
            return string.Join(Environment.NewLine, lines);
        }

        return "[" + string.Join(", ", Grad.Select(v => v.ToString("F4"))) + "]";
    }
}
