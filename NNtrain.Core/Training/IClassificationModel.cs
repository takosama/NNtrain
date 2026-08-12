namespace NNtrain;

public interface IClassificationModel
{
    int InputRows { get; }

    int InputColumns { get; }

    int ClassCount { get; }

    Tensor Forward(Tensor input);

    Tensor forward(Tensor input) => Forward(input);
}
