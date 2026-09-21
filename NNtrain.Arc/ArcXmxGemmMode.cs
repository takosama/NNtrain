namespace NNtrain.Arc;

/// <summary>Session-local selection for measured XMX GEMM A/B comparisons.</summary>
public enum ArcXmxGemmMode
{
    Auto,
    Legacy,
    Narrow,
    Wide,
}
