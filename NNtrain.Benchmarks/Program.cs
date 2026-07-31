using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

namespace NNtrain.Benchmarks;

internal static class Program
{
    internal static void Main(string[] args)
    {
        IConfig configuration = DefaultConfig.Instance
            .AddJob(CreateJob("Debug", "Debug", isBaseline: true))
            .AddJob(CreateJob("Release", "Release", isBaseline: false))
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddExporter(JsonExporter.Full);

        BenchmarkSwitcher
            .FromAssembly(typeof(Program).Assembly)
            .Run(args, configuration);
    }

    private static Job CreateJob(
        string id,
        string buildConfiguration,
        bool isBaseline)
    {
        return Job.Default
            .WithId(id)
            .WithCustomBuildConfiguration(buildConfiguration)
            .WithBaseline(isBaseline)
            .WithLaunchCount(1)
            .WithWarmupCount(5)
            .WithIterationCount(10)
            .WithInvocationCount(1)
            .WithUnrollFactor(1);
    }
}
