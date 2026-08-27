using Xunit;

// Tensor execution device, CUDA adapter selection, and precision mode are
// process-wide runtime settings. Running test classes concurrently can switch
// those settings while another CUDA assertion is still synchronizing, which
// produces nondeterministic cross-device numerical failures.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
