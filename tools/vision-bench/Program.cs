using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Security.Cryptography;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using NNtrain;
using NNtrain.Gui;

string label = args.FirstOrDefault() ?? "baseline";
bool optimized = !args.Contains("--reference");
bool useXmxLinear = args.Contains("--xmx");
bool useXmxAttention = args.Contains("--xmx-attention");
bool useFlashAttention = args.Contains("--flash-attention");
bool batchAttention = !args.Contains("--no-batch-attention");
bool collectModelTimings = args.Contains("--profile");
bool legacy = args.Contains("--legacy");
bool xmxPrefill = args.Contains("--xmx-prefill");
bool packedPrefill = args.Contains("--packed-prefill");
bool factoredPrefill = args.Contains("--factored-prefill");
bool residentIq2Panels = args.Contains("--resident-iq2");
bool ggufBslmPrefill = args.Contains("--gguf-bslm");
bool prepareAttachedImage = args.Contains("--prepared-image");
bool prepareVisionOnLoad = !args.Contains("--reference") && !legacy;
bool batchRecurrent = !legacy && !args.Contains("--no-batch-recurrent");
bool subgroupRecurrentRms = !legacy && !args.Contains("--no-subgroup-rms");
int bufferPoolMiB = 64, deferredReleaseMiB = 0;
int imageSize = 768;
int prefill = args.Contains("--reference") ? 0 : 16;
int generationRuns = 2;
int projectionRows = optimized ? 4 : 1;
foreach (string arg in args)
{
    if (arg.StartsWith("--prefill=")) prefill = int.Parse(arg[10..]);
    if (arg.StartsWith("--runs=")) generationRuns = int.Parse(arg[7..]);
    if (arg.StartsWith("--projection-rows=")) projectionRows = int.Parse(arg[18..]);
    if (arg.StartsWith("--pool=")) bufferPoolMiB = int.Parse(arg[7..]);
    if (arg.StartsWith("--deferred=")) deferredReleaseMiB = int.Parse(arg[11..]);
    if (arg.StartsWith("--image-size=")) imageSize = int.Parse(arg[13..]);
}
string root = Environment.CurrentDirectory;
string outputDirectory = Path.Combine(root, "benchmark-results", args.Contains("--v2")
    ? "vision-ttft-v2-20261005" : "vision-speed-20261005");
Directory.CreateDirectory(outputDirectory);
if (args.Contains("--parity"))
{
    RealPrefillParity.Run(args, outputDirectory, prefill, bufferPoolMiB, deferredReleaseMiB);
    return;
}
string projection = Path.Combine(root, "models/huggingface/unsloth/Qwen3.5-27B-GGUF/mmproj-F16.gguf");
string? adapterPath = args.Contains("--lora") ? Path.Combine(root,
    "models/lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf") : null;
var records = new List<object>();
var timer = Stopwatch.StartNew();
if (!args.Contains("--generation-only"))
using (var encoder = Qwen35VisionEncoder.Load(projection, collectKernelTimings: true,
    useOptimizedKernels: optimized, useXmxLinear: useXmxLinear, useXmxAttention: useXmxAttention,
    useFlashAttention: useFlashAttention))
{
    double loadMs = timer.Elapsed.TotalMilliseconds;
    foreach (int size in new[] { 256, 768 })
    {
        byte[] rgb = new byte[size * size * 3];
        for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
        {
            int i = (y * size + x) * 3;
            rgb[i] = (byte)(x * 255 / size);
            rgb[i + 1] = (byte)(y * 255 / size);
            rgb[i + 2] = (byte)((x ^ y) % 256);
        }
        timer.Restart();
        var input = Qwen35VisionPreprocessor.Prepare(rgb, size, size);
        double prepareMs = timer.Elapsed.TotalMilliseconds;
        byte[] patchBytes = new byte[input.Patches.Length * sizeof(float)];
        Buffer.BlockCopy(input.Patches, 0, patchBytes, 0, patchBytes.Length);
        string patchSha256 = Convert.ToHexString(SHA256.HashData(patchBytes)).ToLowerInvariant();
        var runs = new List<double>();
        Dictionary<string, double> before = encoder.KernelTimings.ToDictionary();
        for (int run = 0; run < 3; run++)
        {
            timer.Restart();
            Qwen35VisionEmbedding image = encoder.Encode(input);
            runs.Add(timer.Elapsed.TotalMilliseconds);
            if (run == 0)
            {
                byte[] bytes = new byte[image.Values.Length * sizeof(float)];
                Buffer.BlockCopy(image.Values, 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(Path.Combine(outputDirectory, $"{label}-{size}.f32"), bytes);
            }
        }
        var kernels = encoder.KernelTimings.ToDictionary(pair => pair.Key,
            pair => pair.Value - before.GetValueOrDefault(pair.Key));
        var record = new { size, load_ms = loadMs, program_cache_hit = encoder.ProgramBinaryCacheHit,
            uses_xmx_linear = encoder.UsesXmxLinear, resident_weights_bytes = encoder.ResidentWeightBytes,
            patch_sha256 = patchSha256, prepare_ms = prepareMs, encode_ms = runs, kernels_ms = kernels };
        records.Add(record);
        Console.WriteLine(JsonSerializer.Serialize(record));
    }
}
byte[] large = new byte[4096 * 3072 * 3];
new Random(42).NextBytes(large);
timer.Restart();
_ = Qwen35VisionPreprocessor.Prepare(large, 4096, 3072);
double largePrepareMs = timer.Elapsed.TotalMilliseconds;
Console.WriteLine($"large RGB prepare: {largePrepareMs:F2}ms");
if (!args.Contains("--encoder-only"))
{
    ChatImage MakeImage(int color)
    {
        byte[] pixels = new byte[imageSize * imageSize * 3];
        for (int i = 0; i < imageSize * imageSize; i++) pixels[i * 3 + color] = 255;
        var bitmap = BitmapSource.Create(imageSize, imageSize, 96, 96, PixelFormats.Rgb24, null, pixels, imageSize * 3);
        var png = new PngBitmapEncoder();
        png.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = new MemoryStream();
        png.Save(stream);
        return new ChatImage(stream.ToArray());
    }
    ChatImage image = MakeImage(0);
    using var session = new InferenceSession(prefill, optimized, batchAttention, collectModelTimings, projectionRows,
        prepareVisionOnLoad, xmxPrefill, bufferPoolMiB, deferredReleaseMiB, batchRecurrent, packedPrefill,
        useXmxLinear, useXmxAttention, factoredPrefill, useFlashAttention, residentIq2Panels,
        subgroupRecurrentRms, ggufBslmPrefill);
    timer.Restart();
    await session.LoadAsync(Path.Combine(root, "models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf"),
        adapterPath, new Progress<string>(Console.WriteLine), CancellationToken.None, [0, 1], projection);
    double modelLoadMs = timer.Elapsed.TotalMilliseconds;
    var generation = new List<object>();
    for (int run = 0; run < generationRuns; run++)
    {
        if (run > 0 && args.Contains("--new-images")) image = MakeImage(run % 3);
        ImagePreparationStats? preparationStats = null;
        double? attachmentPrepareMs = null;
        if (prepareAttachedImage)
        {
            var preparationTimer = Stopwatch.StartNew();
            preparationStats = await session.PrepareImageAsync(image,
                [new("user", "", Image: image)], CancellationToken.None);
            attachmentPrepareMs = preparationTimer.Elapsed.TotalMilliseconds;
        }
        timer.Restart();
        double? firstTextMs = null;
        Dictionary<string, double> before = session.KernelMilliseconds.ToDictionary();
        string answer = await session.GenerateAsync([new("user", "この画像の色を日本語で答えて。", Image: image)],
            false, 24, _ => firstTextMs ??= timer.Elapsed.TotalMilliseconds, CancellationToken.None, new(0, 1, 1));
        var kernels = session.KernelMilliseconds.ToDictionary(pair => pair.Key,
            pair => pair.Value - before.GetValueOrDefault(pair.Key));
        var record = new { run, elapsed_ms = timer.Elapsed.TotalMilliseconds,
            first_text_ms = firstTextMs, answer, stats = session.LastGenerationStats, kernels_ms = kernels,
            image_prepare_ms = session.LastImagePreparationMilliseconds,
            attachment_prepare_ms = attachmentPrepareMs, preparation_stats = preparationStats,
            mmproj_initialization_ms = session.VisionInitializationMilliseconds };
        generation.Add(record);
        Console.WriteLine(JsonSerializer.Serialize(record));
    }
    records.Add(new { model_load_ms = modelLoadMs, generation });
}
File.WriteAllText(Path.Combine(outputDirectory, label + ".json"), JsonSerializer.Serialize(
    new { label, optimized, use_xmx_linear = useXmxLinear, use_xmx_attention = useXmxAttention,
        use_flash_attention = useFlashAttention, prefill_chunk = prefill,
        prepared_image = prepareAttachedImage,
        batch_attention = batchAttention, collect_model_timings = collectModelTimings,
        projection_rows = projectionRows,
        xmx_prefill = xmxPrefill, packed_prefill = packedPrefill, factored_prefill = factoredPrefill,
        resident_iq2_panels = residentIq2Panels,
        gguf_bslm_prefill = ggufBslmPrefill,
        adapter_path = adapterPath,
        subgroup_recurrent_rms = subgroupRecurrentRms,
        prepare_vision_on_load = prepareVisionOnLoad,
        batch_recurrent = batchRecurrent, buffer_pool_mib = bufferPoolMiB,
        deferred_release_mib = deferredReleaseMiB, image_size = imageSize,
        large_prepare_ms = largePrepareMs, results = records }, new JsonSerializerOptions { WriteIndented = true }));
