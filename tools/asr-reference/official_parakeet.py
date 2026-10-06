"""Official NeMo CTC CPU oracle. Uses safe weights-only loading, never ambient recording."""
import argparse, json, os, tarfile, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import soundfile as sf
from omegaconf import OmegaConf
from safetensors.torch import load_file
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--wav', required=True)
p.add_argument('--output', required=True)
p.add_argument('--round-fp16', action='store_true')
p.add_argument('--prefix-samples', type=int)
p.add_argument('--trace')
args = p.parse_args()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
directory = Path(args.model).resolve()
config = OmegaConf.load(directory/'model_config.yaml')
config.train_ds = config.validation_ds = config.test_ds = None
config.tokenizer.dir = str(directory)
config.tokenizer.model_path = str(directory/'tokenizer.model')
config.tokenizer.vocab_path = str(directory/'vocab.txt')
config.tokenizer.spe_tokenizer_vocab = str(directory/'tokenizer.vocab')
clock = time.perf_counter()
model = EncDecHybridRNNTCTCBPEModel(cfg=config, trainer=None).cpu().eval()
if args.round_fp16:
    frontend_buffers = {key: value.clone() for key,value in model.preprocessor.state_dict().items()}
    weights = load_file(directory/'model.safetensors', device='cpu')
else:
    with tarfile.open(directory/'parakeet-tdt_ctc-0.6b-ja.nemo', 'r:') as archive:
        member = next(x for x in archive if x.name.lstrip('./') == 'model_weights.ckpt')
        assert member.isfile() and member.size < 3_000_000_000
        with archive.extractfile(member) as data:
            weights = torch.load(data, map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(weights, strict=False)
assert not unexpected, unexpected
assert all(x.endswith('num_batches_tracked') for x in missing), missing
del weights
if args.round_fp16:
    # C# constructs the signal frontend in FP32; rounded filter/window archive buffers
    # are unused. Keep this oracle's frontend in FP32 too, while BN buffers stay rounded.
    model.preprocessor.load_state_dict(frontend_buffers)
model.preprocessor.featurizer.dither = 0
if args.trace:
    trace = Path(args.trace)
    trace.mkdir(parents=True, exist_ok=True)
    def capture(name):
        def hook(module, inputs, output):
            values = output[0] if isinstance(output, tuple) else output
            values.detach().contiguous().numpy().astype('<f4').tofile(trace/(name+'.f32'))
        return hook
    model.encoder.pre_encode.register_forward_hook(capture('pre_encode'))
    for index, layer in enumerate(model.encoder.layers):
        layer.register_forward_hook(capture(f'layer-{index}'))
load_seconds = time.perf_counter()-clock
audio, rate = sf.read(args.wav, dtype='float32')
assert rate == 16000 and audio.ndim == 1
if args.prefix_samples:
    audio = audio[:args.prefix_samples]
signal = torch.from_numpy(audio).unsqueeze(0)
length = torch.tensor([len(audio)], dtype=torch.long)
clock = time.perf_counter()
with torch.inference_mode():
    features, feature_lengths = model.preprocessor(input_signal=signal, length=length)
    encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
    log_probs = model.ctc_decoder(encoder_output=encoded)
    ids = log_probs[0, :int(encoded_lengths[0])].argmax(-1).tolist()
blank = log_probs.shape[-1]-1
tokens, previous = [], None
for token in ids:
    if token != previous and token != blank:
        tokens.append(token)
    previous = token
text = model.tokenizer.ids_to_text(tokens)
out = Path(args.output)
features[0, :, :int(feature_lengths[0])].contiguous().numpy().astype('<f4').tofile(out.with_suffix('.features.f32'))
encoded[0, :, :int(encoded_lengths[0])].transpose(0,1).contiguous().numpy().astype('<f4').tofile(out.with_suffix('.encoder.f32'))
result = dict(text=text, loadSeconds=load_seconds, recognitionSeconds=time.perf_counter()-clock,
              audioSeconds=len(audio)/rate, featureShape=[80,int(feature_lengths[0])],
              encoderShape=[int(encoded_lengths[0]), encoded.shape[1]], decoder='official CTC greedy',
              precision='FP16 rounded parameters, FP32 operations' if args.round_fp16 else 'original FP32',
              missingBatchCounters=missing, device='CPU', microphoneOpened=False, llmStarted=False)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
