"""Offline official Transformers CPU oracle; no microphone or remote model code."""
import argparse, json, os, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import soundfile as sf
from transformers import AutoModelForRNNT, AutoProcessor

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--wav', required=True)
p.add_argument('--output', required=True)
p.add_argument('--round-fp16', action='store_true')
p.add_argument('--encoder-only', action='store_true')
args = p.parse_args()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
audio, rate = sf.read(args.wav, dtype='float32')
assert rate == 16000 and audio.ndim == 1
if args.encoder_only:
    audio = audio[:9120]
start = time.perf_counter()
processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
model = AutoModelForRNNT.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, dtype=torch.float32).eval()
if args.round_fp16:
    model.half().float()
load_seconds = time.perf_counter() - start
inputs = processor(audio, sampling_rate=rate, return_tensors='pt', language='ja-JP')
feature = inputs['input_features'].detach().contiguous()
out = Path(args.output)
feature.transpose(1, 2).contiguous().numpy().astype('<f4').tofile(out.with_suffix('.features.f32'))
if args.encoder_only:
    past = padding = None
    encoded = []
    with torch.inference_mode():
        for begin, end in ((0, 25), (25, 57)):
            output = model.encoder(input_features=feature[:, begin:end, :], past_key_values=past,
                                   padding_cache=padding, use_cache=True, output_attention_mask=False,
                                   num_lookahead_tokens=3)
            encoded.append(output.last_hidden_state)
            past, padding = output.past_key_values, output.padding_cache
    encoded = torch.cat(encoded, dim=1).contiguous()
    encoded.numpy().astype('<f4').tofile(out.with_suffix('.encoder.f32'))
    out.write_text(json.dumps(dict(shape=list(encoded.shape), precision='FP16 rounded parameters, FP32 operations',
                                   mode='two cached chunks; 25 + 32 valid mel frames')), encoding='utf-8')
    raise SystemExit(0)
start = time.perf_counter()
def chunks():
    offset = 0
    width = 25
    while offset < feature.shape[1]:
        chunk = feature[:, offset:offset+width, :]
        if chunk.shape[1] < width:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, width-chunk.shape[1]))
        yield chunk
        offset += width
        width = 32
inputs.pop('attention_mask', None)
inputs['input_features'] = chunks()
with torch.inference_mode():
    generated = model.generate(**inputs)
recognition_seconds = time.perf_counter()-start
text = processor.batch_decode(generated.sequences, skip_special_tokens=True)
result = dict(text=text, loadSeconds=load_seconds, recognitionSeconds=time.perf_counter()-start,
              audioSeconds=len(audio)/rate, featureShape=list(feature.shape),
              parameterPrecision='FP16 rounded then FP32 operations' if args.round_fp16 else 'original FP32',
              torchVersion=torch.__version__, microphoneOpened=False, llmStarted=False, device='CPU')
result['recognitionSeconds'] = recognition_seconds
result['mode'] = 'cached streaming: first 25 mel frames, subsequent 32, lookahead 3, padded final chunk'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
