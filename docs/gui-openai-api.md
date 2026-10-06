# NNtrain GUI local API

The Windows GUI starts a separate loopback API server and sends preload, image preparation and chat requests to it. The console shows loading and inference progress; closing the GUI stops its own server. The current GUI/server inference route supports compatible `general.architecture=qwen35` GGUF models.

## Authentication and startup

The server listens on `127.0.0.1` only. All routes, including health and internal routes, require session-specific `Authorization: Bearer <token>`. The GUI creates a random token and passes it privately to its child server. A manually started server prints its own session token when `NNTRAIN_GUI_SERVER_TOKEN` has not been supplied. Do not publish the token.

```powershell
.\NNtrain.Gui.exe --server --port 8000
```

For GUI startup and publishing, see the [GUI guide](../NNtrain.Gui/README.md). GUI startup accepts `--model`, optional `--lora`, `--temperature`, `--top_p`, `--top_k`, `--maxtokens`, `--stream on|off` and `--think on|off`. LoRA may be an NNtrain `.bin` or an NNtrain-exported compatible Qwen3.5 F32 GGUF adapter; training resume requires `.bin`. Bonsai PQ2_0 / PTQ1_0 LoRA is unsupported.

Requests also check the local Host, peer and browser Origin. Model paths reject parent traversal and network paths. POST routes require JSON. Admission is bounded to four requests, with overload returning 429 before body parsing. The body limit is 40 MiB and JSON parsing has a 15-second deadline. These checks do not make this a remote service.

## Chat example

Use the token printed by your manually started server. Replace the model path and token below with your own values.

```powershell
$headers = @{ Authorization = 'Bearer <session-token>' }
$body = @{
  model = 'C:\models\base.gguf'
  messages = @(@{ role = 'user'; content = 'こんにちは' })
  max_tokens = 64
  temperature = 0.6
  top_p = 0.95
  top_k = 20
  stream = $false
  think = $false
  lora = $null
  devices = @(0, 1)
} | ConvertTo-Json -Depth 6
Invoke-RestMethod 'http://127.0.0.1:8000/v1/chat/completions' `
  -Headers $headers -Method Post -ContentType 'application/json' -Body $body
```

`stream: true` returns Server-Sent Events with `chat.completion.chunk` objects and a final `data: [DONE]`. Nonstreaming requests return a `chat.completion` object. `max_completion_tokens` is accepted as an alternative to `max_tokens`. This is a local implementation of selected OpenAI-compatible routes, not a complete implementation of the OpenAI API.

## Routes and reusable state

| Route | Purpose |
| --- | --- |
| `GET /health` | Server health |
| `GET /v1/models` | Available GGUF models |
| `POST /v1/chat/completions` | Chat and generation |
| `POST /internal/load` | Preload model, optional adapter / image model |
| `POST /internal/prepare-image` | Image preparation |
| `POST /internal/unload` | Release model |
| `GET /internal/state` | Load and inference state |
| `POST /internal/shutdown` | Stop this server; requires valid JSON and authentication |

Internal routes are NNtrain extensions. GPU requests are serialized so a model switch cannot interrupt a load-and-generate pair. After five minutes without a completed preload or generation the server releases its model; a later request reloads it.

Extensions `top_k`, `think`, `lora`, `devices` and `messages[].assistant_prefix` select local inference settings. The GUI keeps prior visible answers without thinking text. Streamed GUI requests enable `prime_history`: after the final event, the server prepares the visible answer prefix for later KV / recurrent-state reuse. A subsequent request waits for this preparation if needed.

External streamed clients may opt in with `prime_history: true`. The next assistant message must preserve the visible answer and use `assistant_prefix: "<think>\n</think>\n"` when Thinking was on, or `"<think>\n\n</think>\n\n"` when it was off. Clients that omit the extension prefill the conversation again. Prefix mismatches and cancellations safely invalidate reuse. The response may contain model-produced thinking delimiters; clients decide how to display them.

For image and ASR usage, follow the [GUI guide](../NNtrain.Gui/README.md). ASR recognition runs locally in the GUI/Core path; it is not a cloud transcription route. Its edited final text enters the usual chat request.
