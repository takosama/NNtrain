# NNtrain GUI local API

Starting `NNtrain.Gui.exe` also starts a local OpenAI-compatible server in a
separate console window. The GUI sends its own preload and chat requests to
that server over HTTP. The console shows model loading, request settings,
generated text, completion statistics, and errors. Closing the GUI stops its
server and releases the resident model.

The server binds only to `127.0.0.1`. Its chosen port and base URL are printed
in the server console. The API is intended for programs on the same computer;
it has no API-key authentication or remote-network listener.

## Start the GUI

```powershell
.\NNtrain.Gui.exe --model "C:\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" `
  --lora "C:\checkpoints\tuku-qwen35\adapter.bin" `
  --temperature 0.6 --top_p 0.95 --top_k 20 --maxtokens 512 `
  --stream on --think on
```

`--lora` is optional. It accepts an NNtrain `adapter.bin`; exported GGUF LoRA
adapters are not loadable by this inference path. The GUI also accepts the
legacy misspelling `--tempreture`. The model and LoRA can be changed in the
GUI. A parent-directory path component (`../` or `..\`) is rejected in
startup options and API requests. Absolute paths and ordinary paths within
the current directory are accepted.

Use **プリロード** to load the selected model and LoRA before chatting. The
server keeps weights and reusable inference state resident. Five minutes
after the last completed preload or generation, it releases the model and GPU
memory. The next request loads it again.

## API

`GET /v1/models` lists available GGUF models. `POST /v1/chat/completions`
accepts standard text `messages`, `model`, `temperature`, `top_p`,
`max_tokens` (or `max_completion_tokens`), and `stream`. Streaming replies are
Server-Sent Events containing `chat.completion.chunk` objects and a final
`data: [DONE]` event. Nonstreaming replies are `chat.completion` objects.

The local extensions `top_k`, `think`, `lora`, `devices`, and
`messages[].assistant_prefix` control NNtrain inference. `think` selects the
Qwen chat-template mode. The GUI keeps prior assistant answers in `messages`
without previous thinking text; `assistant_prefix` preserves the token prefix
needed for KV reuse. The response text may include model-generated
`<think>` / `</think>` markers. The GUI shows these in its collapsible
thinking panel.

For streamed GUI requests, `prime_history` is enabled. After the final SSE
event, the server processes the visible answer without its thinking text and
keeps that conversation prefix in the GPU KV/recurrent cache. The GUI can show
the completed answer while this work continues. If the next request arrives
before priming finishes, it waits for the same work; no prior thinking text is
added back to the prompt. External API clients can opt in with
`"prime_history": true` on streamed requests. To reuse that state, the next
request must include the visible answer as an assistant message with
`assistant_prefix` set to `"<think>\n</think>\n"` when `think` was on, or
`"<think>\n\n</think>\n\n"` when it was off. Ordinary OpenAI clients that omit
this extension will prefill the conversation again.

```powershell
$body = @{
  model = 'C:\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf'
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
Invoke-RestMethod 'http://127.0.0.1:<port>/v1/chat/completions' `
  -Method Post -ContentType 'application/json' -Body $body
```

The GUI also uses `POST /internal/load`, `POST /internal/unload`, and
`GET /internal/state` for preload, release, and status. These routes are local
extensions and are not OpenAI API routes. Requests to the GPU are serialized
so a model switch cannot interrupt another request's load-and-generate pair.
