# Evaluation

The evaluation harness in this directory supports local and hosted readers. Install its
locked environment before running a benchmark:

```bash
uv sync --frozen
```

## MiniMax readers

Set `MINIMAX_API_KEY` and select either registered model ID. The model context length
must be passed explicitly so retrieval truncation uses the correct limit.

| Model ID | Context window | Input modalities | Thinking |
| --- | ---: | --- | --- |
| `MiniMax-M3` | 1,000,000 | text, image, video | adaptive or disabled |
| `MiniMax-M2.7` | 204,800 | text | always on |

Pricing is in USD per million tokens:

| Model ID | Service tier | Input range | Input | Output | Cache read | Cache write |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `MiniMax-M3` | standard | <= 512k input tokens | $0.30 | $1.20 | $0.06 | - |
| `MiniMax-M3` | standard | > 512k input tokens | $0.60 | $2.40 | $0.12 | - |
| `MiniMax-M3` | priority | <= 512k input tokens | $0.45 | $1.80 | $0.09 | - |
| `MiniMax-M3` | priority | > 512k input tokens | $0.90 | $3.60 | $0.18 | - |
| `MiniMax-M2.7` | standard | all | $0.30 | $1.20 | $0.06 | $0.375 |

```bash
export MINIMAX_API_KEY=your-api-key
uv run python run_bench.py \
  --task simpleqa \
  --model MiniMax-M3 \
  --model-context-length 1000000 \
  --num-examples 1
```

`run_bench.py` uses the global OpenAI-compatible endpoint by default. Set
`MINIMAX_API_BASE=https://api.minimaxi.com/v1` to use the China endpoint.

The official API entry points are:

| Region | OpenAI-compatible Base URL | Anthropic-compatible Base URL |
| --- | --- | --- |
| Global | `https://api.minimax.io/v1` | `https://api.minimax.io/anthropic` |
| China | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` |

For Anthropic SDK integrations, pass the `/anthropic` Base URL unchanged. The SDK
appends `/v1/messages` when creating a message.
