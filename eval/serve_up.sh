#!/bin/bash
# Bring up the search serves a reproduction cell needs, downloading the FAISS index
# from Hugging Face first if it is not already on disk. Pairs with reproduce.sh's preflight
# (same port/index manifest). Run this ON a GPU box that will host the serves.
#
#   bash serve_up.sh <role>...        role = base | lora | text | news | reader | all
#   e.g.  bash serve_up.sh base text  bash serve_up.sh all
#
# Env:
#   INDEX_ROOT   where indexes live / get downloaded   [/data/pixelrag/indexes]
#   HF_INDEX_REPO  HF dataset repo holding the indexes  [StarTrail-org/pixelrag-faiss-indexes]
#   GPU          CUDA device for the serves            [0]
#   READER_GPU   CUDA device for the reader (H100)     [0]
# Ports default to the reproduce.sh manifest (override with BASE_PORT/LORA_PORT/TEXT_PORT/NEWS_PORT).
set -uo pipefail
cd "$(dirname "$0")"

INDEX_ROOT="${INDEX_ROOT:-/data/pixelrag/indexes}"
HF_INDEX_REPO="${HF_INDEX_REPO:-StarTrail-org/pixelrag-faiss-indexes}"
GPU="${GPU:-0}"; READER_GPU="${READER_GPU:-0}"
BASE_PORT="${BASE_PORT:-30088}"; LORA_PORT="${LORA_PORT:-30096}"
TEXT_PORT="${TEXT_PORT:-30097}"; NEWS_PORT="${NEWS_PORT:-30095}"
SERVE="pixelrag serve"        # = python -m pixelrag_serve.api
mkdir -p "$INDEX_ROOT"

# role -> index-subdir : port : extra serve args
declare -A IDX=( [base]=search_index_normed_v2 [lora]=search_index_lora_vit_ckpt200_v2
                 [text]=text_search_index_1024_normed [news]=news_image_search_index )
declare -A PORT=( [base]=$BASE_PORT [lora]=$LORA_PORT [text]=$TEXT_PORT [news]=$NEWS_PORT )

fetch() {  # download index dir from HF if missing
  local sub=$1 dir="$INDEX_ROOT/$1"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then echo "  have $dir"; return; fi
  echo "  downloading $sub from $HF_INDEX_REPO ..."
  hf download "$HF_INDEX_REPO" --repo-type dataset --include "$sub/*" --local-dir "$INDEX_ROOT"
}

up_search() {  # role
  local role=$1 sub=${IDX[$1]} port=${PORT[$1]}
  fetch "$sub"
  echo ">>> serve $role on :$port (index $sub, gpu $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU nohup $SERVE --index-dir "$INDEX_ROOT/$sub" --port "$port" \
      > "/tmp/pixelrag_serve_${role}.log" 2>&1 &
  echo "    log: /tmp/pixelrag_serve_${role}.log"
}

up_reader() {
  echo ">>> reader Qwen3.5-4B on :8010 (gpu $READER_GPU, vLLM 0.19.0)"
  CUDA_VISIBLE_DEVICES=$READER_GPU nohup vllm serve Qwen/Qwen3.5-4B --port 8010 \
      --max-model-len 32768 --gpu-memory-utilization 0.85 > /tmp/pixelrag_reader.log 2>&1 &
  echo "    log: /tmp/pixelrag_reader.log"
}

[ $# -eq 0 ] && { echo "usage: serve_up.sh <base|lora|text|news|reader|all>..."; exit 1; }
for r in "$@"; do
  case "$r" in
    base|lora|text|news) up_search "$r" ;;
    reader) up_reader ;;
    all) up_search base; up_search lora; up_search text; up_search news; up_reader ;;
    *) echo "unknown role: $r" >&2 ;;
  esac
done
echo ">>> launched. Wait for load, then verify with: bash reproduce.sh <bench> <retrieval>  (its preflight checks /status)."
