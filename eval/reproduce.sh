#!/bin/bash
# PixelRAG paper Table 1 reproduction — one cell at a time.
# Self-contained: uses this repo's eval/run_bench.py + eval/lib (no external checkout needed).
#
#   bash reproduce.sh <bench> <retrieval>
#     bench     = nq | nqt | sqa | mms | evqa | livevqa
#     retrieval = naive | traf | base | lora
#
# Runs the full pipeline (retrieve -> read -> grade) and prints the score.
# It does NOT compare to the paper and does NOT detect the GPU: run the reader on an
# H100 (see README.md) and the numbers naturally land within ~1pp of the paper.
#
# Env (defaults in [] — see README.md for the serve topology):
#   READER_URL  reader (Qwen3.5-4B, vLLM 0.19.0) OpenAI API base  [http://localhost:8010/v1]
#   BASE_PORT   base pixel search serve   [30088]
#   LORA_PORT   lora pixel search serve   [30096]
#   TEXT_PORT   trafilatura text serve    [30097]
#   NEWS_PORT   news pixel serve (livevqa)[30095]
#   TILES_DIR   local wiki tiles dir; set EMPTY to use serve-returned base64 tiles  [/mnt/data/yichuan/kiwix_tiles]
#   OPENAI_API_KEY / OPENAI_BASE_URL  for the LLM-judge grader (auto-loaded from ../.env)
set -euo pipefail
cd "$(dirname "$0")"

BENCH="${1:?Usage: reproduce.sh <nq|nqt|sqa|mms|evqa|livevqa> <naive|traf|base|lora>}"
RETR="${2:?Usage: reproduce.sh <bench> <naive|traf|base|lora>}"

READER_URL="${READER_URL:-http://localhost:8010/v1}"
BASE_PORT="${BASE_PORT:-30088}"; LORA_PORT="${LORA_PORT:-30096}"
TEXT_PORT="${TEXT_PORT:-30097}"; NEWS_PORT="${NEWS_PORT:-30095}"
TILES_DIR="${TILES_DIR:-/mnt/data/yichuan/kiwix_tiles}"
TILESFLAG=""; [ -n "$TILES_DIR" ] && TILESFLAG="--tiles-dir $TILES_DIR"   # empty TILES_DIR = read tiles from the serve's base64
PY="$(pwd)/.venv/bin/python"
PIXEL_INSTR="Retrieve images or text relevant to the user's query."
TEXT_INSTR="Retrieve text relevant to the user's query."
mkdir -p eval_output
[ -f ../.env ] && { export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' ../.env | cut -d= -f2-)"; \
                    export OPENAI_BASE_URL="$(grep '^OPENAI_BASE_URL=' ../.env | cut -d= -f2-)"; }

# --- LiveVQA: separate news pipeline (run_livevqa.py) ---------------------
if [ "$BENCH" = livevqa ]; then
  OUT="eval_output/repro_livevqa_${RETR}.jsonl"
  COMMON=(--api-base "$READER_URL" --model Qwen/Qwen3.5-4B --no-think --max-tokens 16
          --livevqa-images /mnt/data/yichuan/livevqa --output "$OUT")
  case "$RETR" in
    naive) "$PY" run_livevqa.py --mode naive "${COMMON[@]}" ;;
    base)  "$PY" run_livevqa.py --mode pixel --pixel-api "http://localhost:${NEWS_PORT}/search" \
             --pages-db /mnt/data/yichuan/news_state.db --tiles-dir /mnt/data/yichuan/news_tiles "${COMMON[@]}" ;;
    *) echo "livevqa supports: naive | base (MCQ exact-match, scored by run_livevqa.py)" >&2; exit 1 ;;
  esac
  exit 0
fi

# --- per-benchmark config (Qwen3.5-4B, rtk=5, rk=3) -----------------------
case "$BENCH" in
  nq)   TASK=nq;               GRADE=nq;               THINK=off; MAXTOK=200;   N=1000; EXTRA="" ;;
  nqt)  TASK=nq_tables;        GRADE=nq_tables;        THINK=off; MAXTOK=200;   N="";   EXTRA="" ;;
  sqa)  TASK=simpleqa;         GRADE=simpleqa;         THINK=off; MAXTOK=200;   N=1000; EXTRA="--nprobe 2000" ;;
  mms)  TASK=mmsearch;         GRADE=mmsearch;         THINK=on;  MAXTOK=16384; N="";   EXTRA="" ;;
  evqa) TASK=encyclopedic_vqa; GRADE=encyclopedic_vqa; THINK=off; MAXTOK=16384; N="";
        EXTRA="--evqa-dataset-filter landmarks --evqa-question-type-filter automatic" ;;
  *) echo "unknown bench: $BENCH" >&2; exit 1 ;;
esac
# MMS naive is the one MMS cell the paper ran no-think / max_tokens=200.
[ "$BENCH" = mms ] && [ "$RETR" = naive ] && { THINK=off; MAXTOK=200; }
N="${NUM:-$N}"   # NUM env overrides example count (handy for a quick smoke test)
NUMFLAG=""; [ -n "$N" ] && NUMFLAG="--num-examples $N"   # empty N = run the whole set
THINKFLAG=""; [ "$THINK" = off ] && THINKFLAG="--no-think"

# --- retrieval condition --------------------------------------------------
case "$RETR" in
  naive) RFLAGS=() ;;
  base)  RFLAGS=(--local-api --local-api-url "http://localhost:${BASE_PORT}/search" --query-instruction "$PIXEL_INSTR") ;;
  lora)  RFLAGS=(--local-api --local-api-url "http://localhost:${LORA_PORT}/search" --query-instruction "$PIXEL_INSTR") ;;
  # --no-query-image: match the paper's text-only text retrieval. See README.md.
  traf)  RFLAGS=(--text-api  --text-api-url  "http://localhost:${TEXT_PORT}/search" --query-instruction "$TEXT_INSTR" --no-query-image) ;;
  *) echo "unknown retrieval: $RETR" >&2; exit 1 ;;
esac

# --- PREFLIGHT: verify required serves are up AND the right index is loaded ----------
# (catches "serve down / wrong port / wrong index" before a silent empty run.)
preflight_fail=0
check_reader() {
  local m; m=$(curl -s --max-time 5 "${READER_URL%/v1}/v1/models" 2>/dev/null | grep -o "Qwen/Qwen3.5-4B" | head -1 || true)
  if [ "$m" = "Qwen/Qwen3.5-4B" ]; then echo "  ok  reader   $READER_URL  (Qwen/Qwen3.5-4B)"; else
    echo "  FAIL reader  $READER_URL not serving Qwen/Qwen3.5-4B"
    echo "       launch on an H100: CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-4B --port 8010  (vLLM 0.19.0)"
    preflight_fail=1; fi
}
# check_search <role> <port> <min_vectors> <launch-index-dir>
check_search() {
  local role=$1 port=$2 minv=$3 idx=$4
  local st; st=$(curl -s --max-time 5 "http://localhost:${port}/status" 2>/dev/null || true)
  local nv; nv=$(echo "$st" | grep -o '"total_vectors":[0-9]*' | grep -o '[0-9]*' || true)
  if [ -n "$nv" ] && [ "$nv" -ge "$minv" ]; then echo "  ok  $role :$port  ($nv vectors)"; else
    echo "  FAIL $role :$port  down or wrong index (got vectors='${nv:-none}', need >=$minv)"
    echo "       launch: pixelrag serve --index-dir $idx --port $port   (+ tunnel if remote)"
    preflight_fail=1; fi
}
echo ">>> [$BENCH/$RETR] preflight:"
check_reader
case "$RETR" in
  base) check_search base-pixel "$BASE_PORT" 28000000 "search_index_normed_v2" ;;
  lora) check_search lora-pixel "$LORA_PORT" 26000000 "search_index_lora_vit_ckpt200_v2" ;;
  traf) check_search traf-text  "$TEXT_PORT" 15000000 "text_search_index_1024_normed" ;;
esac
if [ "$preflight_fail" = 1 ]; then echo ">>> preflight FAILED — bring the serve(s) up (commands above), then re-run." >&2; exit 2; fi

OUT="eval_output/repro_${BENCH}_${RETR}.jsonl"
echo ">>> [$BENCH/$RETR] run_bench: reader=$READER_URL task=$TASK think=$THINK max_tokens=$MAXTOK n=${N:-all}"
# shellcheck disable=SC2086
"$PY" run_bench.py --task "$TASK" --model Qwen/Qwen3.5-4B \
    --api-base "$READER_URL" --api-key dummy $THINKFLAG \
    --retrieval-top-k 5 --reader-top-k 3 $NUMFLAG --max-tokens "$MAXTOK" \
    $TILESFLAG --output "$OUT" --force --max-concurrent 24 \
    $EXTRA "${RFLAGS[@]}"

echo ">>> [$BENCH/$RETR] grading ($GRADE)"
# nq/nqt: the paper's published numbers used the LLM judge (semantic match), not strict
# exact-match — grade with --llm-judge to match. (For strict exact-match, drop the flag.)
JUDGEFLAG=""; case "$GRADE" in nq|nq_tables) JUDGEFLAG="--llm-judge" ;; esac
# shellcheck disable=SC2086
PYTHONPATH=. "$PY" -m lib.grader "$GRADE" "$OUT" $JUDGEFLAG
