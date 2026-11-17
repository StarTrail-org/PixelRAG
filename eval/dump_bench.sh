#!/bin/bash
# Export the frozen-retrieval PixelRAG reader benchmark for one task.
#
#   bash dump_bench.sh <nq|nqt|sqa|mms|evqa> <out_root>
#
# Uses the same example sets, query instruction, query images and nprobe as the LoRA
# cells of reproduce.sh, against a serve holding search_index_lora_vit_ckpt200_v2, and
# keeps the top 5 tiles per question (the paper's reader reads the top 3).
#   LORA_URL   lora pixel search serve  [http://localhost:30096/search]
#   TILES_DIR  wiki tiles dir; EVQA landmark query images are found beside it
#              [/mnt/data/yichuan/kiwix_tiles]
#   NUM        overrides the example count (smoke tests)
set -euo pipefail
cd "$(dirname "$0")"

BENCH="${1:?Usage: dump_bench.sh <nq|nqt|sqa|mms|evqa> <out_root>}"
OUT_ROOT="${2:?Usage: dump_bench.sh <bench> <out_root>}"
LORA_URL="${LORA_URL:-http://localhost:30096/search}"
TILES_DIR="${TILES_DIR:-/mnt/data/yichuan/kiwix_tiles}"

case "$BENCH" in
  nq)   TASK=nq;               N=1000; EXTRA=() ;;
  nqt)  TASK=nq_tables;        N="";   EXTRA=() ;;
  sqa)  TASK=simpleqa;         N=1000; EXTRA=(--nprobe 2000) ;;
  mms)  TASK=mmsearch;         N="";   EXTRA=() ;;
  evqa) TASK=encyclopedic_vqa; N="";
        EXTRA=(--evqa-dataset-filter landmarks --evqa-question-type-filter automatic) ;;
  *) echo "unknown bench: $BENCH" >&2; exit 1 ;;
esac
N="${NUM:-$N}"
NUMFLAG=(); [ -n "$N" ] && NUMFLAG=(--num-examples "$N")

OUT="$OUT_ROOT/$BENCH"
mkdir -p "$OUT"
# The reader is never called in dump mode; --model/--api-base only satisfy the parser.
.venv/bin/python run_bench.py --task "$TASK" --model Qwen/Qwen3.5-4B \
    --api-base http://localhost:1/v1 --api-key dummy --no-think \
    --retrieval-top-k 5 --reader-top-k 3 "${NUMFLAG[@]}" \
    --tiles-dir "$TILES_DIR" --output "$OUT/unused_reader_output.jsonl" --force \
    --local-api --local-api-url "$LORA_URL" \
    --query-instruction "Retrieve images or text relevant to the user's query." \
    --dump-retrieval "$OUT" "${EXTRA[@]}"
rm -f "$OUT/unused_reader_output.jsonl"
