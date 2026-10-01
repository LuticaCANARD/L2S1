set -eu
PI_WORK="$HOME/l2s1-pi-20260927"
cd "$PI_WORK/source"
"$PI_WORK/venv/bin/python" -m unittest discover -s training/tests -v > "$PI_WORK/runs/package-tests.log" 2>&1
"$PI_WORK/source/target/release/examples/evaluate_jsonl" \
  --model "$PI_WORK/models/gemma-3-1b-it-Q8_0.gguf" \
  --input "$PI_WORK/jev-demo-data/test-requests.jsonl" \
  --output "$PI_WORK/runs/jev-demo-predictions.jsonl" \
  --context 2048 --batch 256 --threads 4 --execution-mode fresh \
  --prompt-layout legacy --prompt-detail minimal --request-batch-size 1 --warmup \
  > "$PI_WORK/runs/jev-demo-native.log" 2>&1
"$PI_WORK/venv/bin/l2s1-train" report \
  --data "$PI_WORK/jev-demo-data" \
  --predictions "$PI_WORK/runs/jev-demo-predictions.jsonl" \
  --output "$PI_WORK/runs/jev-demo-report"
