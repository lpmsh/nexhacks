# LongBench v2 Evaluation Runbook

This runbook aligns with the LongBench v2 paper and the bear-1 benchmark methodology described by The Token Company. It is designed to make our Track 1 submission reproducible and comparable.

## Goals
- Compare compressed vs uncompressed accuracy on LongBench v2 MCQ tasks.
- Report token reduction and latency proxies.
- Match bear-1 settings where possible: multiple runs, deterministic decoding, and consistent token counting.

## Dataset
LongBench v2 format (fields: context, question, choice_A-D, answer, domain, difficulty, length).
Source references:
- Paper: https://arxiv.org/abs/2412.15204
- Project page: https://longbench2.github.io
- HF dataset: THUDM/LongBench-v2

## Token counting
Bear-1 reports use tiktoken (GPT-4o-mini encoding).
Use `get_token_counter()` for tiktoken when installed; otherwise fallback to word tokens.

## Reproducible evaluation checklist
1) Subset selection
   - Filter to <= 100k tokens (tiktoken) to match the bear-1 benchmark subset.
   - Sample 230 questions with a fixed seed (e.g., 13).
2) Compression settings
   - Use a fixed `target_ratio` or `token_budget`.
   - Record all toggle values in the report.
3) Model inference
   - Use GPT-4o-mini for comparability or your target model.
   - Temperature = 0, deterministic output format: single letter.
4) Runs
   - 50 runs per configuration if possible.
   - Compute mean accuracy, std dev, and significance vs baseline.
5) Report
   - Accuracy, token reduction, cost proxy, and ablation deltas.
   - Include at least one error analysis slice (domain, difficulty, length).

## Example usage (compression + prompts)
```python
from cosmos.longbench_eval import get_token_counter, load_longbench, LongBenchRunner, write_jsonl

token_counter = get_token_counter(model_name="gpt-4o-mini")
samples = load_longbench(
    path="data/longbench_v2.jsonl",
    limit=230,
    seed=13,
    shuffle=True,
    max_context_tokens=100000,
    token_counter=token_counter,
)

runner = LongBenchRunner()
compressed = runner.compress_samples(samples, target_ratio=0.4, seed=13)
compressed_prompts = runner.build_compressed_prompts(compressed)
baseline_prompts = runner.build_prompts(samples)

write_jsonl("out/longbench_baseline_prompts.jsonl", baseline_prompts)
write_jsonl("out/longbench_compressed_prompts.jsonl", compressed_prompts)
```

## Scripted end-to-end runner
Use the CLI to generate prompts and optionally run model evaluation via an OpenAI-compatible endpoint.
```bash
python scripts/run_longbench.py \
  --data data/longbench_v2.jsonl \
  --limit 230 \
  --max-context-tokens 100000 \
  --target-ratio 0.4 \
  --mode both \
  --out-dir out \
  --eval
```

Environment variables (required for `--eval`):
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: gpt-4o-mini)
- `OPENAI_BASE_URL` (default: https://api.openai.com/v1)

## Optional LLM evaluation hook
Implement a predictor that takes a prompt and returns a single letter (A/B/C/D).
```python
def predictor(prompt: str) -> str:
    # Call your model here (OpenAI, local, etc.)
    return "A"

baseline_eval = runner.evaluate(baseline_prompts, predictor)
compressed_eval = runner.evaluate(compressed_prompts, predictor)
```

## Recommended config sweeps
- Conservative: target_ratio = 0.75
- Balanced: target_ratio = 0.5
- Aggressive: target_ratio = 0.35

## Track 1 framing
- Novelty: option-aware scoring + hierarchical windowing + redundancy control.
- Effectiveness: show accuracy vs token reduction curves and significant wins.
- Robustness: domain and length stratified reporting.
