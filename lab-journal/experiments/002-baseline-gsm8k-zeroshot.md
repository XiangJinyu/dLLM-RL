---
id: "002"
slug: baseline-gsm8k-zeroshot
type: exploration
status: completed
created: 2026-02-27
concluded: 2026-02-27
depends_on: []
conclusion_type: quantitative
conclusion: "LLaDA-8B-Base zero-shot on GSM8K: 8.5% accuracy. Low baseline expected for non-instruction-tuned diffusion LM."
tags: [baseline, gsm8k, llada, zero-shot]
commit: ""
---

# 002: Baseline GSM8K Zero-Shot Evaluation

## Question

What is LLaDA-8B-Base's zero-shot accuracy on GSM8K without any RL training or tool access?

## Method

- Model: LLaDA-8B-Base (no SFT/RLHF)
- Eval: 200 samples from GSM8K test set
- Generation: 128 tokens, block_size=32, temperature=0.8
- No tools, no multi-turn
- Script: `scripts/run_experiment.py --mode baseline_eval`

## Evidence

| Metric | Value |
|--------|-------|
| Accuracy | 0.0850 |
| Avg response length | 128.0 |
| Tool call rate | 0.0000 |

## Interpretation

8.5% is the floor. This is expected: LLaDA-8B-Base is a pre-trained model without instruction tuning. It struggles with multi-step reasoning required by GSM8K. Any RL training that exceeds this is meaningful improvement.

For reference, autoregressive base models of similar size (e.g., Llama-2-7B-Base) score ~5-15% on GSM8K zero-shot, so 8.5% is in the expected range.

## Next

- 003: Standard single-turn RL (no tools) to see how much vanilla TraceRL helps
- 004: Agent RL with calculator tool to see if tool access provides additional gains
