---
id: "001"
slug: preliminary-agent-rl
type: exploration
status: completed
created: 2026-02-27
concluded: 2026-02-27
depends_on: []
conclusion_type: insight
conclusion: "Agent RL pipeline for diffusion LLMs is feasible. Tool call rate improves with training, but accuracy gains require SFT model and more data."
tags: [agent-rl, llada, baseline, feasibility]
commit: 3aa911e
---

# 001: Preliminary Agent RL Feasibility Test

## Question

Can we train a diffusion language model (LLaDA-8B) to use tools via multi-turn RL? Is the segment-aware masking + cross-turn step_map pipeline mechanically sound?

## Method

- Model: LLaDA-8B-Base (no instruction tuning)
- Tool: Calculator
- Training: 5 steps, 6 tasks/step, 4 samples/task
- Method: TraceRL with segment-aware masking (only model-generated tokens trained)
- Eval: 20 arithmetic questions
- GPU: NVIDIA A100-SXM4-80GB
- Stack: PyTorch 2.2.0, CUDA 12.1, transformers 4.45.2
- Script: `run_agent_experiment.py`

Config:
```
lr=5e-6, eps=0.2, beta=0.01, shrink=8, temperature=0.8
max_turns=2, max_tokens_per_turn=128, block_size=32
```

## Evidence

| Metric | Baseline | After 5 steps | Delta |
|--------|----------|---------------|-------|
| Accuracy | 0.5500 | 0.5000 | -0.05 |
| Avg length | 134.4 | 128.0 | -6.4 |
| Tool call rate | 0.9000 | 1.0000 | +0.10 |

Training log:
- Step 1: rollout_acc=0.833, kept=12/24, samples=96, loss=0.000
- Step 2: rollout_acc=0.917, kept=8/24, samples=64, loss=-0.000
- Step 3: rollout_acc=0.667, kept=20/24, samples=100, loss=-0.112
- Step 4: rollout_acc=1.000, kept=0/24, loss=0 (skipped)
- Step 5: rollout_acc=0.917, kept=8/24, samples=64, loss=-0.000

GPU peak memory: 32.3 GB (training), 18.6 GB (inference)

## Interpretation

1. **Pipeline works end-to-end.** All components (multi-turn diffusion generation, tool call parsing, execution, observation injection, segment-aware masking, TraceRL PPO) function correctly on GPU.

2. **Tool call rate improved.** 90% -> 100% shows RL signal reaches the model and increases tool usage behavior.

3. **Accuracy flat/slightly down.** Expected for three reasons:
   - Base model (no SFT) has shallow understanding of tool-call protocol
   - Training data too small (only ~300 effective sub-samples over 5 steps)
   - Tasks too easy (most >80% accuracy, filtered out by difficulty filter)
   - Eval set too small (20 questions, each worth 5%)

4. **OOM at batch=4.** Had to limit to batch=2 and cap at 100 TraceRL sub-samples per step.

5. **Key environment finding:** PyTorch 2.4+ causes segfault with LLaDA. Must use PyTorch 2.2.0 + CUDA 12.1 + transformers < 4.46.

## Next

- 002: Use GSM8K as training data (harder, more diverse)
- 003: Use MATH500 as eval set (standard benchmark)
- Run 30+ training steps instead of 5
- Compare TraceRL vs random masking vs coupled
- Ablation: remove segment-aware masking to validate its importance
- Consider using LLaDA-8B-Instruct if available, or doing SFT cold-start
