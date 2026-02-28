# Lab Journal Summary

**Updated: 2026-02-27 (after 5 experiments)**

## Project

**Title (working):** Agent RL for Diffusion Language Models: Multi-Turn Tool-Calling Training via Segment-Aware Policy Optimization

**Target:** Workshop paper (4-6 pages), NeurIPS/ICML 2026 workshop

**Repository:** https://github.com/XiangJinyu/dLLM-RL (branch: `feature/agent-rl-training`)

## Current Results Table

| Exp | Method | GSM8K Acc | Tool Rate | Delta vs Baseline |
|-----|--------|-----------|-----------|-------------------|
| 002 | Zero-shot (no RL) | 0.0850 | 0.000 | -- |
| 003 (Exp004) | Agent RL + TraceRL + segment-aware | 0.1400 | 0.035 | +0.055 |
| 004 (Exp003) | Standard RL (no tools) | **0.1800** | 0.000 | **+0.095** |
| 006 | Agent RL + TraceRL + NO segment mask | **0.2000** | 0.005 | **+0.115** |
| 005 | Agent RL + random masking + segment | _running_ | -- | -- |

## Key Findings

1. **All RL methods improve over zero-shot**: 8.5% -> 14-20%, confirming that TraceRL works for diffusion LLMs.

2. **Tool access does NOT help on GSM8K with base model**: Standard RL (18%) > Agent RL with segment masking (14%). Calculator can't help with the reasoning bottleneck. The model needs to SET UP the computation, not just EXECUTE it.

3. **Segment-aware masking HURTS on this benchmark**: No-segment (20%) > segment-aware (14%). Training on observation tokens may actually be helpful -- the model learns from tool output patterns. This contradicts our initial hypothesis.

4. **Tool call rate DECREASES with training**: Starts at 17% and drops to 3.5%. The model learns NOT to use tools because correct answers without tools get higher reward.

5. **Training is stable**: All experiments complete without crashes, losses are bounded, no divergence.

## Reframing for Paper

The original hypothesis (segment-aware masking is essential) is NOT supported by data. Need to reframe:

**Option A: Honest negative result paper**
- "Agent RL for dLLMs works, but tool access doesn't help on computation-focused benchmarks where reasoning is the bottleneck"
- Contribution: first to show the pipeline works, identify when tools help vs. hurt

**Option B: Find a better benchmark**
- Search-based QA (HotpotQA, TriviaQA) where tools genuinely help
- Code generation with interpreter
- Multi-hop reasoning where intermediate tool calls provide new information

**Option C: Reframe contribution as infrastructure**
- The agent RL framework itself (segment masking, cross-turn step_map, tool protocol) is the contribution
- GSM8K results show it works, even if tools aren't the bottleneck here

## Open Questions

1. Would an instruction-tuned model (LLaDA-Instruct if available) show better tool usage?
2. Would a search-based QA benchmark show clearer tool benefit?
3. Is the no-segment result because observation tokens are actually informative (they tell the model what "tool output" looks like)?

## Infrastructure Notes

- GPU: A100-SXM4-80GB ($1.39/hr RunPod Community)
- Stack: PyTorch 2.2.0 + CUDA 12.1 + transformers 4.45.2
- batch_size=2, max 100 sub-samples/step, ~2 min/step
- Pod ID: d328giwzr4d5g6
