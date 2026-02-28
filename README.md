# Agent RL for Diffusion Language Models

Multi-turn tool-calling agent training for diffusion language models, built on top of the [dLLM-RL (TraceRL)](https://github.com/Gen-Verse/dLLM-RL) framework.

**[[Paper]](paper/Agent_RL_for_Diffusion_LLMs.pdf)**

## Overview

All existing multi-turn agent RL methods (RAGEN, SimpleTIR, RLFactory, etc.) assume autoregressive LLMs. This project extends agent RL training to **diffusion language models** (dLLMs), addressing three key challenges:

1. **Segment-aware masking** -- distinguishes model-generated tokens from environment-injected observations in multi-turn trajectories
2. **Cross-turn step-map composition** -- extends TraceRL's trajectory-aware credit assignment across interaction turns
3. **Turn-by-turn diffusion generation** -- each turn runs a separate denoising process, with tool observations injected as frozen context

## Results (GSM8K, LLaDA-8B-Base)

| Method | Masking | Segment-Aware | Accuracy | Tool Rate |
|--------|---------|:---:|:---:|:---:|
| Zero-shot (no RL) | -- | -- | 8.5% | 0.0% |
| Standard RL (no tools) | TraceRL | N/A | 18.0% | 0.0% |
| Agent RL + tools | TraceRL | Yes | 14.0% | 3.5% |
| Agent RL + tools | Random | Yes | 12.0% | 7.0% |
| Agent RL + tools | TraceRL | No | **20.0%** | 0.5% |

Key findings:
- RL post-training consistently improves dLLMs on math reasoning (+5.5 to +11.5pp over zero-shot)
- TraceRL outperforms random masking in the agent setting (14.0% vs 12.0%)
- Training on observation tokens (without segment masking) yields the strongest agent RL result
- Tool access benefit depends on whether the task bottleneck aligns with what tools provide

## Project Structure

```
agent/                     # Agent RL module
  base_tool.py             # BaseTool abstraction
  base_env.py              # BaseToolEnv (multi-turn interaction loop)
  tools/                   # Built-in tools (calculator, search, python)
  envs/                    # Example environments (math, search QA)
sample/
  agent_llada_rl_rollout.py  # Multi-turn diffusion rollout
reward/
  rl_agent_reward.py       # Trajectory + turn-level reward computation
train/
  rl_agent_llada.py        # Segment-aware PPO training
rl_agent.py                # Top-level orchestrator
configs/
  rl_agent_llada.yaml      # Math agent config
  rl_agent_llada_search.yaml # Search QA agent config
scripts/
  run_experiment.py        # Unified experiment script
paper/                     # Workshop paper + figures
```

## Quick Start

```bash
# Install dependencies
pip install omegaconf termcolor jinja2 accelerate scipy einops "transformers>=4.38,<4.46"

# Run agent RL training (requires GPU)
python rl_agent.py config=configs/rl_agent_llada.yaml

# Or use the experiment script
python scripts/run_experiment.py \
  --mode agent_rl \
  --masking_method TraceRL \
  --train_steps 30 \
  --output_dir results/my_experiment
```

### Experiment Modes

| Mode | Description |
|------|-------------|
| `baseline_eval` | Zero-shot evaluation (no training) |
| `standard_rl` | Single-turn RL, no tools |
| `agent_rl` | Multi-turn agent RL with tools + segment-aware masking |
| `agent_rl_no_segment` | Agent RL without segment masking (ablation) |

## Adding Custom Tools

```python
from agent.base_tool import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Description for the model"
    parameters = {"query": {"type": "string", "description": "..."}}

    def execute(self, query="", **kwargs):
        result = do_something(query)
        return ToolResult(tool_name=self.name, success=True, output=result)
```

Then add it to your config:
```yaml
agent:
  tools:
    - class: "path.to.MyTool"
```

## Environment Notes

- **PyTorch 2.2.0 + CUDA 12.1** is the tested stable stack for LLaDA
- `transformers` must be `< 4.46` (newer versions have compatibility issues)
- Training batch_size <= 2 on A100 80GB; max ~100 TraceRL sub-samples per step

## Acknowledgement

This project builds on:
- [dLLM-RL / TraceRL](https://github.com/Gen-Verse/dLLM-RL) -- the base RL framework for diffusion language models
- [LLaDA](https://github.com/ML-GSAI/LLaDA) -- the masked diffusion language model used in experiments

## Citation

```bibtex
@article{agent-rl-dllm-2026,
  title={Agent RL for Diffusion Language Models: Multi-Turn Tool-Calling via Segment-Aware Policy Optimization},
  year={2026}
}
```
