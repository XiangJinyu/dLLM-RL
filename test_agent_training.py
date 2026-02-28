"""
End-to-end test for agent RL training pipeline.
Tests the full loop: rollout -> reward -> training data prep -> forward/backward.
"""

import sys, os, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/workspace/dLLM-RL")
sys.path.insert(0, "sample")

print("=" * 60)
print("TEST: Agent RL Training Pipeline (Mini)")
print("=" * 60)

# ── 1. Generate some agent trajectories ──────────────────────
print("\n[1] Generating agent trajectories...")
from agent.tools.calculator_tool import CalculatorTool
from agent.envs.math_tool_env import MathToolEnv
from agent.base_env import (
    Trajectory,
    Turn,
    TOOL_CALL_START,
    TOOL_CALL_END,
    FINAL_ANSWER_START,
    FINAL_ANSWER_END,
)
from llada_rl_rollout import generate_with_prefix_cache, denoise_step_map
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM

model_path = "/workspace/models/LLaDA-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = (
    LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    .to("cuda")
    .eval()
)
mask_id = tokenizer.encode("<|mdm_mask|>")[0]
pad_id = tokenizer.encode("<|endoftext|>")[0]
print(f"   Model loaded, GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

env = MathToolEnv(tools=[CalculatorTool()], max_turns=3)

# Generate 4 trajectories for 2 tasks (2 per task)
tasks = [
    {"question": "What is 17 * 23 + 45?", "ground_truth_answer": "436"},
    {"question": "What is 2^10?", "ground_truth_answer": "1024"},
]

all_data = []
for task in tasks:
    system_prompt = env.get_system_prompt({})
    prompt = (
        "<|startoftext|><|start_header_id|>system<|end_header_id|>\n"
        + system_prompt
        + "<|eot_id|>"
        + "<|startoftext|><|start_header_id|>user<|end_header_id|>\n"
        + task["question"]
        + "<|eot_id|>"
        + "<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    task_outputs = {
        "question": task["question"],
        "ground_truth_answer": task["ground_truth_answer"],
        "prompt": prompt,
        "full_output": [],
        "step_map": [],
        "extracted_output": [],
        "response_length": [],
        "token_segments": [],
        "turns": [],
        "num_turns": [],
    }

    for k in range(2):  # 2 samples per task
        enc = tokenizer([prompt], return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        context_len = input_ids.shape[1]
        gen_length = 128
        block_size = 32
        gen_length = (gen_length // block_size) * block_size

        out = generate_with_prefix_cache(
            model,
            input_ids,
            steps=64,
            gen_length=gen_length,
            block_length=block_size,
            temperature=0.8,
            target="confidence",
            mask_id=mask_id,
            further_horizon=64,
            use_cache=True,
            unmask_threshold=None,
        )

        gen_ids = out.sequences[0, context_len:].tolist()
        gen_text = tokenizer.decode(
            gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        gen_text_clean = (
            gen_text.replace(tokenizer.pad_token or "", "")
            .replace("<|mdm_mask|>", "")
            .strip()
        )

        turn_step_map = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
        turn_step_map = turn_step_map[context_len:].tolist()

        token_segments = [(context_len, context_len + len(gen_ids), True)]

        final_answer = env.parse_final_answer(gen_text_clean)
        if final_answer is None:
            final_answer = gen_text_clean[:50]

        task_outputs["full_output"].append(gen_text_clean)
        task_outputs["step_map"].append(turn_step_map)
        task_outputs["extracted_output"].append(final_answer)
        task_outputs["response_length"].append(len(gen_ids))
        task_outputs["token_segments"].append(token_segments)
        task_outputs["turns"].append(
            [
                {
                    "role": "assistant",
                    "content": gen_text_clean,
                    "is_model_generated": True,
                    "tool_calls": [],
                    "tool_results": [],
                }
            ]
        )
        task_outputs["num_turns"].append(1)

        del out
        torch.cuda.empty_cache()

    all_data.append(task_outputs)

print(f"   Generated {sum(len(d['full_output']) for d in all_data)} trajectories")

# ── 2. Compute rewards ───────────────────────────────────────
print("\n[2] Computing rewards...")
from reward.rl_agent_reward import z_score_normalize

for item in all_data:
    item["correctness"] = []
    for j in range(len(item["extracted_output"])):
        traj = Trajectory(
            prompt=item["prompt"],
            final_answer=item["extracted_output"][j],
            ground_truth=item["ground_truth_answer"],
        )
        traj.turns = [
            Turn(
                role="assistant",
                content=item["extracted_output"][j],
                is_model_generated=True,
            )
        ]
        traj = env.compute_reward(traj)
        item["correctness"].append(traj.trajectory_reward)
    item["rewards"] = z_score_normalize(item["correctness"])

total_correct = sum(sum(d["correctness"]) for d in all_data)
print(f"   Correctness: {total_correct}/{sum(len(d['correctness']) for d in all_data)}")
print(f"   Rewards: {[d['rewards'] for d in all_data]}")

# ── 3. Prepare training data ─────────────────────────────────
print("\n[3] Preparing training data...")
training_data = []
for item in all_data:
    for j in range(len(item["rewards"])):
        training_data.append(
            {
                "prompt": item["prompt"],
                "response": item["full_output"][j],
                "step_map": item["step_map"][j],
                "reward": item["rewards"][j],
                "token_segments": item["token_segments"][j],
            }
        )
print(f"   Training samples: {len(training_data)}")

# ── 4. Test segment-aware masking ────────────────────────────
print("\n[4] Testing segment-aware masking (TraceRL)...")
sys.path.insert(0, "train")
from train.prompting_utils import UniversalPrompting

uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=1024, max_gen_length=512)

prompt_list = [d["prompt"] for d in training_data]
response_list = [d["response"] for d in training_data]
step_map_list = [d["step_map"] for d in training_data]
reward_list = [d["reward"] for d in training_data]

input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting(
    (prompt_list, response_list)
)
print(
    f"   Tokenized: input_ids={input_ids_lm.shape}, start_pos={start_pos}, dropped={drop_num}"
)

# Build trainable mask
token_segments_list = [d["token_segments"] for d in training_data]
B, L = input_ids_lm.shape
trainable_masks = []
for b in range(B):
    mask = torch.zeros(L, dtype=torch.bool)
    if b < len(token_segments_list) and token_segments_list[b]:
        for seg_s, seg_e, is_t in token_segments_list[b]:
            if is_t:
                s = max(seg_s, start_pos)
                e = min(seg_e, L)
                if s < e:
                    mask[s:e] = True
    else:
        mask[start_pos:] = True
    trainable_masks.append(mask)
trainable_masks_t = torch.stack(trainable_masks)
print(
    f"   Trainable mask: {trainable_masks_t.shape}, avg trainable ratio: {trainable_masks_t.float().mean():.3f}"
)


# Simulate TraceRL masking with segment awareness
def collapse_k_unique(lst, k):
    uniq = sorted(set(lst))
    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        mapping[val] = uniq[end_idx]
    return [mapping[x] for x in lst]


noisy_count = 0
for b in range(B):
    order_list = list(step_map_list[b]) if b < len(step_map_list) else []
    if not order_list:
        continue
    order_list = collapse_k_unique(order_list, 8)  # shrink=8
    order = torch.tensor(order_list)
    order_full = torch.full((L,), -1)
    resp_len = min(len(order_list), L - start_pos)
    order_full[start_pos : start_pos + resp_len] = order[:resp_len]
    tmask_b = trainable_masks_t[b]
    uniq_steps = torch.unique(order_full[start_pos:], sorted=True)
    uniq_steps = uniq_steps[uniq_steps >= 0]

    for step_val in uniq_steps:
        tgt_mask = (order_full == step_val) & tmask_b
        if tgt_mask.any():
            noisy_count += 1

print(f"   TraceRL masking: {noisy_count} training sub-samples from {B} trajectories")

# ── 5. Test forward/backward pass ────────────────────────────
print("\n[5] Testing forward/backward pass...")
model.train()

# Take first training sample
sample_input = input_ids_lm[0:1].to("cuda")
sample_labels = labels_lm[0:1].to("cuda")

# Create a masked version (mask some response tokens)
noisy_input = sample_input.clone()
mask_positions = torch.rand(L) < 0.3
mask_positions[:start_pos] = False
mask_positions = mask_positions & trainable_masks_t[0]
noisy_input[0, mask_positions] = mask_id

# Forward
logits = model(noisy_input).logits
print(f"   Forward: logits={logits.shape}")

# Compute loss (simplified)
import torch.nn.functional as F

log_probs = F.log_softmax(logits, dim=-1)
safe_labels = sample_labels.clone()
safe_labels[safe_labels == -100] = 0
tok_lp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

# PPO-style loss with mask
p_mask = mask_positions.unsqueeze(0).float().to("cuda")
reward_val = torch.tensor([training_data[0]["reward"]], device="cuda")
loss = -(tok_lp * p_mask * reward_val.unsqueeze(1)).sum() / max(p_mask.sum(), 1)
print(f"   Loss: {loss.item():.4f}")

# Backward
loss.backward()
grad_norm = (
    sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
    ** 0.5
)
print(f"   Backward OK, grad norm: {grad_norm:.4f}")

# ── 6. Summary ────────────────────────────────────────────────
print(f"\n   GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

print("\n" + "=" * 60)
print("ALL TRAINING PIPELINE TESTS PASSED!")
print("=" * 60)
