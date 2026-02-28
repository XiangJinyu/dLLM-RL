"""
End-to-end test for agent rollout on GPU.
Tests: model loading, multi-turn generation, tool calling, step_map recording.
"""

import sys, os, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("/workspace/dLLM-RL")

print("=" * 60)
print("TEST: Agent Rollout End-to-End")
print("=" * 60)

# 1. Test imports
print("\n[1] Testing imports...")
from agent.base_tool import BaseTool, ToolResult
from agent.base_env import BaseToolEnv, Trajectory, Turn
from agent.tools.calculator_tool import CalculatorTool
from agent.tools.python_tool import PythonExecuteTool
from agent.envs.math_tool_env import MathToolEnv

sys.path.insert(0, "/workspace/dLLM-RL/sample")
from llada_rl_rollout import (
    generate_with_prefix_cache,
    denoise_step_map,
    DiffusionOutput,
)

print("   All imports OK")

# 2. Test tools
print("\n[2] Testing tools...")
calc = CalculatorTool()
r = calc.execute(expression="17 * 23 + 45")
print(f"   Calculator: 17*23+45 = {r.output} (success={r.success})")
assert r.success and r.output == "436", f"Calculator failed: {r.output}"

py_tool = PythonExecuteTool(timeout=5)
r2 = py_tool.execute(code="print(sum(range(1, 101)))")
print(f"   Python: sum(1..100) = {r2.output.strip()} (success={r2.success})")
assert r2.success and "5050" in r2.output, f"Python tool failed: {r2.output}"
print("   Tools OK")

# 3. Load model
print("\n[3] Loading LLaDA-8B-Base model...")
t0 = time.time()
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM

model_path = "/workspace/models/LLaDA-8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = (
    LLaDAModelLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    .to("cuda")
    .eval()
)
print(f"   Model loaded in {time.time() - t0:.1f}s")
print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

mask_id = tokenizer.encode("<|mdm_mask|>")[0]
pad_id = tokenizer.encode("<|endoftext|>")[0]

# 4. Test single-turn generation (sanity check)
print("\n[4] Testing single-turn generation...")
prompt_text = "<|startoftext|><|start_header_id|>user<|end_header_id|>What is 2+2?<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
enc = tokenizer([prompt_text], return_tensors="pt")
input_ids = enc["input_ids"].to("cuda")
print(f"   Prompt tokens: {input_ids.shape[1]}")

out = generate_with_prefix_cache(
    model,
    input_ids,
    steps=64,
    gen_length=64,
    block_length=32,
    temperature=0.3,
    target="confidence",
    mask_id=mask_id,
    further_horizon=64,
    use_cache=True,
    unmask_threshold=None,
)
gen_text = tokenizer.decode(
    out.sequences[0, input_ids.shape[1] :], skip_special_tokens=False
)
print(f"   Generated: {gen_text[:200]}")
step_map = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
print(
    f"   Step map shape: {step_map.shape}, unique steps: {len(step_map[input_ids.shape[1] :].unique())}"
)
print("   Single-turn generation OK")
torch.cuda.empty_cache()

# 5. Test multi-turn agent rollout
print("\n[5] Testing multi-turn agent rollout...")
env = MathToolEnv(
    tools=[CalculatorTool(), PythonExecuteTool(timeout=5)],
    max_turns=3,
    max_tokens_per_turn=128,
)

# Simulate what agent_rollout_single does
question = "What is 17 * 23 + 45?"
system_prompt = env.get_system_prompt({})
full_prompt = (
    "<|startoftext|><|start_header_id|>system<|end_header_id|>\n"
    + system_prompt
    + "<|eot_id|>"
    + "<|startoftext|><|start_header_id|>user<|end_header_id|>\n"
    + question
    + "<|eot_id|>"
    + "<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
)

current_context = full_prompt
all_step_maps = []
token_segments = []
combined_step_map = []
turns_info = []

enc = tokenizer([current_context], return_tensors="pt")
prompt_len = enc["input_ids"].shape[1]
current_token_pos = prompt_len
print(f"   System prompt + question: {prompt_len} tokens")

for turn_idx in range(3):
    print(f"\n   --- Turn {turn_idx + 1} ---")
    enc = tokenizer([current_context], return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    context_len = input_ids.shape[1]

    gen_length = 128
    block_size = 32
    gen_length = (gen_length // block_size) * block_size

    print(f"   Context: {context_len} tokens, generating {gen_length} tokens...")

    out = generate_with_prefix_cache(
        model,
        input_ids,
        steps=min(64, gen_length),
        gen_length=gen_length,
        block_length=block_size,
        temperature=0.5,
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

    # Step map
    turn_step_map = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
    turn_step_map = turn_step_map[context_len:].tolist()
    offset = max(combined_step_map) + 1 if combined_step_map else 0
    turn_step_map_offset = [s + offset for s in turn_step_map]

    gen_token_count = len(gen_ids)
    token_segments.append(
        (current_token_pos, current_token_pos + gen_token_count, True)
    )
    combined_step_map.extend(turn_step_map_offset)
    current_token_pos += gen_token_count

    print(f"   Generated {gen_token_count} tokens")
    print(f"   Text: {gen_text_clean[:150]}...")

    # Parse tool calls
    tool_calls = env.parse_tool_calls(gen_text_clean)
    final_answer = env.parse_final_answer(gen_text_clean)

    print(f"   Tool calls: {len(tool_calls)}, Final answer: {final_answer is not None}")

    turns_info.append(
        {
            "turn": turn_idx,
            "tokens": gen_token_count,
            "tool_calls": len(tool_calls),
            "has_final_answer": final_answer is not None,
        }
    )

    current_context = current_context + gen_text_clean

    if final_answer is not None or not tool_calls:
        print(f"   Stopping: final_answer={final_answer}")
        break

    # Execute tools
    results = env.execute_tool_calls(tool_calls)
    for r in results:
        print(f"   Tool result: {r.tool_name}={r.output[:80]}")

    observation = env.format_observation(results)
    obs_text = (
        observation
        + "<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    current_context = current_context + obs_text

    obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
    obs_token_count = len(obs_tokens)
    token_segments.append(
        (current_token_pos, current_token_pos + obs_token_count, False)
    )
    combined_step_map.extend([0] * obs_token_count)
    current_token_pos += obs_token_count

    print(f"   Observation: {obs_token_count} tokens (non-trainable)")

    del out
    torch.cuda.empty_cache()

# 6. Verify data structures
print("\n[6] Verifying data structures...")
print(f"   Total turns: {len(turns_info)}")
print(f"   Token segments: {token_segments}")
print(f"   Combined step_map length: {len(combined_step_map)}")
print(f"   Trainable tokens: {sum(e - s for s, e, t in token_segments if t)}")
print(f"   Non-trainable tokens: {sum(e - s for s, e, t in token_segments if not t)}")

# Verify segment boundaries don't overlap
for i in range(1, len(token_segments)):
    prev_end = token_segments[i - 1][1]
    curr_start = token_segments[i][0]
    assert prev_end == curr_start, f"Gap between segments: {prev_end} != {curr_start}"
print("   Segment boundaries: contiguous (OK)")

# Verify step_map length matches total response tokens
total_resp_tokens = sum(e - s for s, e, _ in token_segments)
assert len(combined_step_map) == total_resp_tokens, (
    f"Step map length mismatch: {len(combined_step_map)} != {total_resp_tokens}"
)
print("   Step map length: matches total response tokens (OK)")

print(f"\n   GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

# 7. Test reward computation
print("\n[7] Testing reward computation...")
traj = Trajectory(
    prompt=full_prompt,
    final_answer="436",
    ground_truth="436",
)
traj.turns = [
    Turn(role="assistant", content="The answer is 436", is_model_generated=True)
]
traj = env.compute_reward(traj)
print(f"   Trajectory reward (correct): {traj.trajectory_reward}")
assert traj.trajectory_reward == 1.0

traj2 = Trajectory(prompt="test", final_answer="999", ground_truth="436")
traj2.turns = [Turn(role="assistant", content="999", is_model_generated=True)]
traj2 = env.compute_reward(traj2)
print(f"   Trajectory reward (wrong): {traj2.trajectory_reward}")
assert traj2.trajectory_reward == 0.0
print("   Reward computation OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
