"""
Full Agent RL Experiment: Baseline eval -> Train 5 steps -> Eval again.

This script runs entirely on a single GPU and tests the complete pipeline:
1. Create training/eval datasets
2. Evaluate baseline (no RL)
3. Run agent RL training loop (5 steps)
4. Evaluate trained model
5. Save results
"""

import sys, os, json, time, random, copy, gc
import torch
import torch.nn.functional as F
import numpy as np

os.chdir("/workspace/dLLM-RL")
sys.path.insert(0, ".")
sys.path.insert(0, "sample")
sys.path.insert(0, "train")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "/workspace/models/LLaDA-8B-Base"
NUM_TRAIN_STEPS = 5
NUM_TASKS_PER_STEP = 6  # problems per step
NUM_SAMPLES_PER_TASK = 4  # trajectories per problem
MAX_TOKENS_PER_TURN = 128
MAX_TURNS = 2
BLOCK_SIZE = 32
DIFFUSION_STEPS = 64
TEMPERATURE = 0.8
LEARNING_RATE = 5e-6
SHRINK = 8
EPS = 0.2
BETA = 0.01
EVAL_TASKS = 30

# ── Datasets ────────────────────────────────────────────
TRAIN_DATA = [
    {"question": "What is 17 * 23?", "ground_truth_answer": "391"},
    {"question": "What is 144 / 12?", "ground_truth_answer": "12"},
    {"question": "What is 2^8?", "ground_truth_answer": "256"},
    {"question": "What is 15 * 15?", "ground_truth_answer": "225"},
    {"question": "What is 99 + 101?", "ground_truth_answer": "200"},
    {"question": "What is 7 * 8 * 9?", "ground_truth_answer": "504"},
    {"question": "What is 1000 - 373?", "ground_truth_answer": "627"},
    {"question": "What is 25 * 4?", "ground_truth_answer": "100"},
    {"question": "What is 13 * 17?", "ground_truth_answer": "221"},
    {"question": "What is 256 + 512?", "ground_truth_answer": "768"},
    {"question": "What is 50 * 50?", "ground_truth_answer": "2500"},
    {"question": "What is 3^5?", "ground_truth_answer": "243"},
    {"question": "What is 999 - 111?", "ground_truth_answer": "888"},
    {"question": "What is 64 * 8?", "ground_truth_answer": "512"},
    {"question": "What is 1024 / 16?", "ground_truth_answer": "64"},
    {"question": "What is 33 * 33?", "ground_truth_answer": "1089"},
    {"question": "What is 7^3?", "ground_truth_answer": "343"},
    {"question": "What is 45 + 55 + 100?", "ground_truth_answer": "200"},
    {"question": "What is 12 * 12 * 12?", "ground_truth_answer": "1728"},
    {"question": "What is 500 - 247?", "ground_truth_answer": "253"},
]

EVAL_DATA = [
    {"question": "What is 19 * 21?", "ground_truth_answer": "399"},
    {"question": "What is 2^10?", "ground_truth_answer": "1024"},
    {"question": "What is 37 * 43?", "ground_truth_answer": "1591"},
    {"question": "What is 256 + 789?", "ground_truth_answer": "1045"},
    {"question": "What is 81 * 9?", "ground_truth_answer": "729"},
    {"question": "What is 4^4?", "ground_truth_answer": "256"},
    {"question": "What is 123 + 456?", "ground_truth_answer": "579"},
    {"question": "What is 17 * 23 + 45?", "ground_truth_answer": "436"},
    {"question": "What is 100 * 100 - 1?", "ground_truth_answer": "9999"},
    {"question": "What is 15 * 16?", "ground_truth_answer": "240"},
    {"question": "What is 2^12?", "ground_truth_answer": "4096"},
    {"question": "What is 77 * 13?", "ground_truth_answer": "1001"},
    {"question": "What is 999 + 1?", "ground_truth_answer": "1000"},
    {"question": "What is 48 * 52?", "ground_truth_answer": "2496"},
    {"question": "What is 5^5?", "ground_truth_answer": "3125"},
    {"question": "What is 1111 - 222?", "ground_truth_answer": "889"},
    {"question": "What is 16 * 32?", "ground_truth_answer": "512"},
    {"question": "What is 3^6?", "ground_truth_answer": "729"},
    {"question": "What is 88 + 88 + 88?", "ground_truth_answer": "264"},
    {"question": "What is 29 * 31?", "ground_truth_answer": "899"},
]

print("=" * 60)
print("AGENT RL EXPERIMENT")
print(f"Train steps: {NUM_TRAIN_STEPS}")
print(f"Tasks/step: {NUM_TASKS_PER_STEP}, Samples/task: {NUM_SAMPLES_PER_TASK}")
print(f"Eval tasks: {len(EVAL_DATA)}")
print("=" * 60)

# ── Load model and tools ────────────────────────────────
from transformers import AutoTokenizer
from llada.modeling_llada import LLaDAModelLM
from llada_rl_rollout import generate_with_prefix_cache, denoise_step_map
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
from train.prompting_utils import UniversalPrompting

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LLaDAModelLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
mask_id = tokenizer.encode("<|mdm_mask|>")[0]
pad_id = tokenizer.encode("<|endoftext|>")[0]
print(f"Model loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

env = MathToolEnv(tools=[CalculatorTool()], max_turns=MAX_TURNS)
uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=1024, max_gen_length=512)


def make_prompt(question):
    system_prompt = env.get_system_prompt({})
    return (
        "<|startoftext|><|start_header_id|>system<|end_header_id|>\n"
        + system_prompt
        + "<|eot_id|>"
        + "<|startoftext|><|start_header_id|>user<|end_header_id|>\n"
        + question
        + "<|eot_id|>"
        + "<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def extract_answer(text):
    """Try to extract a numerical answer from model output."""
    # Try final_answer format
    ans = env.parse_final_answer(text)
    if ans is not None:
        return ans.strip()
    # Try boxed format
    import re

    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    # Try to find a number after "="
    eq_match = re.findall(r"=\s*(\d+[\d,]*\.?\d*)", text)
    if eq_match:
        return eq_match[-1].replace(",", "").strip()
    # Try last number in text
    nums = re.findall(r"\b(\d+)\b", text)
    if nums:
        return nums[-1]
    return text.strip()[:50]


def check_answer(predicted, ground_truth):
    """Simple numerical answer checking."""
    try:
        p = float(predicted.replace(",", "").strip())
        g = float(ground_truth.replace(",", "").strip())
        return abs(p - g) < 0.01
    except (ValueError, AttributeError):
        return predicted.strip() == ground_truth.strip()


def generate_trajectory(model, task, num_turns=MAX_TURNS):
    """Generate one agent trajectory."""
    prompt = make_prompt(task["question"])
    current_context = prompt
    combined_step_map = []
    token_segments = []
    turns_data = []

    enc = tokenizer([current_context], return_tensors="pt")
    prompt_len = enc["input_ids"].shape[1]
    current_pos = prompt_len

    for turn_idx in range(num_turns):
        enc = tokenizer([current_context], return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        ctx_len = input_ids.shape[1]

        gen_length = MAX_TOKENS_PER_TURN
        gen_length = (gen_length // BLOCK_SIZE) * BLOCK_SIZE

        with torch.no_grad():
            out = generate_with_prefix_cache(
                model,
                input_ids,
                steps=min(DIFFUSION_STEPS, gen_length),
                gen_length=gen_length,
                block_length=BLOCK_SIZE,
                temperature=TEMPERATURE,
                target="confidence",
                mask_id=mask_id,
                further_horizon=64,
                use_cache=True,
                unmask_threshold=None,
            )

        gen_ids = out.sequences[0, ctx_len:].tolist()
        gen_text = tokenizer.decode(
            gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        gen_text_clean = (
            gen_text.replace(tokenizer.pad_token or "", "")
            .replace("<|mdm_mask|>", "")
            .strip()
        )

        sm = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
        sm = sm[ctx_len:].tolist()
        offset = max(combined_step_map) + 1 if combined_step_map else 0
        combined_step_map.extend([s + offset for s in sm])
        token_segments.append((current_pos, current_pos + len(gen_ids), True))
        current_pos += len(gen_ids)

        tool_calls = env.parse_tool_calls(gen_text_clean)
        final_answer = env.parse_final_answer(gen_text_clean)
        turns_data.append(
            {
                "role": "assistant",
                "content": gen_text_clean,
                "is_model_generated": True,
                "tool_calls": tool_calls or [],
                "tool_results": [],
            }
        )
        current_context += gen_text_clean

        if final_answer is not None or not tool_calls:
            break

        results = env.execute_tool_calls(tool_calls)
        turns_data[-1]["tool_results"] = [
            {"tool_name": r.tool_name, "success": r.success, "output": r.output}
            for r in results
        ]
        obs = env.format_observation(results)
        obs_text = (
            obs
            + "<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        current_context += obs_text
        obs_toks = tokenizer.encode(obs_text, add_special_tokens=False)
        token_segments.append((current_pos, current_pos + len(obs_toks), False))
        combined_step_map.extend([0] * len(obs_toks))
        current_pos += len(obs_toks)

        del out
        torch.cuda.empty_cache()

    response = current_context[len(prompt) :]
    extracted = extract_answer(response)

    return {
        "prompt": prompt,
        "full_output": response,
        "step_map": combined_step_map,
        "extracted_output": extracted,
        "response_length": sum(e - s for s, e, _ in token_segments if _),
        "token_segments": token_segments,
        "turns": turns_data,
        "num_turns": len([t for t in turns_data if t["role"] == "assistant"]),
        "has_tool_call": any(t.get("tool_calls") for t in turns_data),
    }


def evaluate(model, eval_data, label=""):
    """Evaluate model on a set of tasks."""
    model.eval()
    correct = 0
    total = len(eval_data)
    tool_calls = 0
    total_len = 0
    results_detail = []

    for i, task in enumerate(eval_data):
        traj = generate_trajectory(model, task, num_turns=MAX_TURNS)
        is_correct = check_answer(traj["extracted_output"], task["ground_truth_answer"])
        correct += int(is_correct)
        total_len += traj["response_length"]
        tool_calls += int(traj["has_tool_call"])
        results_detail.append(
            {
                "question": task["question"],
                "ground_truth": task["ground_truth_answer"],
                "predicted": traj["extracted_output"],
                "correct": is_correct,
                "num_turns": traj["num_turns"],
                "has_tool_call": traj["has_tool_call"],
            }
        )
        if (i + 1) % 5 == 0:
            print(f"   [{label}] {i + 1}/{total}: acc={correct / (i + 1):.3f}")

    acc = correct / total
    avg_len = total_len / total
    tool_rate = tool_calls / total
    print(
        f"   [{label}] Final: acc={acc:.4f}, avg_len={avg_len:.0f}, tool_rate={tool_rate:.3f}"
    )
    return {
        "accuracy": round(acc, 4),
        "avg_length": round(avg_len, 1),
        "tool_call_rate": round(tool_rate, 4),
        "details": results_detail,
    }


def z_score_normalize(lst):
    if not lst:
        return lst
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std == 0:
        return [0.0 for _ in lst]
    return [(x - mean) / std for x in lst]


def collapse_k_unique(lst, k):
    uniq = sorted(set(lst))
    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        mapping[val] = uniq[end_idx]
    return [mapping[x] for x in lst]


# ── Phase 1: Baseline Evaluation ────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: Baseline Evaluation (no RL)")
print("=" * 60)
baseline_results = evaluate(model, EVAL_DATA, label="baseline")

# ── Phase 2: Agent RL Training ──────────────────────────
print("\n" + "=" * 60)
print("PHASE 2: Agent RL Training")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
training_log = []

for step in range(1, NUM_TRAIN_STEPS + 1):
    t0 = time.time()
    print(f"\n--- Train Step {step}/{NUM_TRAIN_STEPS} ---")

    # 2a. Sample tasks and generate trajectories
    tasks = random.sample(TRAIN_DATA, min(NUM_TASKS_PER_STEP, len(TRAIN_DATA)))
    model.eval()

    all_prompts, all_responses, all_step_maps, all_rewards, all_segments = (
        [],
        [],
        [],
        [],
        [],
    )
    step_correct, step_total = 0, 0

    for task in tasks:
        correctness_list = []
        task_trajs = []
        for k in range(NUM_SAMPLES_PER_TASK):
            traj = generate_trajectory(model, task, num_turns=MAX_TURNS)
            is_correct = check_answer(
                traj["extracted_output"], task["ground_truth_answer"]
            )
            correctness_list.append(float(is_correct))
            task_trajs.append(traj)
            step_correct += int(is_correct)
            step_total += 1

        # Filter: skip tasks that are too easy or too hard
        prop = sum(correctness_list) / len(correctness_list)
        if prop > 0.8 or prop < 0.2:
            continue

        rewards = z_score_normalize(correctness_list)
        for k, traj in enumerate(task_trajs):
            all_prompts.append(traj["prompt"])
            all_responses.append(traj["full_output"])
            all_step_maps.append(traj["step_map"])
            all_rewards.append(rewards[k])
            all_segments.append(traj["token_segments"])

    train_acc = step_correct / max(step_total, 1)
    print(
        f"   Rollout: {step_total} trajs, acc={train_acc:.3f}, kept={len(all_prompts)} for training"
    )

    if len(all_prompts) == 0:
        print("   No valid training data this step, skipping")
        training_log.append(
            {"step": step, "train_acc": train_acc, "num_samples": 0, "loss": 0}
        )
        continue

    # 2b. Prepare training data with segment-aware TraceRL masking
    input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting(
        (all_prompts, all_responses)
    )
    B, L = input_ids_lm.shape

    # Build trainable masks
    trainable_masks = []
    for b in range(B):
        mask = torch.zeros(L, dtype=torch.bool)
        if b < len(all_segments) and all_segments[b]:
            for seg_s, seg_e, is_t in all_segments[b]:
                if is_t:
                    s, e = max(seg_s, start_pos), min(seg_e, L)
                    if s < e:
                        mask[s:e] = True
        else:
            mask[start_pos:] = True
        trainable_masks.append(mask)
    trainable_masks_t = torch.stack(trainable_masks)

    # Apply TraceRL masking
    noisy_list, label_list, pmask_list, reward_list_out = [], [], [], []
    for b in range(B):
        order_list = list(all_step_maps[b]) if b < len(all_step_maps) else []
        if not order_list:
            continue
        order_list = collapse_k_unique(order_list, SHRINK)
        order = torch.tensor(order_list)
        order_full = torch.full((L,), -1)
        resp_len = min(len(order_list), L - start_pos)
        order_full[start_pos : start_pos + resp_len] = order[:resp_len]
        tmask_b = trainable_masks_t[b]
        uniq_steps = torch.unique(order_full[start_pos:], sorted=True)
        uniq_steps = uniq_steps[uniq_steps >= 0]
        base_ids = input_ids_lm[b]

        for step_val in uniq_steps:
            tgt_mask = (order_full == step_val) & tmask_b
            if not tgt_mask.any():
                continue
            noisy_ids = base_ids.clone()
            mask_pos = (order_full >= step_val) & tmask_b
            noisy_ids[mask_pos] = mask_id
            noisy_list.append(noisy_ids)
            label_list.append(labels_lm[b])
            pmask_list.append(tgt_mask)
            reward_list_out.append(all_rewards[b])

    if not noisy_list:
        print("   No valid masking data, skipping")
        training_log.append(
            {"step": step, "train_acc": train_acc, "num_samples": 0, "loss": 0}
        )
        continue

    # Limit training samples to avoid OOM (max ~100 per step)
    MAX_TRAIN_SAMPLES = 100
    if len(noisy_list) > MAX_TRAIN_SAMPLES:
        sample_idx = random.sample(range(len(noisy_list)), MAX_TRAIN_SAMPLES)
        noisy_list = [noisy_list[i] for i in sample_idx]
        label_list = [label_list[i] for i in sample_idx]
        pmask_list = [pmask_list[i] for i in sample_idx]
        reward_list_out = [reward_list_out[i] for i in sample_idx]

    noisy_batch = torch.stack(noisy_list)
    labels_batch = torch.stack(label_list)
    pmask_batch = torch.stack(pmask_list).float()
    rewards_tensor = torch.tensor(reward_list_out, dtype=torch.float32)

    print(
        f"   Training samples: {noisy_batch.shape[0]}, seq_len={noisy_batch.shape[1]}"
    )

    # 2c. Compute old log-probs
    model.eval()
    with torch.no_grad():
        batch_size = 2
        logp_old_all = torch.zeros(noisy_batch.shape[0], noisy_batch.shape[1])
        for i in range(0, noisy_batch.shape[0], batch_size):
            end = min(i + batch_size, noisy_batch.shape[0])
            inp = noisy_batch[i:end].to("cuda")
            lab = labels_batch[i:end].to("cuda")
            logits = model(inp).logits
            lp = F.log_softmax(logits, dim=-1)
            safe_lab = lab.clone()
            safe_lab[lab == -100] = 0
            tok_lp = lp.gather(-1, safe_lab.unsqueeze(-1)).squeeze(-1)
            logp_old_all[i:end] = tok_lp.cpu()
            del logits, lp, inp, lab
            torch.cuda.empty_cache()

    # 2d. PPO training
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    indices = list(range(noisy_batch.shape[0]))
    random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : min(i + batch_size, len(indices))]
        inp = noisy_batch[batch_idx].to("cuda")
        lab = labels_batch[batch_idx].to("cuda")
        pmask = pmask_batch[batch_idx].to("cuda")
        old_lp = logp_old_all[batch_idx].to("cuda")
        adv = rewards_tensor[batch_idx].to("cuda")

        logits = model(inp).logits
        log_probs = F.log_softmax(logits, dim=-1)
        safe_lab = lab.clone()
        safe_lab[lab == -100] = 0
        new_lp = log_probs.gather(-1, safe_lab.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
        adv_tok = adv.unsqueeze(1)
        surrogate = torch.min(ratio * adv_tok, clipped * adv_tok)
        surrogate = surrogate * pmask
        num_mask = torch.clamp(pmask.sum(dim=1), min=1)
        surrogate = surrogate.sum(dim=1) / num_mask
        policy_loss = -surrogate.mean()

        # KL penalty
        if BETA > 0:
            kl = new_lp - old_lp
            kl = (-kl).exp() - 1.0 + kl
            kl = (kl * pmask).sum(dim=1)
            kl_loss = BETA * kl.mean()
            loss = policy_loss + kl_loss
        else:
            loss = policy_loss

        loss = loss / max(len(indices) // batch_size, 1)
        loss.backward()
        total_loss += loss.item()
        num_batches += 1

        del (
            logits,
            log_probs,
            inp,
            lab,
            pmask,
            old_lp,
            new_lp,
            ratio,
            clipped,
            surrogate,
            adv_tok,
        )
        try:
            del kl, kl_loss
        except NameError:
            pass
        try:
            del loss, policy_loss
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    avg_loss = total_loss
    elapsed = time.time() - t0
    print(f"   Loss: {avg_loss:.6f}, Time: {elapsed:.1f}s")
    training_log.append(
        {
            "step": step,
            "train_acc": round(train_acc, 4),
            "num_samples": noisy_batch.shape[0],
            "loss": round(avg_loss, 6),
        }
    )

    gc.collect()
    torch.cuda.empty_cache()

# ── Phase 3: Post-training Evaluation ───────────────────
print("\n" + "=" * 60)
print("PHASE 3: Post-training Evaluation")
print("=" * 60)
trained_results = evaluate(model, EVAL_DATA, label="trained")

# ── Phase 4: Save Results ───────────────────────────────
results = {
    "config": {
        "model": "LLaDA-8B-Base",
        "num_train_steps": NUM_TRAIN_STEPS,
        "tasks_per_step": NUM_TASKS_PER_STEP,
        "samples_per_task": NUM_SAMPLES_PER_TASK,
        "max_turns": MAX_TURNS,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "shrink": SHRINK,
        "eps": EPS,
        "beta": BETA,
        "seed": SEED,
    },
    "baseline": {
        "accuracy": baseline_results["accuracy"],
        "avg_length": baseline_results["avg_length"],
        "tool_call_rate": baseline_results["tool_call_rate"],
    },
    "trained": {
        "accuracy": trained_results["accuracy"],
        "avg_length": trained_results["avg_length"],
        "tool_call_rate": trained_results["tool_call_rate"],
    },
    "improvement": {
        "accuracy_delta": round(
            trained_results["accuracy"] - baseline_results["accuracy"], 4
        ),
        "tool_call_rate_delta": round(
            trained_results["tool_call_rate"] - baseline_results["tool_call_rate"], 4
        ),
    },
    "training_log": training_log,
}

os.makedirs("/workspace/results", exist_ok=True)
with open("/workspace/results/RESULTS.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "=" * 60)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 60)
print(f"{'Metric':<25} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
print("-" * 55)
print(
    f"{'Accuracy':<25} {baseline_results['accuracy']:>10.4f} {trained_results['accuracy']:>10.4f} {results['improvement']['accuracy_delta']:>+10.4f}"
)
print(
    f"{'Avg Response Length':<25} {baseline_results['avg_length']:>10.1f} {trained_results['avg_length']:>10.1f} {trained_results['avg_length'] - baseline_results['avg_length']:>+10.1f}"
)
print(
    f"{'Tool Call Rate':<25} {baseline_results['tool_call_rate']:>10.4f} {trained_results['tool_call_rate']:>10.4f} {results['improvement']['tool_call_rate_delta']:>+10.4f}"
)
print("\nTraining log:")
for log in training_log:
    print(
        f"  Step {log['step']}: acc={log['train_acc']:.3f}, samples={log['num_samples']}, loss={log['loss']:.6f}"
    )
print(f"\nResults saved to /workspace/results/RESULTS.json")

# Detail comparison
print("\nPer-question comparison (first 10):")
for i in range(min(10, len(EVAL_DATA))):
    b = baseline_results["details"][i]
    t = trained_results["details"][i]
    b_mark = "V" if b["correct"] else "X"
    t_mark = "V" if t["correct"] else "X"
    change = ""
    if t["correct"] and not b["correct"]:
        change = " [IMPROVED]"
    elif not t["correct"] and b["correct"]:
        change = " [REGRESSED]"
    print(
        f"  Q: {b['question'][:40]:<40} GT={b['ground_truth']:<8} Base={b['predicted'][:8]:<8}({b_mark}) Train={t['predicted'][:8]:<8}({t_mark}){change}"
    )
