#!/usr/bin/env python3
"""
Unified experiment script for Agent RL on Diffusion LLMs.

Modes:
  baseline_eval          Zero-shot eval only
  standard_rl            Single-turn RL (no tools), standard TraceRL
  agent_rl               Multi-turn agent RL with tools + segment-aware masking
  agent_rl_no_segment    Agent RL WITHOUT segment-aware masking (ablation)

Usage:
  python scripts/run_experiment.py --mode agent_rl --train_steps 30 --output_dir /workspace/results/exp004
"""

import argparse, sys, os, json, time, random, gc, re, copy
import numpy as np
import torch
import torch.nn.functional as F

os.chdir("/workspace/dLLM-RL")
sys.path.insert(0, ".")
sys.path.insert(0, "sample")
sys.path.insert(0, "train")


# ━━━━━━━━━━━━━━━━ CLI ━━━━━━━━━━━━━━━━
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["baseline_eval", "standard_rl", "agent_rl", "agent_rl_no_segment"],
    )
    p.add_argument(
        "--masking_method", default="TraceRL", choices=["TraceRL", "random_masking"]
    )
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--tasks_per_step", type=int, default=8)
    p.add_argument("--samples_per_task", type=int, default=4)
    p.add_argument("--eval_samples", type=int, default=200)
    p.add_argument("--max_turns", type=int, default=3)
    p.add_argument("--max_tokens_per_turn", type=int, default=128)
    p.add_argument("--model_path", default="/workspace/models/LLaDA-8B-Base")
    p.add_argument("--train_data", default="/workspace/data/gsm8k_train.json")
    p.add_argument("--eval_data", default="/workspace/data/gsm8k_test.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--shrink", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=64)
    return p.parse_args()


# ━━━━━━━━━━━━━━━━ Helpers ━━━━━━━━━━━━━━━━
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer(text):
    """Extract numerical answer from model output."""
    # 1. #### format (GSM8K)
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip().replace(",", "")
    # 2. \boxed{} format
    tag = r"\boxed{"
    start = text.rfind(tag)
    if start != -1:
        i, depth, buf = start + len(tag), 1, []
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(text[i])
            i += 1
        if depth == 0:
            return "".join(buf).strip()
    # 3. <|final_answer_start|>
    m2 = re.search(
        r"<\|final_answer_start\|>(.*?)<\|final_answer_end\|>", text, re.DOTALL
    )
    if m2:
        return m2.group(1).strip()
    # 4. After "=" sign
    eq = re.findall(r"=\s*([\d,]+\.?\d*)", text)
    if eq:
        return eq[-1].replace(",", "").strip()
    # 5. Last number
    nums = re.findall(r"(?<!\|)\b(\d+(?:\.\d+)?)\b", text)
    if nums:
        return nums[-1]
    return text.strip()[:50]


def check_answer(pred, gt):
    try:
        p = float(str(pred).replace(",", "").strip())
        g = float(str(gt).replace(",", "").strip())
        return abs(p - g) < 0.01
    except (ValueError, TypeError):
        return str(pred).strip() == str(gt).strip()


def z_score(lst):
    if not lst:
        return lst
    m = sum(lst) / len(lst)
    s = (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5
    return [(x - m) / s if s > 0 else 0.0 for x in lst]


def collapse_k(lst, k):
    uniq = sorted(set(lst))
    mp = {}
    for idx, val in enumerate(uniq):
        g = idx // k
        mp[val] = uniq[min((g + 1) * k - 1, len(uniq) - 1)]
    return [mp[x] for x in lst]


LOG = []


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ━━━━━━━━━━━━━━━━ Core ━━━━━━━━━━━━━━━━
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log(f"Mode: {args.mode}, Masking: {args.masking_method}")
    log(f"Output: {args.output_dir}")

    # ── Load model ──
    log("Loading model...")
    from transformers import AutoTokenizer
    from llada.modeling_llada import LLaDAModelLM
    from llada_rl_rollout import generate_with_prefix_cache, denoise_step_map
    from agent.tools.calculator_tool import CalculatorTool
    from agent.envs.math_tool_env import MathToolEnv
    from agent.base_env import (
        TOOL_CALL_START,
        TOOL_CALL_END,
        OBSERVATION_START,
        OBSERVATION_END,
    )
    from train.prompting_utils import UniversalPrompting

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LLaDAModelLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to("cuda")
    mask_id = tokenizer.encode("<|mdm_mask|>")[0]
    pad_id = tokenizer.encode("<|endoftext|>")[0]
    log(f"Model loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    env = MathToolEnv(tools=[CalculatorTool()], max_turns=args.max_turns)
    uni_prompting = UniversalPrompting(
        tokenizer, max_prompt_len=1024, max_gen_length=1024
    )

    use_tools = args.mode in ("agent_rl", "agent_rl_no_segment")
    segment_aware = args.mode == "agent_rl"  # only True for full agent_rl

    # ── Load data ──
    with open(args.train_data) as f:
        train_pool = json.load(f)
    with open(args.eval_data) as f:
        eval_pool = json.load(f)
    eval_data = eval_pool[: args.eval_samples]
    log(f"Train pool: {len(train_pool)}, Eval: {len(eval_data)}")

    # ── Prompt builders ──
    def make_prompt(question, with_tools=False):
        if with_tools:
            sys_prompt = env.get_system_prompt({})
        else:
            sys_prompt = "You are a math problem solver. Show your work step by step. Put your final answer after ####."
        return (
            "<|startoftext|><|start_header_id|>system<|end_header_id|>\n"
            + sys_prompt
            + "<|eot_id|>"
            + "<|startoftext|><|start_header_id|>user<|end_header_id|>\n"
            + question
            + "<|eot_id|>"
            + "<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    # ── Generate one trajectory ──
    def generate_trajectory(mdl, task):
        prompt = make_prompt(task["question"], with_tools=use_tools)
        ctx = prompt
        step_map_combined = []
        token_segs = []
        turns = []
        enc = tokenizer([ctx], return_tensors="pt")
        prompt_len = enc["input_ids"].shape[1]
        pos = prompt_len
        num_turns = args.max_turns if use_tools else 1

        for tidx in range(num_turns):
            enc = tokenizer([ctx], return_tensors="pt")
            ids = enc["input_ids"].to("cuda")
            clen = ids.shape[1]
            gl = args.max_tokens_per_turn
            gl = (gl // args.block_size) * args.block_size
            if gl < args.block_size:
                break

            with torch.no_grad():
                out = generate_with_prefix_cache(
                    mdl,
                    ids,
                    steps=min(args.diffusion_steps, gl),
                    gen_length=gl,
                    block_length=args.block_size,
                    temperature=args.temperature,
                    target="confidence",
                    mask_id=mask_id,
                    further_horizon=64,
                    use_cache=True,
                    unmask_threshold=None,
                )

            gids = out.sequences[0, clen:].tolist()
            gtxt = tokenizer.decode(
                gids, skip_special_tokens=False, clean_up_tokenization_spaces=True
            )
            gtxt_c = (
                gtxt.replace(tokenizer.pad_token or "", "")
                .replace("<|mdm_mask|>", "")
                .strip()
            )

            sm = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
            sm = sm[clen:].tolist()
            off = (max(step_map_combined) + 1) if step_map_combined else 0
            step_map_combined.extend([s + off for s in sm])
            token_segs.append((pos, pos + len(gids), True))
            pos += len(gids)
            turns.append(
                {
                    "role": "assistant",
                    "content": gtxt_c,
                    "tool_calls": [],
                    "tool_results": [],
                }
            )
            ctx += gtxt_c
            del out
            torch.cuda.empty_cache()

            if not use_tools:
                break
            tc = env.parse_tool_calls(gtxt_c)
            fa = env.parse_final_answer(gtxt_c)
            if fa is not None or not tc:
                break
            turns[-1]["tool_calls"] = tc
            results = env.execute_tool_calls(tc)
            turns[-1]["tool_results"] = [
                {"name": r.tool_name, "ok": r.success, "out": r.output} for r in results
            ]
            obs = env.format_observation(results)
            obs_txt = (
                obs
                + "<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            ctx += obs_txt
            otoks = tokenizer.encode(obs_txt, add_special_tokens=False)
            token_segs.append((pos, pos + len(otoks), False))
            step_map_combined.extend([0] * len(otoks))
            pos += len(otoks)

        resp = ctx[len(prompt) :]
        ans = extract_answer(resp)
        has_tc = any(t.get("tool_calls") for t in turns)
        tc_ok = sum(1 for t in turns for r in t.get("tool_results", []) if r.get("ok"))
        return {
            "prompt": prompt,
            "response": resp,
            "answer": ans,
            "step_map": step_map_combined,
            "token_segments": token_segs,
            "resp_len": sum(e - s for s, e, t in token_segs if t),
            "n_turns": len([t for t in turns if t["role"] == "assistant"]),
            "has_tool_call": has_tc,
            "tool_ok": tc_ok,
        }

    # ── Evaluate ──
    def evaluate(mdl, data, label="eval"):
        mdl.eval()
        correct = total = tc_count = tc_ok_total = resp_total = 0
        for i, task in enumerate(data):
            try:
                tr = generate_trajectory(mdl, task)
                ok = check_answer(tr["answer"], task["ground_truth_answer"])
                correct += int(ok)
                total += 1
                resp_total += tr["resp_len"]
                tc_count += int(tr["has_tool_call"])
                tc_ok_total += tr["tool_ok"]
            except Exception as e:
                log(f"   Eval error on task {i}: {e}")
                total += 1
            if (i + 1) % 50 == 0:
                log(f"   [{label}] {i + 1}/{len(data)}: acc={correct / total:.4f}")
        acc = correct / max(total, 1)
        avg_len = resp_total / max(total, 1)
        tc_rate = tc_count / max(total, 1)
        log(f"   [{label}] DONE: acc={acc:.4f} len={avg_len:.0f} tc_rate={tc_rate:.4f}")
        return {
            "accuracy": round(acc, 4),
            "avg_length": round(avg_len, 1),
            "tool_call_rate": round(tc_rate, 4),
            "n_eval": total,
        }

    # ── Baseline eval mode ──
    if args.mode == "baseline_eval":
        log("=== BASELINE EVAL ===")
        res = evaluate(model, eval_data, "baseline")
        results = {"config": config, "baseline_eval": res}
        with open(os.path.join(args.output_dir, "RESULTS.json"), "w") as f:
            json.dump(results, f, indent=2)
        log(f"Saved. Accuracy={res['accuracy']}")
        return

    # ━━━━━━━━━━━━━━━━ TRAINING ━━━━━━━━━━━━━━━━
    log("=== INITIAL EVAL ===")
    baseline = evaluate(model, eval_data[:100], "baseline")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    training_log = []
    eval_log = [{"step": 0, **baseline}]
    MAX_SUB_SAMPLES = 100
    BATCH = 2

    for step in range(1, args.train_steps + 1):
        t0 = time.time()
        log(f"\n--- Step {step}/{args.train_steps} ---")

        # ── Rollout ──
        tasks = random.sample(train_pool, min(args.tasks_per_step, len(train_pool)))
        model.eval()
        prompts, responses, smaps, rewards_raw, segs_all = [], [], [], [], []
        step_correct = step_total = 0

        for task in tasks:
            corr_list, trajs = [], []
            for _ in range(args.samples_per_task):
                try:
                    tr = generate_trajectory(model, task)
                except Exception as e:
                    log(f"   Rollout error: {e}")
                    continue
                ok = check_answer(tr["answer"], task["ground_truth_answer"])
                corr_list.append(float(ok))
                trajs.append(tr)
                step_correct += int(ok)
                step_total += 1

            if not corr_list:
                continue
            prop = sum(corr_list) / len(corr_list)
            if prop > 0.9 or prop < 0.1:
                continue  # skip too easy/hard

            rews = z_score(corr_list)
            for k, tr in enumerate(trajs):
                prompts.append(tr["prompt"])
                responses.append(tr["response"])
                smaps.append(tr["step_map"])
                rewards_raw.append(rews[k])
                segs_all.append(tr["token_segments"])

        train_acc = step_correct / max(step_total, 1)
        log(f"   Rollout: {step_total} trajs, acc={train_acc:.3f}, kept={len(prompts)}")

        if len(prompts) < 2:
            log("   Skip: too few samples")
            training_log.append(
                {"step": step, "train_acc": round(train_acc, 4), "n": 0, "loss": 0}
            )
            continue

        # ── Tokenize + mask ──
        ids_lm, labels_lm, start_pos, drop = uni_prompting((prompts, responses))
        B, L = ids_lm.shape

        # Build trainable masks
        tmasks = []
        for b in range(B):
            mk = torch.zeros(L, dtype=torch.bool)
            if segment_aware and b < len(segs_all) and segs_all[b]:
                for s0, s1, tr in segs_all[b]:
                    if tr:
                        a, c = max(s0, start_pos), min(s1, L)
                        if a < c:
                            mk[a:c] = True
            else:
                mk[start_pos:] = True
            tmasks.append(mk)
        tmasks_t = torch.stack(tmasks)

        # ── Create masked training samples ──
        noisy_l, label_l, pmask_l, rew_l = [], [], [], []
        for b in range(B):
            ol = list(smaps[b]) if b < len(smaps) else []
            if not ol:
                continue
            tm = tmasks_t[b]
            base = ids_lm[b]

            if args.masking_method == "TraceRL":
                ol = collapse_k(ol, args.shrink)
                order = torch.tensor(ol)
                of = torch.full((L,), -1)
                rl = min(len(ol), L - start_pos)
                of[start_pos : start_pos + rl] = order[:rl]
                uniq = torch.unique(of[start_pos:], sorted=True)
                uniq = uniq[uniq >= 0]
                for sv in uniq:
                    tgt = (of == sv) & tm
                    if not tgt.any():
                        continue
                    ni = base.clone()
                    ni[(of >= sv) & tm] = mask_id
                    noisy_l.append(ni)
                    label_l.append(labels_lm[b])
                    pmask_l.append(tgt)
                    rew_l.append(rewards_raw[b])
            else:  # random_masking
                for _ in range(min(10, max(1, 100 // max(B, 1)))):
                    t_val = 0.1 + 0.8 * random.random()
                    rm = (torch.rand(L) < t_val) & tm
                    rm[:start_pos] = False
                    if not rm.any():
                        continue
                    ni = base.clone()
                    ni[rm] = mask_id
                    noisy_l.append(ni)
                    label_l.append(labels_lm[b])
                    pmask_l.append(rm)
                    rew_l.append(rewards_raw[b])

        if not noisy_l:
            log("   Skip: no masked samples")
            training_log.append(
                {"step": step, "train_acc": round(train_acc, 4), "n": 0, "loss": 0}
            )
            continue

        # Limit samples
        if len(noisy_l) > MAX_SUB_SAMPLES:
            idx = random.sample(range(len(noisy_l)), MAX_SUB_SAMPLES)
            noisy_l = [noisy_l[i] for i in idx]
            label_l = [label_l[i] for i in idx]
            pmask_l = [pmask_l[i] for i in idx]
            rew_l = [rew_l[i] for i in idx]

        nb = torch.stack(noisy_l)
        lb = torch.stack(label_l)
        pm = torch.stack(pmask_l).float()
        rw = torch.tensor(rew_l, dtype=torch.float32)
        N = nb.shape[0]
        log(f"   Training: {N} sub-samples, seq={nb.shape[1]}")

        # ── Old log-probs ──
        model.eval()
        old_lp = torch.zeros(N, L)
        with torch.no_grad():
            for i in range(0, N, BATCH):
                e = min(i + BATCH, N)
                inp = nb[i:e].to("cuda")
                lab = lb[i:e].to("cuda")
                lgts = model(inp).logits
                lp = F.log_softmax(lgts, dim=-1)
                sl = lab.clone()
                sl[lab == -100] = 0
                old_lp[i:e] = lp.gather(-1, sl.unsqueeze(-1)).squeeze(-1).cpu()
                del lgts, lp, inp, lab
                torch.cuda.empty_cache()

        # ── PPO update ──
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        idx_shuf = list(range(N))
        random.shuffle(idx_shuf)
        n_batches = max((N + BATCH - 1) // BATCH, 1)

        for i in range(0, N, BATCH):
            bi = idx_shuf[i : min(i + BATCH, N)]
            inp = nb[bi].to("cuda")
            lab = lb[bi].to("cuda")
            pmk = pm[bi].to("cuda")
            olp = old_lp[bi].to("cuda")
            adv = rw[bi].to("cuda")

            try:
                lgts = model(inp).logits
                lp_new = F.log_softmax(lgts, dim=-1)
                sl = lab.clone()
                sl[lab == -100] = 0
                nlp = lp_new.gather(-1, sl.unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(nlp - olp)
                clip = torch.clamp(ratio, 1 - args.eps, 1 + args.eps)
                adv_t = adv.unsqueeze(1)
                surr = torch.min(ratio * adv_t, clip * adv_t) * pmk
                nmsk = torch.clamp(pmk.sum(1), min=1)
                ploss = -(surr.sum(1) / nmsk).mean()
                kl = nlp - olp
                kl_est = (-kl).exp() - 1.0 + kl
                kl_loss = (
                    args.beta * (kl_est * pmk).sum(1).mean() if args.beta > 0 else 0
                )
                loss = (ploss + kl_loss) / n_batches
                loss.backward()
                total_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                log(f"   OOM at batch {i}, skipping")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            finally:
                for v in [lgts, lp_new, nlp, ratio, clip, surr, inp, lab, pmk, olp]:
                    try:
                        del v
                    except:
                        pass
                gc.collect()
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        elapsed = time.time() - t0
        log(f"   Loss={total_loss:.6f} Time={elapsed:.0f}s")
        training_log.append(
            {
                "step": step,
                "train_acc": round(train_acc, 4),
                "n": N,
                "loss": round(total_loss, 6),
                "time": round(elapsed, 1),
            }
        )

        # ── Periodic eval ──
        if step % args.eval_every == 0 or step == args.train_steps:
            log(f"   Evaluating at step {step}...")
            ev = evaluate(model, eval_data[:100], f"step{step}")
            ev["step"] = step
            eval_log.append(ev)
            log(f"   Step {step} eval: acc={ev['accuracy']}")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Final full eval ──
    log("\n=== FINAL EVAL (full) ===")
    final = evaluate(model, eval_data, "final")

    # ── Save everything ──
    results = {
        "config": config,
        "baseline_eval": baseline,
        "final_eval": final,
        "improvement": {
            "accuracy_delta": round(final["accuracy"] - baseline["accuracy"], 4),
            "tool_call_rate_delta": round(
                final["tool_call_rate"] - baseline["tool_call_rate"], 4
            ),
        },
        "training_log": training_log,
        "eval_log": eval_log,
    }
    with open(os.path.join(args.output_dir, "RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "experiment.log"), "w") as f:
        f.write("\n".join(LOG))

    # Print summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Metric':<25} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    log("-" * 55)
    log(
        f"{'Accuracy':<25} {baseline['accuracy']:>10.4f} {final['accuracy']:>10.4f} {results['improvement']['accuracy_delta']:>+10.4f}"
    )
    log(
        f"{'Tool Call Rate':<25} {baseline['tool_call_rate']:>10.4f} {final['tool_call_rate']:>10.4f} {results['improvement']['tool_call_rate_delta']:>+10.4f}"
    )
    log(
        f"{'Avg Length':<25} {baseline['avg_length']:>10.1f} {final['avg_length']:>10.1f}"
    )
    log(f"\nResults: {os.path.join(args.output_dir, 'RESULTS.json')}")


if __name__ == "__main__":
    main()
