"""
Agent RL rollout script for LLaDA-based diffusion language models.

This extends the standard llada_rl_rollout.py to support multi-turn
agent interactions with tool calling. The key difference is:

1. Instead of generating the full response in one shot, we generate
   turn-by-turn, parsing tool calls after each generation.
2. When a tool call is detected, we execute it, format the observation,
   and append it to the context for the next generation turn.
3. We record segment boundaries (which tokens are model-generated vs.
   env-injected) for the training script to know what to train on.
4. The step_map from diffusion is recorded per-turn and concatenated.

Architecture:
  - The model generates up to `max_tokens_per_turn` tokens per turn
  - After each turn, tool calls are parsed and executed
  - Observations are tokenized and appended to the input
  - The process repeats until max_turns or final answer
  - The full trajectory is saved with segment boundaries
"""

from __future__ import annotations
import math, json, os, sys, time, re, random, importlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from jinja2 import Template
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from termcolor import cprint
import multiprocessing as mp

from omegaconf import DictConfig, ListConfig, OmegaConf

# Import the base generation function from the existing rollout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llada_rl_rollout import (
    generate_with_prefix_cache,
    denoise_step_map,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    DiffusionOutput,
)

# Import agent abstractions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.base_env import (
    BaseToolEnv,
    Trajectory,
    Turn,
    TOOL_CALL_START,
    TOOL_CALL_END,
    OBSERVATION_START,
    OBSERVATION_END,
    FINAL_ANSWER_START,
    FINAL_ANSWER_END,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def load_env_from_config(config) -> BaseToolEnv:
    """
    Dynamically load the tool environment from the config.

    Config should specify:
      agent.env_class: "agent.envs.search_qa_env.SearchQAEnv"
      agent.tools: ["agent.tools.search_tool.SearchTool", ...]
      agent.max_turns: 5
      agent.max_tokens_per_turn: 512
    """
    # Import and instantiate tools
    tools = []
    tool_configs = config.agent.get("tools", [])
    for tool_spec in tool_configs:
        if isinstance(tool_spec, str):
            # Simple class path
            module_path, class_name = tool_spec.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            tool_cls = getattr(mod, class_name)
            tools.append(tool_cls())
        elif isinstance(tool_spec, dict) or hasattr(tool_spec, "keys"):
            # Dict with class and kwargs
            tool_spec = (
                dict(tool_spec) if not isinstance(tool_spec, dict) else tool_spec
            )
            module_path, class_name = tool_spec["class"].rsplit(".", 1)
            mod = importlib.import_module(module_path)
            tool_cls = getattr(mod, class_name)
            kwargs = {k: v for k, v in tool_spec.items() if k != "class"}
            tools.append(tool_cls(**kwargs))

    # Import and instantiate environment
    env_class_path = config.agent.env_class
    module_path, class_name = env_class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    env_cls = getattr(mod, class_name)

    env_kwargs = {}
    if hasattr(config.agent, "env_kwargs"):
        env_kwargs = dict(config.agent.env_kwargs)

    env = env_cls(
        tools=tools,
        max_turns=config.agent.get("max_turns", 5),
        max_tokens_per_turn=config.agent.get("max_tokens_per_turn", 512),
        **env_kwargs,
    )
    return env


@dataclass
class AgentTrajectoryOutput:
    """Output from a single agent trajectory rollout."""

    prompt: str
    full_text: str  # Complete flattened text of the trajectory
    turns: List[Dict[str, Any]]  # Serializable turn data
    final_answer: str
    # Segment info: list of (start_char_idx, end_char_idx, is_model_generated)
    segments: List[Tuple[int, int, bool]]
    # Step maps per model-generated segment
    step_maps_per_segment: List[List[int]]
    # Combined step map for the full response (model-gen tokens only)
    combined_step_map: List[int]
    # Token-level segment boundaries: (start_tok, end_tok, is_trainable)
    token_segments: List[Tuple[int, int, bool]]
    response_length: int
    num_turns: int
    trajectory_reward: float = 0.0
    turn_rewards: List[float] = field(default_factory=list)


def agent_rollout_single(
    model,
    tokenizer,
    env: BaseToolEnv,
    prompt_text: str,
    task_data: Dict[str, Any],
    config,
    device,
) -> AgentTrajectoryOutput:
    """
    Run a single agent trajectory: multi-turn generation with tool calling.

    This is the core function that implements the generate-parse-execute loop
    for diffusion language models. Key design choices:

    1. Each turn generates a fixed budget of tokens (max_tokens_per_turn)
    2. After generation, we check for tool calls
    3. If tool calls are found, we execute them and append observations
    4. The next turn starts from the extended context
    5. Diffusion step_maps are recorded per-turn and concatenated
    """
    mask_id = tokenizer.encode("<|mdm_mask|>")[0]
    pad_id = tokenizer.encode("<|endoftext|>")[0]

    max_turns = config.agent.get("max_turns", 5)
    max_tokens_per_turn = config.agent.get("max_tokens_per_turn", 512)

    # Build initial prompt with system message and tool descriptions
    system_prompt = env.get_system_prompt(task_data)
    question = task_data.get("question", "")

    # Format as chat template
    full_prompt = (
        f"<|startoftext|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        f"<|startoftext|><|start_header_id|>user<|end_header_id|>\n"
        f"{question}<|eot_id|>"
        f"<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    current_context = full_prompt
    turns = []
    all_step_maps = []
    token_segments = []  # (start_tok, end_tok, is_trainable)
    total_model_tokens = 0
    combined_step_map = []

    # Tokenize the initial prompt to know where the response starts
    prompt_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    current_token_pos = prompt_len  # Track where we are in the full sequence

    for turn_idx in range(max_turns):
        # Tokenize current context
        enc = tokenizer(
            [current_context],
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        context_len = input_ids.shape[1]

        # Check if context is getting too long
        max_total = config.agent.get("max_total_tokens", 4096)
        remaining_budget = max_total - context_len
        if remaining_budget < 64:
            break

        gen_length = min(max_tokens_per_turn, remaining_budget)
        # Align gen_length to block_size
        block_size = config.rollout.block_size
        gen_length = (gen_length // block_size) * block_size
        if gen_length < block_size:
            break

        # Configure unmasking strategy
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        if not config.rollout.use_cache:
            further_horizon = None
        else:
            further_horizon = config.rollout.further_horizon

        # Generate this turn's tokens
        out = generate_with_prefix_cache(
            model,
            input_ids,
            steps=min(config.rollout.steps, gen_length),
            gen_length=gen_length,
            block_length=block_size,
            temperature=config.rollout.temperature,
            target=config.rollout.target,
            mask_id=mask_id,
            further_horizon=further_horizon,
            use_cache=config.rollout.use_cache,
            unmask_threshold=unmask_threshold,
        )

        # Extract generated tokens
        gen_ids = out.sequences[0, context_len:].tolist()
        gen_text = tokenizer.decode(
            gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        # Remove trailing pad/mask tokens
        gen_text_clean = gen_text.replace(tokenizer.pad_token or "", "").strip()
        # Also remove mask tokens from display
        gen_text_clean = gen_text_clean.replace("<|mdm_mask|>", "").strip()

        # Compute step map for this turn's generation
        turn_step_map = denoise_step_map(out.history, mask_id=mask_id, sample_idx=0)
        turn_step_map = turn_step_map[context_len:].tolist()

        # Offset step_map values to be globally unique across turns
        if combined_step_map:
            offset = max(combined_step_map) + 1
        else:
            offset = 0
        turn_step_map_offset = [s + offset for s in turn_step_map]

        # Record this model-generated segment
        gen_token_count = len(gen_ids)
        token_segments.append(
            (current_token_pos, current_token_pos + gen_token_count, True)
        )
        combined_step_map.extend(turn_step_map_offset)
        all_step_maps.append(turn_step_map)
        current_token_pos += gen_token_count
        total_model_tokens += gen_token_count

        # Parse tool calls from generated text
        tool_calls = env.parse_tool_calls(gen_text_clean)
        final_answer = env.parse_final_answer(gen_text_clean)

        # Record turn
        turn = Turn(
            role="assistant",
            content=gen_text_clean,
            is_model_generated=True,
            tool_calls=tool_calls,
        )
        turns.append(turn)

        # Update context with the generated text
        current_context = current_context + gen_text_clean

        # Check if we should stop
        if final_answer is not None:
            break

        if not tool_calls:
            # No tool calls and no final answer -- implicit end
            break

        # Execute tool calls and get observations
        results = env.execute_tool_calls(tool_calls)
        turn.tool_results = results
        observation = env.format_observation(results)

        # Record observation turn
        obs_turn = Turn(
            role="observation",
            content=observation,
            is_model_generated=False,
            tool_results=results,
        )
        turns.append(obs_turn)

        # Append observation to context and continue generating
        # Add the observation text and a new assistant header
        obs_text = (
            observation
            + "<|eot_id|>"
            + "<|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        current_context = current_context + obs_text

        # Record observation segment (not trainable)
        obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)
        obs_token_count = len(obs_tokens)
        token_segments.append(
            (current_token_pos, current_token_pos + obs_token_count, False)
        )
        # Pad the combined_step_map with zeros for observation tokens
        combined_step_map.extend([0] * obs_token_count)
        current_token_pos += obs_token_count

        # Free GPU memory
        del out
        torch.cuda.empty_cache()

    # Extract final answer if not already found
    if final_answer is None:
        # Try to extract from the last assistant turn
        for turn in reversed(turns):
            if turn.role == "assistant":
                final_answer = env.parse_final_answer(turn.content)
                if final_answer is None:
                    # Use the full content as the answer
                    final_answer = turn.content
                break

    # Serialize turns for JSON output
    turns_data = []
    for t in turns:
        turns_data.append(
            {
                "role": t.role,
                "content": t.content,
                "is_model_generated": t.is_model_generated,
                "tool_calls": [
                    {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                    for tc in t.tool_calls
                ],
                "tool_results": [
                    {
                        "tool_name": tr.tool_name,
                        "success": tr.success,
                        "output": tr.output,
                    }
                    for tr in t.tool_results
                ],
            }
        )

    return AgentTrajectoryOutput(
        prompt=full_prompt,
        full_text=current_context,
        turns=turns_data,
        final_answer=final_answer or "",
        segments=[],  # char-level segments (optional)
        step_maps_per_segment=all_step_maps,
        combined_step_map=combined_step_map,
        token_segments=token_segments,
        response_length=total_model_tokens,
        num_turns=len([t for t in turns if t.role == "assistant"]),
    )


def agent_worker(
    pretrained_model,
    rank,
    task_list,
    result_dict,
    batch_size,
    config,
    env_config,
):
    """
    Per-GPU worker that processes multiple tasks sequentially.

    Each task is an independent agent trajectory (multi-turn interaction).
    Unlike the original worker which batches multiple prompts, here we
    process one trajectory at a time because each trajectory's context
    grows differently based on tool interactions.
    """
    from llada.modeling_llada import LLaDAModelLM

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load model
    model = (
        LLaDAModelLM.from_pretrained(
            pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    # Load environment
    env = load_env_from_config(env_config)

    from tqdm import tqdm

    for task_idx, task_data in tqdm(
        enumerate(task_list),
        desc=f"GPU {rank}",
        position=rank,
        total=len(task_list),
        leave=True,
    ):
        global_idx = task_data["_global_idx"]
        k_sample = task_data.get("_k_sample", 1)

        trajectories = []
        for k in range(k_sample):
            env.reset()
            traj = agent_rollout_single(
                model=model,
                tokenizer=tokenizer,
                env=env,
                prompt_text="",  # Built inside the function
                task_data=task_data,
                config=config,
                device=device,
            )
            trajectories.append(traj)

        # Store results
        result_dict[global_idx] = {
            "question": task_data.get("question", ""),
            "ground_truth_answer": task_data.get("ground_truth_answer", ""),
            "trajectories": [
                {
                    "prompt": t.prompt,
                    "full_text": t.full_text,
                    "full_output": t.full_text[len(t.prompt) :],
                    "turns": t.turns,
                    "final_answer": t.final_answer,
                    "combined_step_map": t.combined_step_map,
                    "token_segments": t.token_segments,
                    "response_length": t.response_length,
                    "num_turns": t.num_turns,
                }
                for t in trajectories
            ],
        }

        torch.cuda.empty_cache()


def random_select(data_list, random_k):
    return random.sample(data_list, min(random_k, len(data_list)))


if __name__ == "__main__":
    config = get_config()
    mp.set_start_method("spawn", force=True)

    project_name = config.experiment.project
    k_sample = config.rollout.num_response_per_task

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    # Load dataset
    dataset_name = (
        config.dataset.train_dataset
        if config.experiment.function == "train"
        else config.evaluation.eval_dataset
    )
    with open(f"../data/{dataset_name}.json", "r") as f:
        data = json.load(f)

    if config.experiment.function == "train":
        random_select_num = config.rollout.num_task_per_step
        data = random_select(data, random_select_num)

    # Assign global indices
    for i, d in enumerate(data):
        d["_global_idx"] = i
        d["_k_sample"] = k_sample

    num = len(data)
    cprint(f"Agent rollout: {num} tasks, {k_sample} trajectories each", "green")

    # Distribute across GPUs
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "Need at least 1 GPU"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    task_chunks = split_even(data, n_gpu)

    # Launch workers
    manager = mp.Manager()
    result_dict = manager.dict()
    procs = []

    for rk in range(n_gpu):
        if not task_chunks[rk]:
            continue
        p = mp.Process(
            target=agent_worker,
            args=(
                pretrained_model,
                rk,
                task_chunks[rk],
                result_dict,
                1,  # batch_size=1 for agent rollout
                config,
                config,  # env_config (same as main config)
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    cprint("Agent rollout done!", "green")

    # Assemble output data
    output_data = []
    for i in range(num):
        if i not in result_dict:
            cprint(f"Warning: task {i} missing from results", "yellow")
            continue

        item = dict(result_dict[i])

        # Flatten to match expected format for reward and training
        flat_item = {
            "question": item["question"],
            "ground_truth_answer": item["ground_truth_answer"],
            "prompt": item["trajectories"][0]["prompt"] if item["trajectories"] else "",
            "full_output": [],
            "step_map": [],
            "extracted_output": [],
            "response_length": [],
            "token_segments": [],
            "turns": [],
            "num_turns": [],
        }

        for traj in item["trajectories"]:
            flat_item["full_output"].append(traj["full_output"])
            flat_item["step_map"].append(traj["combined_step_map"])
            flat_item["extracted_output"].append(traj["final_answer"])
            flat_item["response_length"].append(traj["response_length"])
            flat_item["token_segments"].append(traj["token_segments"])
            flat_item["turns"].append(traj["turns"])
            flat_item["num_turns"].append(traj["num_turns"])

        output_data.append(flat_item)

    # Write output
    outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset_name
    output_file = f"../{project_name}/temp_data/outputs-{outputs_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    cprint(f"Saved {len(output_data)} agent trajectories to {output_file}", "green")

    # Print statistics
    total_turns = sum(
        sum(item["num_turns"]) / len(item["num_turns"])
        for item in output_data
        if item["num_turns"]
    )
    avg_turns = total_turns / len(output_data) if output_data else 0
    avg_resp_len = (
        sum(
            sum(item["response_length"]) / len(item["response_length"])
            for item in output_data
            if item["response_length"]
        )
        / len(output_data)
        if output_data
        else 0
    )

    cprint(
        f"Avg turns: {avg_turns:.1f}, Avg response length: {avg_resp_len:.0f}", "green"
    )
