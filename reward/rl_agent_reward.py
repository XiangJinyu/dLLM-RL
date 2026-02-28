"""
Agent RL reward computation.

Extends the standard rl_reward.py to support:
1. Multi-turn trajectory rewards (via environment reward functions)
2. Turn-level credit assignment for better learning
3. Z-score normalization across trajectories
4. Filtering by solve rate (same as standard RL)
5. Segment-aware training data construction
"""

import json
import os
import sys
import importlib
from scipy.stats import norm
from termcolor import cprint

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def z_score_normalize(lst):
    """Z-score normalize a list of values."""
    if not lst:
        return lst
    mean = sum(lst) / len(lst)
    std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
    if std == 0:
        return [0 for x in lst]
    return [(x - mean) / std for x in lst]


def load_env_for_reward(config):
    """Load the environment to use its compute_reward method."""
    from agent.base_env import BaseToolEnv, Trajectory, Turn

    tools = []
    tool_configs = config.agent.get("tools", [])
    for tool_spec in tool_configs:
        if isinstance(tool_spec, str):
            module_path, class_name = tool_spec.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            tool_cls = getattr(mod, class_name)
            tools.append(tool_cls())
        elif isinstance(tool_spec, dict) or hasattr(tool_spec, "keys"):
            tool_spec = (
                dict(tool_spec) if not isinstance(tool_spec, dict) else tool_spec
            )
            module_path, class_name = tool_spec["class"].rsplit(".", 1)
            mod = importlib.import_module(module_path)
            tool_cls = getattr(mod, class_name)
            kwargs = {k: v for k, v in tool_spec.items() if k != "class"}
            tools.append(tool_cls(**kwargs))

    env_class_path = config.agent.env_class
    module_path, class_name = env_class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    env_cls = getattr(mod, class_name)

    env_kwargs = {}
    if hasattr(config.agent, "env_kwargs"):
        env_kwargs = dict(config.agent.env_kwargs)

    return env_cls(
        tools=tools,
        max_turns=config.agent.get("max_turns", 5),
        **env_kwargs,
    )


def build_trajectory(item, traj_idx, env):
    """Reconstruct a Trajectory object from serialized data for reward computation."""
    from agent.base_env import Trajectory, Turn
    from agent.base_tool import ToolResult

    trajectory = Trajectory(
        prompt=item.get("prompt", ""),
        ground_truth=item.get("ground_truth_answer", ""),
    )

    # Reconstruct turns
    if "turns" in item and traj_idx < len(item["turns"]):
        turns_data = item["turns"][traj_idx]
        for t in turns_data:
            tool_results = []
            for tr in t.get("tool_results", []):
                tool_results.append(
                    ToolResult(
                        tool_name=tr.get("tool_name", ""),
                        success=tr.get("success", False),
                        output=tr.get("output", ""),
                    )
                )

            turn = Turn(
                role=t.get("role", "assistant"),
                content=t.get("content", ""),
                is_model_generated=t.get("is_model_generated", True),
                tool_calls=t.get("tool_calls", []),
                tool_results=tool_results,
            )
            trajectory.turns.append(turn)

    # Set final answer
    if "extracted_output" in item and traj_idx < len(item["extracted_output"]):
        trajectory.final_answer = item["extracted_output"][traj_idx]

    return trajectory


if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project

    if config.experiment.current_epoch == 1:
        pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    if config.experiment.function == "train":
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
    elif config.experiment.function == "evaluation":
        dataset = config.evaluation.eval_dataset
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # Load rollout data
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    with open(file_name, "r") as f:
        data = json.load(f)

    # Load environment for reward computation
    env = load_env_for_reward(config)

    # Compute rewards for all trajectories
    cprint(f"Computing agent rewards for {len(data)} tasks...", "green")

    reward_mode = config.agent.get(
        "reward_mode", "trajectory"
    )  # "trajectory" or "turn_level"

    all_rewards = []
    all_correctness = []
    response_length_list = []

    for i in range(len(data)):
        data[i]["correctness"] = []
        data[i]["rewards"] = []

        k_sample = len(data[i].get("extracted_output", []))

        for j in range(k_sample):
            # Build trajectory and compute reward
            trajectory = build_trajectory(data[i], j, env)
            trajectory = env.compute_reward(trajectory)

            correctness = trajectory.trajectory_reward

            # Length penalty
            resp_len = (
                data[i]["response_length"][j]
                if j < len(data[i].get("response_length", []))
                else 0
            )
            response_length_list.append(resp_len)

            max_gen = OmegaConf.select(
                config, "agent.max_total_tokens", default=MISSING
            )
            if max_gen is not MISSING and resp_len >= max_gen - 5:
                correctness = 0.0

            data[i]["correctness"].append(correctness)
            data[i]["rewards"].append(correctness)

            # Store turn-level rewards if available
            if hasattr(trajectory, "turn_rewards") and trajectory.turn_rewards:
                if "turn_rewards" not in data[i]:
                    data[i]["turn_rewards"] = []
                data[i]["turn_rewards"].append(trajectory.turn_rewards)

        all_correctness.extend(data[i]["correctness"])

    # Z-score normalize rewards within each task group
    final_data = []
    for i in range(len(data)):
        rewards = z_score_normalize(data[i]["correctness"])
        data[i]["rewards"] = rewards

        if config.experiment.function == "train":
            k_sample = len(data[i].get("correctness", []))
            if k_sample == 0:
                continue

            proportion = sum(data[i]["correctness"]) / k_sample
            # Filter: keep only tasks with intermediate difficulty
            if proportion > 0.8 or proportion < 0.2:
                continue

            for j in range(k_sample):
                data_item = {}
                data_item["prompt"] = data[i]["prompt"]
                data_item["reward"] = rewards[j]
                data_item["response"] = data[i]["full_output"][j]
                data_item["step_map"] = data[i]["step_map"][j]

                # Agent-specific fields
                data_item["token_segments"] = (
                    data[i].get("token_segments", [[]])[j]
                    if j < len(data[i].get("token_segments", []))
                    else []
                )
                data_item["turns"] = (
                    data[i].get("turns", [[]])[j]
                    if j < len(data[i].get("turns", []))
                    else []
                )
                data_item["num_turns"] = (
                    data[i].get("num_turns", [1])[j]
                    if j < len(data[i].get("num_turns", []))
                    else 1
                )

                # Turn-level rewards for fine-grained credit assignment
                if (
                    reward_mode == "turn_level"
                    and "turn_rewards" in data[i]
                    and j < len(data[i]["turn_rewards"])
                ):
                    data_item["turn_rewards"] = data[i]["turn_rewards"][j]

                final_data.append(data_item)

    # Save training data
    if config.experiment.function == "train":
        output_path = (
            "../"
            + project_name
            + "/temp_data/"
            + config.dataset.optimization_data
            + ".json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        cprint(
            f"Saved {len(final_data)} agent training samples to {output_path}", "green"
        )

    # Save full data with rewards
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Log results
    outputs_result_name = (
        "../" + project_name + "/results/results-" + outputs_name + ".txt"
    )
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)

    with open(outputs_result_name, "a") as f:
        acc = sum(all_correctness) / len(all_correctness) if all_correctness else 0
        avg_len = (
            sum(response_length_list) / len(response_length_list)
            if response_length_list
            else 0
        )
        avg_turns = sum(
            sum(item.get("num_turns", [1])) / max(len(item.get("num_turns", [1])), 1)
            for item in data
        ) / max(len(data), 1)

        output_text = (
            f"train step: {config.experiment.current_epoch}  "
            f"acc: {acc:.4f}  avg_length: {avg_len:.0f}  avg_turns: {avg_turns:.1f}  "
            f"num_training_samples: {len(final_data) if config.experiment.function == 'train' else 'N/A'}"
        )
        cprint("\n\n" + output_text, color="green")
        f.write(output_text + "\n")
