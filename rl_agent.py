"""
Agent RL orchestrator for dLLM-RL.

This is the top-level script that runs the agent RL training loop.
It extends rl.py to support multi-turn agent interactions:

  while i <= total_step:
      1. agent_sample(i)     -- Multi-turn rollout with tool calling
      2. agent_reward(i)     -- Compute trajectory + turn-level rewards
      3. train(i)            -- Segment-aware PPO training
      4. [eval]              -- Periodic evaluation
      i += 1

Usage:
    python rl_agent.py config=configs/rl_agent_llada.yaml
"""

import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


if __name__ == "__main__":
    config = get_config()

    start_from_scratch = config.experiment.start_from_scratch
    project_name = config.experiment.project
    model_base = config.model.model_base

    # Check for value model
    if (
        OmegaConf.select(config, "model.value_base_model", default=MISSING)
        is not MISSING
    ):
        have_value_model = True
    else:
        have_value_model = False

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")

    if start_from_scratch:
        os.makedirs(f"{project_name}/results", exist_ok=True)
        os.makedirs(f"{project_name}/temp_data", exist_ok=True)
        optimized_model = "../" + project_name + "/ckpt/" + config.model.optimized_name
        begin_with(
            f"{project_name}/results/results-rl-"
            + optimized_model.replace("/", ".")
            + "-"
            + config.dataset.train_dataset
            + ".txt"
        )
        begin_with(
            f"{project_name}/results/results-eval-"
            + optimized_model.replace("/", ".")
            + "-"
            + config.dataset.train_dataset
            + ".txt"
        )

    def agent_sample(i, function="train"):
        """Run multi-turn agent rollout."""
        # Select the appropriate rollout script based on model base
        if model_base == "llada" or model_base == "mmada":
            script_name = "agent_llada_rl_rollout.py"
        else:
            # For now, agent rollout is implemented for LLaDA-family models
            # TraDo/SDAR/Dream support can be added following the same pattern
            cprint(
                f"Warning: Agent rollout for {model_base} not yet implemented, using llada rollout",
                "yellow",
            )
            script_name = "agent_llada_rl_rollout.py"

        subprocess.run(
            f"python {script_name} "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={function} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd="sample",
            check=True,
        )

    def agent_reward(i, function="train"):
        """Compute agent trajectory rewards."""
        subprocess.run(
            f"python rl_agent_reward.py "
            f"config=../configs/{project_name}.yaml "
            f"experiment.function={function} "
            f"experiment.current_epoch={i} ",
            shell=True,
            cwd="reward",
            check=True,
        )

    def train(i, target=None):
        """Run agent-aware training."""
        if target is None:
            # Use agent-specific training script
            if model_base == "llada" or model_base == "mmada":
                script_name = "rl_agent_llada.py"
            else:
                cprint(
                    f"Warning: Agent training for {model_base} not yet implemented, using llada training",
                    "yellow",
                )
                script_name = "rl_agent_llada.py"
        elif target == "policy":
            # TODO: add agent-aware policy training with value model
            script_name = "rl_agent_llada.py"
        elif target == "value":
            # TODO: add agent-aware value model training
            cprint("Value model training for agent RL is not yet implemented", "yellow")
            return

        subprocess.run(
            f"accelerate launch "
            f"--num_machines 1 "
            f"--machine_rank 0 "
            f"--main_process_ip 127.0.0.1 "
            f"--main_process_port 8888 "
            f"--config_file accelerate_configs/{config.experiment.deepspeed_file}.yaml "
            f"train/{script_name} "
            f"config=configs/{project_name}.yaml "
            f"experiment.current_epoch={i} ",
            shell=True,
            check=True,
        )

    # Standard evaluation (can use either agent or standard eval)
    def evaluate(i):
        """Run evaluation."""
        block_size_list = config.evaluation.block_size
        remasking_strategy_list = config.evaluation.remasking_strategy

        if OmegaConf.select(config, "evaluation.top_k", default=MISSING) is not MISSING:
            top_k = config.evaluation.top_k
        else:
            top_k = None

        for j in range(len(remasking_strategy_list)):
            remasking_strategy = remasking_strategy_list[j]
            if isinstance(block_size_list, list):
                block_size = (
                    block_size_list[j]
                    if j < len(block_size_list)
                    else block_size_list[0]
                )
            else:
                block_size = block_size_list

            # Use agent rollout for evaluation too
            agent_sample(i, function="evaluation")
            agent_reward(i, function="evaluation")

    # ── Main training loop ──────────────────────────────────────
    i = config.experiment.current_epoch

    cprint(f"\n{'=' * 60}", "cyan")
    cprint(f"  Agent RL Training: {project_name}", "cyan")
    cprint(f"  Model: {model_base}", "cyan")
    cprint(f"  Steps: {i} -> {config.experiment.total_step}", "cyan")
    cprint(f"  Max turns: {config.agent.get('max_turns', 5)}", "cyan")
    cprint(
        f"  Tools: {[t if isinstance(t, str) else t.get('class', '?') for t in config.agent.get('tools', [])]}",
        "cyan",
    )
    cprint(f"{'=' * 60}\n", "cyan")

    while i <= config.experiment.total_step:
        cprint(f"\n--- Step {i}/{config.experiment.total_step} ---", "green")

        # 1. Multi-turn agent rollout
        cprint("[1/3] Agent rollout (multi-turn generation with tools)...", "yellow")
        agent_sample(i, "train")

        # 2. Compute rewards
        cprint("[2/3] Computing agent rewards...", "yellow")
        agent_reward(i, "train")

        # 3. Train
        cprint("[3/3] Agent RL training (segment-aware PPO)...", "yellow")
        if have_value_model:
            train(i, target="value")
            train(i, target="policy")
        else:
            train(i, target=None)

        # 4. Periodic evaluation
        if i % config.experiment.eval_every == 0:
            cprint("[Eval] Running evaluation...", "yellow")
            evaluate(i)

        cprint(f"--- Step {i} complete ---\n", "green")
        i += 1

    cprint("\nAgent RL training complete!", "green")
