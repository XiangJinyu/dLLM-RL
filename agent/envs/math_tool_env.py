"""
Math tool environment for agent RL training.

The agent can use a calculator and Python execution to solve math problems.
This combines the existing math reward verification from dLLM-RL with
multi-turn tool-calling capabilities.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from agent.base_env import BaseToolEnv, Trajectory, FINAL_ANSWER_START, FINAL_ANSWER_END
from agent.base_tool import BaseTool


class MathToolEnv(BaseToolEnv):
    """
    Environment for training math problem-solving agents with tool access.

    The agent can use a calculator and/or Python interpreter to help
    solve math problems. Final answers are verified against ground truth
    using the same math_utils from the base dLLM-RL framework.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        max_turns: int = 5,
        max_tokens_per_turn: int = 512,
        use_turn_rewards: bool = False,
        math_verify_fn=None,
    ):
        super().__init__(tools, max_turns, max_tokens_per_turn)
        self.use_turn_rewards = use_turn_rewards
        self.math_verify_fn = math_verify_fn

    def parse_final_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer -- supports both the agent protocol format
        and the standard \\boxed{} format from math tasks.
        """
        # First try the agent protocol format
        answer = super().parse_final_answer(text)
        if answer is not None:
            return answer

        # Fallback: try \\boxed{} format
        tag = r"\boxed{"
        start = text.rfind(tag)
        if start == -1:
            return None

        i = start + len(tag)
        depth = 1
        buf = []
        while i < len(text) and depth:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1

        if depth == 0:
            return "".join(buf)
        return None

    def compute_reward(self, trajectory: Trajectory) -> Trajectory:
        """
        Compute rewards for math problem-solving trajectory.

        Trajectory reward: correctness of the final answer.
        Turn rewards: optional credit assignment for tool usage.
        """
        correctness = 0.0

        if trajectory.final_answer and trajectory.ground_truth:
            if self.math_verify_fn is not None:
                try:
                    correctness = float(
                        self.math_verify_fn(
                            trajectory.final_answer, trajectory.ground_truth
                        )
                    )
                except Exception:
                    correctness = 0.0
            else:
                # Simple string matching fallback
                pred = trajectory.final_answer.strip()
                gt = trajectory.ground_truth.strip()
                correctness = 1.0 if pred == gt else 0.0

        trajectory.trajectory_reward = correctness

        # Per-turn rewards
        if self.use_turn_rewards:
            trajectory.turn_rewards = []
            for turn in trajectory.turns:
                if not turn.is_model_generated:
                    trajectory.turn_rewards.append(0.0)
                elif turn.tool_calls:
                    # Small reward for successful tool use
                    turn_reward = 0.0
                    for result in turn.tool_results:
                        if result.success:
                            turn_reward += 0.05
                    trajectory.turn_rewards.append(turn_reward)
                else:
                    # Final answer turn gets the trajectory reward
                    trajectory.turn_rewards.append(correctness)
        else:
            trajectory.turn_rewards = [
                correctness if t.is_model_generated else 0.0 for t in trajectory.turns
            ]

        return trajectory

    def get_system_prompt(self, task_data: Dict[str, Any]) -> str:
        """Build system prompt for math tool tasks."""
        tool_desc = self.get_tool_descriptions()
        return (
            "You are a math problem solver. You can use tools to help verify "
            "your calculations.\n\n"
            + tool_desc
            + "\n\nSolve the problem step by step. Use tools when helpful. "
            "Put your final answer in \\boxed{} or use the final_answer format."
        )
