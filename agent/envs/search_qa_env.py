"""
Search-based QA environment for agent RL training.

The agent can search a knowledge base to answer questions.
Supports datasets like HotpotQA, TriviaQA, etc.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from agent.base_env import BaseToolEnv, Trajectory
from agent.base_tool import BaseTool


class SearchQAEnv(BaseToolEnv):
    """
    Environment for training search-augmented QA agents.

    Reward design:
    - Trajectory reward: correctness of the final answer
    - Turn-level reward (optional): quality of search queries
    """

    def __init__(
        self,
        tools: List[BaseTool],
        max_turns: int = 5,
        max_tokens_per_turn: int = 512,
        answer_match_fn=None,
        use_turn_rewards: bool = False,
    ):
        super().__init__(tools, max_turns, max_tokens_per_turn)
        self.answer_match_fn = answer_match_fn or self._default_answer_match
        self.use_turn_rewards = use_turn_rewards

    @staticmethod
    def _default_answer_match(prediction: str, ground_truth: str) -> float:
        """Simple normalized string matching."""
        pred_clean = prediction.strip().lower()
        gt_clean = ground_truth.strip().lower()

        if pred_clean == gt_clean:
            return 1.0
        if gt_clean in pred_clean or pred_clean in gt_clean:
            return 0.5
        return 0.0

    def compute_reward(self, trajectory: Trajectory) -> Trajectory:
        """
        Compute rewards for the trajectory.

        Trajectory reward: final answer correctness.
        Turn rewards: whether each tool call contributed useful information.
        """
        # Trajectory-level reward: correctness
        if trajectory.final_answer and trajectory.ground_truth:
            trajectory.trajectory_reward = self.answer_match_fn(
                trajectory.final_answer, trajectory.ground_truth
            )
        else:
            # No final answer or no ground truth
            trajectory.trajectory_reward = 0.0

        # Per-turn rewards
        if self.use_turn_rewards:
            trajectory.turn_rewards = []
            for turn in trajectory.turns:
                if turn.role == "assistant" and turn.tool_calls:
                    # Give a small positive reward for making tool calls
                    # (encourages exploration)
                    turn_reward = 0.1
                    # Bonus if tool call succeeded
                    for result in turn.tool_results:
                        if (
                            result.success
                            and result.output != "No relevant results found."
                        ):
                            turn_reward += 0.1
                    trajectory.turn_rewards.append(turn_reward)
                elif turn.role == "assistant":
                    # Final answer turn
                    trajectory.turn_rewards.append(trajectory.trajectory_reward)
                else:
                    # Observation turn -- no reward for env-generated content
                    trajectory.turn_rewards.append(0.0)
        else:
            # All turns get the trajectory reward (like standard GRPO)
            trajectory.turn_rewards = [
                trajectory.trajectory_reward if t.is_model_generated else 0.0
                for t in trajectory.turns
            ]

        return trajectory

    def get_system_prompt(self, task_data: Dict[str, Any]) -> str:
        """Build system prompt for search QA tasks."""
        tool_desc = self.get_tool_descriptions()
        return (
            "You are a helpful assistant that can search for information to answer questions.\n\n"
            + tool_desc
            + "\n\nThink step by step. Search for relevant information before answering."
        )
