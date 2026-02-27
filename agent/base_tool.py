"""
Base tool abstraction for agent RL training.

Every tool the agent can call should inherit from BaseTool and implement
the `execute` method. The framework handles parsing tool calls from the
model's output and dispatching them to the appropriate tool.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result returned by a tool after execution."""

    tool_name: str
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for tools that the agent can invoke.

    Subclass this and implement `execute()` to create a custom tool.
    The tool name and description are used to generate the system prompt
    that describes available tools to the model.

    Example:
        class SearchTool(BaseTool):
            name = "search"
            description = "Search the web for information."
            parameters = {
                "query": {"type": "string", "description": "The search query"}
            }

            def execute(self, query: str, **kwargs) -> ToolResult:
                results = my_search_api(query)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=results
                )
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments parsed from the model's output.

        Returns:
            ToolResult with the tool output.
        """
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool schema for the system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def reset(self):
        """Reset tool state between episodes. Override if stateful."""
        pass
