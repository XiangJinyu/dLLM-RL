"""
Calculator tool for agent RL training.

A simple calculator that evaluates mathematical expressions safely.
"""

from __future__ import annotations
import math
import re
from agent.base_tool import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """
    Evaluate mathematical expressions safely.
    Supports basic arithmetic, powers, and common math functions.
    """

    name = "calculator"
    description = "Evaluate a mathematical expression and return the result."
    parameters = {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'log(100, 10)')",
            "required": True,
        }
    }

    # Whitelist of allowed names in eval
    SAFE_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
        # math module functions
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "comb": math.comb,
        "perm": math.perm,
    }

    def execute(self, expression: str = "", **kwargs) -> ToolResult:
        if not expression:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Error: 'expression' argument is required.",
            )

        # Basic safety check: reject anything that looks like code injection
        if any(
            kw in expression
            for kw in ["import", "exec", "eval", "open", "__", "os.", "sys."]
        ):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Error: Expression contains disallowed keywords.",
            )

        try:
            result = eval(expression, {"__builtins__": {}}, self.SAFE_NAMES)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=str(result),
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Error evaluating expression: {str(e)}",
            )
