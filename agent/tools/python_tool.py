"""
Python code execution tool for agent RL training.

Allows the agent to write and execute Python code as part of its
reasoning process (e.g., for math verification, data analysis).
Execution is sandboxed with timeouts.
"""

from __future__ import annotations
import multiprocessing
import traceback
from typing import Any, Dict
from agent.base_tool import BaseTool, ToolResult


def _execute_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute Python code in a subprocess with timeout."""
    import io
    import sys

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    result = {"success": False, "stdout": "", "stderr": "", "return_value": None}

    try:
        # Create a restricted globals dict
        restricted_globals = {"__builtins__": __builtins__}
        exec(code, restricted_globals)
        result["success"] = True
        result["stdout"] = sys.stdout.getvalue()
        result["stderr"] = sys.stderr.getvalue()
    except Exception as e:
        result["stderr"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return result


def _run_in_process(code: str, result_dict: dict, timeout: int):
    """Run code execution in a separate process."""
    result = _execute_code(code, timeout)
    result_dict.update(result)


class PythonExecuteTool(BaseTool):
    """
    Execute Python code and return the output.

    The code is run in a sandboxed subprocess with a timeout to prevent
    infinite loops or resource exhaustion.
    """

    name = "python"
    description = (
        "Execute Python code and return the output. "
        "Use this for calculations, data processing, or verification. "
        "The code should use print() to produce output."
    )
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
            "required": True,
        }
    }

    def __init__(self, timeout: int = 10, max_output_chars: int = 2000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str = "", **kwargs) -> ToolResult:
        if not code:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Error: 'code' argument is required.",
            )

        try:
            manager = multiprocessing.Manager()
            result_dict = manager.dict()

            proc = multiprocessing.Process(
                target=_run_in_process,
                args=(code, result_dict, self.timeout),
            )
            proc.start()
            proc.join(timeout=self.timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=f"Execution timed out after {self.timeout} seconds.",
                )

            result = dict(result_dict)

            output_parts = []
            if result.get("stdout"):
                output_parts.append(result["stdout"])
            if result.get("stderr") and not result.get("success"):
                output_parts.append(f"Error:\n{result['stderr']}")

            output = "\n".join(output_parts) if output_parts else "(no output)"
            output = output[: self.max_output_chars]

            return ToolResult(
                tool_name=self.name,
                success=result.get("success", False),
                output=output,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Failed to execute code: {str(e)}",
            )
