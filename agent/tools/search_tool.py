"""
Search tool for multi-hop QA agent training.

This tool wraps a retrieval backend (BM25, Wikipedia API, or custom index)
to provide document search capabilities to the agent.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from agent.base_tool import BaseTool, ToolResult


class SearchTool(BaseTool):
    """
    A search tool that retrieves documents from a knowledge source.

    Supports multiple backends:
    - "mock": Returns predefined results (for testing / offline training)
    - "wikipedia": Queries Wikipedia API
    - "custom": Uses a user-provided retrieval function
    """

    name = "search"
    description = (
        "Search for information from a knowledge base. Returns relevant passages."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "The search query to look up",
            "required": True,
        }
    }

    def __init__(
        self,
        backend: str = "mock",
        corpus: Optional[Dict[str, str]] = None,
        retrieval_fn=None,
        top_k: int = 3,
        max_chars: int = 500,
    ):
        """
        Args:
            backend: One of "mock", "wikipedia", "custom"
            corpus: Dict mapping doc_id -> text (for mock backend)
            retrieval_fn: Callable(query) -> List[str] (for custom backend)
            top_k: Number of results to return
            max_chars: Max characters per result
        """
        self.backend = backend
        self.corpus = corpus or {}
        self.retrieval_fn = retrieval_fn
        self.top_k = top_k
        self.max_chars = max_chars

    def execute(self, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Error: 'query' argument is required.",
            )

        if self.backend == "mock":
            return self._mock_search(query)
        elif self.backend == "wikipedia":
            return self._wikipedia_search(query)
        elif self.backend == "custom":
            return self._custom_search(query)
        else:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Unknown backend: {self.backend}",
            )

    def _mock_search(self, query: str) -> ToolResult:
        """Simple keyword matching against the corpus."""
        query_lower = query.lower()
        scored = []
        for doc_id, text in self.corpus.items():
            score = sum(1 for word in query_lower.split() if word in text.lower())
            if score > 0:
                scored.append((score, doc_id, text))
        scored.sort(reverse=True)

        results = []
        for _, doc_id, text in scored[: self.top_k]:
            snippet = text[: self.max_chars]
            results.append(f"[{doc_id}] {snippet}")

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="No relevant results found.",
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n\n".join(results),
            metadata={"num_results": len(results)},
        )

    def _wikipedia_search(self, query: str) -> ToolResult:
        """Search Wikipedia via its public API."""
        try:
            import urllib.request
            import urllib.parse
            import json as _json

            params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": self.top_k,
                    "format": "json",
                }
            )
            url = f"https://en.wikipedia.org/w/api.php?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read().decode())

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item["title"]
                snippet = item.get("snippet", "")
                # Remove HTML tags from snippet
                import re

                snippet = re.sub(r"<[^>]+>", "", snippet)
                results.append(f"[{title}] {snippet[: self.max_chars]}")

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n\n".join(results) if results else "No results found.",
                metadata={"num_results": len(results)},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Search failed: {str(e)}",
            )

    def _custom_search(self, query: str) -> ToolResult:
        """Use the user-provided retrieval function."""
        if self.retrieval_fn is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Custom backend requires a retrieval_fn.",
            )
        try:
            results = self.retrieval_fn(query)
            if isinstance(results, list):
                output = "\n\n".join(
                    str(r)[: self.max_chars] for r in results[: self.top_k]
                )
            else:
                output = str(results)[: self.max_chars * self.top_k]
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Custom search failed: {str(e)}",
            )
