"""工具集：Agent 能调用的全部能力，统一在这里注册。"""

from __future__ import annotations

from pathlib import Path

from app.agent.tools.base import Tool, ToolResult, resolve_repo_path
from app.agent.tools.code_search import CodeSearchTool
from app.agent.tools.explain_code import ExplainCodeTool
from app.agent.tools.symbol_lookup import SymbolLookupTool
from app.agent.tools.write_comment import WriteCommentTool

__all__ = [
    "Tool",
    "ToolResult",
    "CodeSearchTool",
    "ExplainCodeTool",
    "SymbolLookupTool",
    "WriteCommentTool",
    "build_tools",
    "tool_descriptions",
]


def build_tools(retriever, repo_root: str | Path, settings) -> list[Tool]:
    """按配置装配工具集。新增工具只要在这里加一行。"""
    return [
        CodeSearchTool(retriever, default_top_k=settings.top_k),
        ExplainCodeTool(repo_root),
        SymbolLookupTool(
            repo_root,
            include_ext=settings.include_ext_list,
            exclude_dirs=settings.exclude_dir_set,
            max_file_size_kb=settings.max_file_size_kb,
        ),
        WriteCommentTool(repo_root, allow_write=settings.allow_write),
    ]


def tool_descriptions(tools: list[Tool]) -> list[dict[str, str]]:
    """给接口返回用的工具清单，前端可以据此展示「Agent 有哪些能力」。"""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": "、".join(tool.parameters.keys()),
        }
        for tool in tools
    ]