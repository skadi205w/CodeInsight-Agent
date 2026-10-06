"""代码阅读工具：把指定文件的源码原文取出来，交给大模型去解释。

注意它本身并不"解释"代码——解释是由模型基于返回的原文完成的。
工具只负责「精确地把代码取出来」，这种职责划分是 Agent 设计里很关键的一点：
工具做确定性的事，模型做需要理解和推理的事。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.tools.base import (
    Tool,
    ToolResult,
    as_int,
    resolve_repo_path,
    truncate,
)
from app.rag.splitter import split_file

LANGUAGE_BY_EXT = {".py": "python", ".java": "java", ".md": "markdown"}


class ExplainCodeTool(Tool):
    name = "explain_code"
    description = (
        "读取指定文件的源代码原文（可只读某个函数/类，或某段行号区间）。"
        "读到代码之后，由你来解释逻辑、梳理调用关系或排查问题。"
    )
    parameters = {
        "path": "相对于代码库根目录的文件路径，例如 app/service/order_service.py",
        "symbol": "函数名或类名，可选；给了就只读这个符号的实现",
        "start_line": "起始行号，可选",
        "end_line": "结束行号，可选",
        "max_lines": "最多返回多少行，可选，默认 400",
    }

    def __init__(self, repo_root: str | Path, max_lines: int = 400) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.max_lines = max_lines

    # ------------------------------------------------------------------ #

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        rel_path = str(
            payload.get("path") or payload.get("file") or payload.get("input") or ""
        ).strip()
        if not rel_path:
            return ToolResult(text='缺少 path 参数。正确用法：{"path": "app/main.py"}')

        try:
            target = resolve_repo_path(self.repo_root, rel_path)
        except ValueError as exc:
            return ToolResult(text=str(exc))

        if not target.is_file():
            return ToolResult(text=f"文件不存在：{rel_path}")

        rel = target.relative_to(self.repo_root).as_posix()
        source = target.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()

        symbol = str(payload.get("symbol") or "").strip()
        if symbol:
            return self._by_symbol(rel, source, symbol)

        return self._by_lines(rel, lines, payload)

    # ------------------------------------------------------------------ #

    def _by_symbol(self, rel: str, source: str, symbol: str) -> ToolResult:
        """按符号名取代码：直接复用分块器，保证索引和读取用的是同一套边界。"""
        language = LANGUAGE_BY_EXT.get(Path(rel).suffix.lower(), "text")
        chunks = split_file(rel, source, language)

        matched = [c for c in chunks if c.metadata.get("symbol") == symbol]
        if not matched:
            matched = [c for c in chunks if symbol in str(c.metadata.get("symbol", ""))]

        if not matched:
            names = sorted({str(c.metadata.get("symbol")) for c in chunks if c.metadata.get("symbol")})
            preview = "、".join(names[:20]) or "(未识别到符号)"
            return ToolResult(text=f"{rel} 中未找到符号 {symbol!r}。该文件包含的符号：{preview}")

        # 类骨架块只有签名，优先返回方法/函数的实现块
        matched.sort(key=lambda c: (c.metadata.get("symbol_type") == "class", c.start_line))
        chunk = matched[0]
        header = (
            f"{rel}:{chunk.start_line}-{chunk.end_line}  "
            f"[{chunk.metadata.get('symbol_type')}] {chunk.metadata.get('symbol')}\n"
        )
        return ToolResult(
            text=truncate(header + chunk.text, self.max_lines * 120),
            references=[
                {
                    "file_path": rel,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol": str(chunk.metadata.get("symbol", "")),
                    "symbol_type": str(chunk.metadata.get("symbol_type", "")),
                    "score": 1.0,
                }
            ],
        )

    def _by_lines(self, rel: str, lines: list[str], payload: dict[str, Any]) -> ToolResult:
        total = len(lines)
        start = max(1, as_int(payload.get("start_line"), 1))
        end = as_int(payload.get("end_line"), total)
        end = min(total, max(start, end))

        limit = max(1, as_int(payload.get("max_lines"), self.max_lines))
        if end - start + 1 > limit:
            end = start + limit - 1

        body = "\n".join(lines[start - 1 : end])
        text = f"{rel}:{start}-{end}（该文件共 {total} 行）\n\n{body}"
        return ToolResult(
            text=truncate(text, limit * 120),
            references=[
                {
                    "file_path": rel,
                    "start_line": start,
                    "end_line": end,
                    "symbol": "",
                    "symbol_type": "range",
                    "score": 1.0,
                }
            ],
        )