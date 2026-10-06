"""写回源码的工具（生成注释后落盘）。

默认是「只预览、不改文件」：返回 unified diff 供确认。
要真正写盘必须同时满足两个条件：
  1. 服务端配置 ALLOW_WRITE=true（环境变量，默认关闭）
  2. 本次调用显式传 apply=true

这是有意设计的：让大模型直接改源码风险很高（改错行、丢掉缩进、覆盖业务逻辑），
所以留一道人工闸门。这个「写操作要二次确认」的设计点，面试时很好展开。
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from app.agent.tools.base import (
    Tool,
    ToolResult,
    as_bool,
    as_int,
    resolve_repo_path,
    truncate,
)


class WriteCommentTool(Tool):
    name = "write_comment"
    description = (
        "把带注释的新代码写回源文件。默认只返回 diff 预览、不修改文件；"
        "只有服务端允许写入（ALLOW_WRITE=true）且显式传 apply=true 时才会真正落盘。"
    )
    parameters = {
        "path": "相对于代码库根目录的文件路径",
        "start_line": "要被替换的起始行号",
        "end_line": "要被替换的结束行号",
        "new_code": "替换后的完整代码块（含你写好的注释，注意保留原缩进）",
        "apply": "是否真正写入，可选，默认 false（只预览 diff）",
    }

    def __init__(self, repo_root: str | Path, allow_write: bool = False) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.allow_write = allow_write

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        rel_path = str(payload.get("path") or payload.get("file") or "").strip()
        new_code = payload.get("new_code") or payload.get("code") or ""

        if not rel_path or not str(new_code).strip():
            return ToolResult(
                text=(
                    "参数不足。正确用法："
                    '{"path": "app/service/a.py", "start_line": 12, "end_line": 20, '
                    '"new_code": "带注释的完整代码块"}'
                )
            )

        try:
            target = resolve_repo_path(self.repo_root, rel_path)
        except ValueError as exc:
            return ToolResult(text=str(exc))

        if not target.is_file():
            return ToolResult(text=f"文件不存在：{rel_path}")

        raw = target.read_bytes()
        newline = "\r\n" if b"\r\n" in raw else "\n"
        original = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = original.split("\n")

        start = max(1, as_int(payload.get("start_line"), 1))
        end = min(len(lines), max(start, as_int(payload.get("end_line"), start)))
        new_lines = str(new_code).replace("\r\n", "\n").split("\n")

        updated = lines[: start - 1] + new_lines + lines[end:]
        diff = "\n".join(
            difflib.unified_diff(
                lines,
                updated,
                fromfile=f"{rel_path}（原文件）",
                tofile=f"{rel_path}（修改后）",
                lineterm="",
                n=3,
            )
        )
        reference = {
            "file_path": target.relative_to(self.repo_root).as_posix(),
            "start_line": start,
            "end_line": start + len(new_lines) - 1,
            "symbol": "",
            "symbol_type": "edit",
            "score": 1.0,
        }

        apply_now = as_bool(payload.get("apply"), False)

        if apply_now and self.allow_write:
            target.write_bytes(newline.join(updated).encode("utf-8"))
            return ToolResult(
                text=f"已写入 {rel_path}（第 {start}-{end} 行被替换）。\n\n{diff}",
                references=[reference],
            )

        if apply_now and not self.allow_write:
            return ToolResult(
                text=(
                    "服务端未开启写入（ALLOW_WRITE=false），已退回 diff 预览，文件未被修改。"
                    f"\n\n{diff}"
                ),
                references=[reference],
            )

        return ToolResult(
            text=(
                "以下是 diff 预览（文件未被修改）。确认无误后，"
                f"带 apply=true 再调用一次即可落盘：\n\n{truncate(diff, 8000)}"
            ),
            references=[reference],
        )