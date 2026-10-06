"""符号精确查找。

语义检索擅长「按意思找」，但对「我知道它叫 deduct_stock，就想知道定义在哪」
这种精确定位，反而不如关键词搜索靠谱。

所以给 Agent 配了两个互补的检索工具：
- code_search    按语义找，适合「描述得出、但不知道名字」
- symbol_lookup  按名字找，适合「知道名字、要精确定位定义与引用」

这种「工具之间职责互补」的设计，也是面试时可以拿出来讲的点。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agent.tools.base import Tool, ToolResult, as_int, truncate
from app.rag.loader import iter_source_files

# 各语言的「定义」形态：命中这些模式的才算定义，其余算引用
DEFINITION_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+(?P<name>\w+)\s*\("),                       # Python 函数
    re.compile(r"^\s*class\s+(?P<name>\w+)\s*[:(]"),                                # Python 类
    re.compile(r"^\s*(?:public|protected|private|static|final|abstract|sealed|"
               r"strictfp|\s)*(?:class|interface|enum|record)\s+(?P<name>\w+)"),    # Java 类型
    re.compile(r"^\s*(?:(?:public|protected|private|static|final|abstract|"
               r"synchronized|native|default|strictfp)\s+)*"
               r"[\w$<>\[\],\.\?]+\s+(?P<name>\w+)\s*\("),                          # Java 方法
    re.compile(r"^#{1,6}\s+(?P<name>.+?)\s*$"),                                     # Markdown 标题
)


def is_definition(line: str, symbol: str) -> bool:
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(line)
        if match and match.group("name").strip() == symbol:
            return True
    return False


class SymbolLookupTool(Tool):
    name = "symbol_lookup"
    description = (
        "按符号名（函数名 / 类名 / 方法名）在代码库里精确查找定义位置与引用位置。"
        "已经知道名字、需要精确定位时用这个，比语义检索更准。"
    )
    parameters = {
        "symbol_name": "要查找的符号名，例如 deduct_stock",
        "path_prefix": "限定目录前缀，可选，例如 app/service",
        "max_results": "最多返回多少条，可选，默认 30",
    }

    def __init__(
        self,
        repo_root: str | Path,
        include_ext: list[str] | None = None,
        exclude_dirs: set[str] | None = None,
        max_file_size_kb: int = 512,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.include_ext = include_ext
        self.exclude_dirs = exclude_dirs
        self.max_file_size_kb = max_file_size_kb

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        symbol = str(
            payload.get("symbol_name")
            or payload.get("symbol")
            or payload.get("name")
            or payload.get("input")
            or ""
        ).strip()
        if not symbol:
            return ToolResult(text='缺少 symbol_name 参数。正确用法：{"symbol_name": "deduct_stock"}')

        max_results = max(1, as_int(payload.get("max_results"), 30))
        path_prefix = str(payload.get("path_prefix") or "").replace("\\", "/").lstrip("./")

        word_pattern = re.compile(r"(?<![\w.])" + re.escape(symbol) + r"(?![\w])")
        definitions: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []

        for source in iter_source_files(
            self.repo_root,
            include_ext=self.include_ext,
            exclude_dirs=self.exclude_dirs,
            max_file_size_kb=self.max_file_size_kb,
        ):
            if path_prefix and not source.rel_path.startswith(path_prefix):
                continue

            try:
                text = source.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), start=1):
                if symbol not in line:          # 便宜的前置过滤，避免每行都跑正则
                    continue
                snippet = line.strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + " ..."

                entry = {"file_path": source.rel_path, "line": lineno, "text": snippet}
                if is_definition(line, symbol):
                    definitions.append(entry)
                elif word_pattern.search(line):
                    references.append(entry)

            if len(definitions) >= max_results and len(references) >= max_results:
                break

        if not definitions and not references:
            return ToolResult(text=f"代码库里没有找到符号 {symbol!r}。可以先用 code_search 做一次语义检索。")

        parts: list[str] = []
        if definitions:
            parts.append(f"定义位置（{len(definitions)} 处）：")
            for item in definitions[:max_results]:
                parts.append(f"  - {item['file_path']}:{item['line']}\n      {item['text']}")
        if references:
            parts.append(f"\n引用位置（共 {len(references)} 处，最多列出 {max_results} 处）：")
            for item in references[:max_results]:
                parts.append(f"  - {item['file_path']}:{item['line']}\n      {item['text']}")

        refs = [
            {
                "file_path": item["file_path"],
                "start_line": item["line"],
                "end_line": item["line"],
                "symbol": symbol,
                "symbol_type": "definition",
                "score": 1.0,
            }
            for item in definitions[:max_results]
        ]
        return ToolResult(text=truncate("\n".join(parts)), references=refs)