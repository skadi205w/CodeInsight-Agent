"""代码专用分块（code-aware chunking）——本项目与通用 RAG 最大的区别。

通用 RAG 常按固定字符数切分（比如每 500 字一段），放到代码场景会出大问题：
一个函数被拦腰截断，检索出来的片段既看不懂、也跑不通。

本模块改为按语法结构切分：

============  ==================================================================
文件类型       切分依据
============  ==================================================================
.py           用标准库 ast 解析，以函数 / 类 / 方法为边界；类会额外产出一个
              「骨架块」（类声明 + 文档字符串 + 方法签名清单 + 类属性）
.java         轻量语法扫描：先把注释和字符串字面量抹成空白，再按声明正则定位、
              用花括号配对找边界；以类 / 方法 / 字段区为边界
.md           按标题层级切分，并记录所属章节路径
============  ==================================================================

每个分块都会带上 file_path / language / start_line / end_line / symbol /
symbol_type / parent 等元数据，用途有两个：
1. 检索时按语言或目录过滤；
2. 回答时把结论溯源到「文件:行号」，降低大模型幻觉。
"""

from __future__ import annotations

import ast
import bisect
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CodeChunk:
    """一个可被检索的代码分块。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start_line(self) -> int:
        return int(self.metadata.get("start_line", 0))

    @property
    def end_line(self) -> int:
        return int(self.metadata.get("end_line", 0))

    @property
    def file_path(self) -> str:
        return str(self.metadata.get("file_path", ""))

    @property
    def location(self) -> str:
        """人类可读的位置描述，例如 app/service/a.py:12-30。"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def make_metadata(
    rel_path: str,
    language: str,
    start_line: int,
    end_line: int,
    symbol: str = "",
    symbol_type: str = "",
    parent: str = "",
    **extra: Any,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "file_path": rel_path,
        "language": language,
        "start_line": int(start_line),
        "end_line": int(end_line),
        "symbol": symbol or "",
        "symbol_type": symbol_type or "",
        "parent": parent or "",
    }
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


def split_file(
    rel_path: str,
    source: str,
    language: str,
    max_chunk_lines: int = 120,
) -> list[CodeChunk]:
    """按语言选择分块策略，统一入口。"""
    if language == "python":
        chunks = split_python(source, rel_path, max_chunk_lines)
    elif language == "java":
        chunks = split_java(source, rel_path, max_chunk_lines)
    elif language == "markdown":
        chunks = split_markdown(source, rel_path, max_chunk_lines)
    else:
        chunks = split_by_lines(source, rel_path, language, max_chunk_lines)
    return [c for c in chunks if c.text and c.text.strip()]


# --------------------------------------------------------------------------- #
# 公共小工具
# --------------------------------------------------------------------------- #


def _slice(lines: list[str], start_line: int, end_line: int) -> str:
    """按 1-based 闭区间取行。越界自动收敛，保证不会抛异常。"""
    start = max(0, start_line - 1)
    stop = max(start, min(len(lines), end_line))
    return "\n".join(lines[start:stop])


def split_by_lines(
    source: str,
    rel_path: str,
    language: str,
    max_chunk_lines: int = 120,
) -> list[CodeChunk]:
    """兜底策略：解析失败或未知语言时，按固定行数切。"""
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for i in range(0, max(1, len(lines)), max_chunk_lines):
        part = lines[i : i + max_chunk_lines]
        chunks.append(
            CodeChunk(
                "\n".join(part),
                make_metadata(rel_path, language, i + 1, i + len(part), symbol_type="block"),
            )
        )
    return chunks


def _split_oversize(chunks: list[CodeChunk], max_chunk_lines: int) -> list[CodeChunk]:
    """超长分块再按行切开，并保留原有元数据（行号、符号名会同步调整）。"""
    out: list[CodeChunk] = []
    for chunk in chunks:
        lines = chunk.text.split("\n")
        if len(lines) <= max_chunk_lines:
            out.append(chunk)
            continue
        base = chunk.start_line
        symbol = chunk.metadata.get("symbol") or ""
        for part_no, i in enumerate(range(0, len(lines), max_chunk_lines), start=1):
            part = lines[i : i + max_chunk_lines]
            meta = dict(chunk.metadata)
            meta["start_line"] = base + i
            meta["end_line"] = base + i + len(part) - 1
            if symbol:
                meta["symbol"] = f"{symbol} (part {part_no})"
            out.append(CodeChunk("\n".join(part), meta))
    return out


# --------------------------------------------------------------------------- #
# Python：基于 ast 的结构化切分
# --------------------------------------------------------------------------- #

_PY_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _py_node_start(node: ast.AST) -> int:
    """节点的起始行；带装饰器时把装饰器也算进去。"""
    start = getattr(node, "lineno", 0)
    for deco in getattr(node, "decorator_list", []) or []:
        start = min(start, getattr(deco, "lineno", start))
    return start


def _py_node_end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


def _py_signature(node: ast.AST, lines: list[str]) -> str:
    """把可能跨多行的函数签名压成一行，用于类骨架块。"""
    collected: list[str] = []
    depth = 0
    for idx in range(node.lineno - 1, min(len(lines), node.lineno + 20)):
        line = lines[idx]
        collected.append(line.strip())
        depth += line.count("(") - line.count(")")
        if depth <= 0 and line.rstrip().endswith(":"):
            break
    return " ".join(collected)


def _is_docstring_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _py_class_skeleton(node: ast.ClassDef, lines: list[str]) -> str:
    """类骨架：类声明 + 文档字符串 + 属性 + 方法签名清单。

    这样检索时既能知道「这个类负责什么、有哪些能力」，
    又不会把几百行实现塞进同一个分块。
    """
    start = _py_node_start(node)
    first_body_line = node.body[0].lineno if node.body else node.lineno
    head = _slice(lines, start, max(start, first_body_line - 1))
    parts = [head.rstrip()]

    doc = ast.get_docstring(node, clean=False)
    if doc:
        parts.append('    """' + doc.strip() + '"""')

    attributes = [n for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for attr in attributes:
        parts.append("    " + _slice(lines, _py_node_start(attr), _py_node_end(attr)).strip())

    methods = [n for n in node.body if isinstance(n, _PY_FUNC_TYPES)]
    if methods:
        parts.append("    # ---- 方法列表（实现见各自的分块）----")
        for method in methods:
            parts.append(f"    {_py_signature(method, lines)}  # L{method.lineno}")

    return "\n".join(parts)


def split_python(source: str, rel_path: str, max_chunk_lines: int = 120) -> list[CodeChunk]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # 语法错误的文件（比如模板、片段）退回按行切分，保证索引不中断
        return split_by_lines(source, rel_path, "python", max_chunk_lines)

    chunks: list[CodeChunk] = []

    # 1) 文件头：模块 docstring + 所有 import。检索时经常需要确认依赖来源
    header_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_docstring_node(node):
            header_end = max(header_end, _py_node_end(node))
    if header_end:
        chunks.append(
            CodeChunk(
                _slice(lines, 1, header_end),
                make_metadata(rel_path, "python", 1, header_end, symbol="<module-header>", symbol_type="header"),
            )
        )

    # 2) 顶层节点逐个成块
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_docstring_node(node):
            continue

        start, end = _py_node_start(node), _py_node_end(node)

        if isinstance(node, ast.ClassDef):
            chunks.append(
                CodeChunk(
                    _py_class_skeleton(node, lines),
                    make_metadata(rel_path, "python", start, end, symbol=node.name, symbol_type="class"),
                )
            )
            for child in node.body:
                if isinstance(child, _PY_FUNC_TYPES):
                    c_start, c_end = _py_node_start(child), _py_node_end(child)
                    chunks.append(
                        CodeChunk(
                            _slice(lines, c_start, c_end),
                            make_metadata(
                                rel_path, "python", c_start, c_end,
                                symbol=f"{node.name}.{child.name}", symbol_type="method", parent=node.name,
                            ),
                        )
                    )
        elif isinstance(node, _PY_FUNC_TYPES):
            chunks.append(
                CodeChunk(
                    _slice(lines, start, end),
                    make_metadata(rel_path, "python", start, end, symbol=node.name, symbol_type="function"),
                )
            )
        else:
            chunks.append(
                CodeChunk(
                    _slice(lines, start, end),
                    make_metadata(rel_path, "python", start, end, symbol_type="statement"),
                )
            )

    return _split_oversize(chunks, max_chunk_lines)


# --------------------------------------------------------------------------- #
# Java：轻量语法扫描
# --------------------------------------------------------------------------- #

_JAVA_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "try", "do", "else", "return",
    "new", "case", "assert", "super", "this", "synchronized", "throw", "throws",
}

_JAVA_TYPE_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:public|protected|private|static|final|abstract|sealed|strictfp)\s+)*"
    r"(?P<kind>@interface|class|interface|enum|record)\s+(?P<name>\w+)"
)

_JAVA_METHOD_RE = re.compile(
    r"(?m)^[ \t]*(?P<prefix>(?:(?:public|protected|private|static|final|abstract|"
    r"synchronized|native|default|strictfp)[ \t]+)*)"
    r"(?P<ret>[\w$<>\[\],\.\? \t]*?)[ \t]*(?P<name>\w+)[ \t]*\((?P<params>[^()]*)\)[ \t]*"
    r"(?:throws[ \t]+[\w,\.$ \t]+?)?\{"
)


def _blank_noise(source: str) -> str:
    """把注释和字符串字面量替换成等长空白（保留换行）。

    这样后续可以直接用正则做结构扫描，而不会被注释里的花括号、
    或者字符串里的 "if (x) {" 之类的文本带偏。
    """
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if ch == "/" and nxt == "/":                      # 行注释
            j = source.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif ch == "/" and nxt == "*":                    # 块注释 / javadoc
            j = source.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in source[i:j]))
            i = j
        elif ch in "\"'":                                 # 字符串 / 字符字面量
            quote = ch
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == quote:
                    j += 1
                    break
                if source[j] == "\n":                     # 未闭合，别把整篇吃掉
                    break
                j += 1
            out.append("".join("\n" if c == "\n" else " " for c in source[i:j]))
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _brace_pairs(clean: str) -> dict[int, int]:
    """返回 {左花括号位置: 右花括号位置}。"""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for idx, ch in enumerate(clean):
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            pairs[stack.pop()] = idx
    return pairs


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _line_of(starts: list[int], pos: int) -> int:
    """字符偏移 -> 1-based 行号。"""
    return bisect.bisect_right(starts, pos)


def _extend_start_back(lines: list[str], index: int) -> int:
    """从声明所在行往上，把紧邻的注解和注释也纳进来。"""
    i = index - 1
    while i >= 0:
        stripped = lines[i].strip()
        if (
            stripped.startswith("@")
            or stripped.startswith("//")
            or stripped.startswith("*")
            or stripped.startswith("/*")
            or stripped.endswith("*/")
        ):
            i -= 1
            continue
        break
    return max(0, i + 1)


def split_java(source: str, rel_path: str, max_chunk_lines: int = 120) -> list[CodeChunk]:
    lines = source.splitlines()
    clean = _blank_noise(source)
    starts = _line_starts(source)
    pairs = _brace_pairs(clean)

    # ---- 1. 找类 / 接口 / 枚举 ----
    types: list[dict[str, Any]] = []
    for match in _JAVA_TYPE_RE.finditer(clean):
        brace = clean.find("{", match.end())
        if brace == -1 or brace not in pairs:
            continue
        decl_line = _line_of(starts, match.start())
        types.append(
            {
                "name": match.group("name"),
                "kind": match.group("kind"),
                "open": brace,
                "close": pairs[brace],
                "decl_line": decl_line,
                "start_line": _extend_start_back(lines, decl_line - 1) + 1,
                "end_line": _line_of(starts, pairs[brace]),
            }
        )

    # ---- 2. 找方法 ----
    methods: list[dict[str, Any]] = []
    for match in _JAVA_METHOD_RE.finditer(clean):
        name = match.group("name")
        if name in _JAVA_KEYWORDS:
            continue
        brace = match.end() - 1          # 正则末尾就是 "{"
        if brace not in pairs:
            continue
        decl_line = _line_of(starts, match.start())
        methods.append(
            {
                "name": name,
                "params": " ".join(match.group("params").split()),
                "open": brace,
                "close": pairs[brace],
                "decl_line": decl_line,
                "start_line": _extend_start_back(lines, decl_line - 1) + 1,
                "end_line": _line_of(starts, pairs[brace]),
            }
        )

    if not types and not methods:
        return split_by_lines(source, rel_path, "java", max_chunk_lines)

    # ---- 3. 把方法归属到最内层的类 ----
    grouped: dict[int, list[dict[str, Any]]] = {id(t): [] for t in types}
    orphans: list[dict[str, Any]] = []
    for method in methods:
        host: dict[str, Any] | None = None
        for t in types:
            if t["open"] < method["open"] < t["close"]:
                if host is None or t["open"] > host["open"]:
                    host = t
        if host is None:
            orphans.append(method)
        else:
            grouped[id(host)].append(method)

    chunks: list[CodeChunk] = []

    for t in types:
        own = sorted(grouped[id(t)], key=lambda m: m["decl_line"])

        # 3.1 类骨架：声明 + 方法签名清单
        header_end_line = _line_of(starts, t["open"])
        parts = [_slice(lines, t["start_line"], header_end_line).rstrip()]
        if own:
            parts.append("")
            parts.append("    // ---- 方法列表（实现见各自分块）----")
            for m in own:
                signature = lines[m["decl_line"] - 1].strip()
                parts.append(f"    {signature}  # L{m['decl_line']}")
        chunks.append(
            CodeChunk(
                "\n".join(parts),
                make_metadata(
                    rel_path, "java", t["start_line"], header_end_line,
                    symbol=t["name"], symbol_type=t["kind"],
                ),
            )
        )

        # 3.2 字段区：类体开头到第一个方法之前
        body_start = header_end_line + 1
        first_method_line = own[0]["start_line"] if own else None
        fields_end = (first_method_line - 1) if first_method_line else (t["end_line"] - 1)
        if fields_end >= body_start:
            field_text = _slice(lines, body_start, fields_end)
            if field_text.strip():
                chunks.append(
                    CodeChunk(
                        field_text,
                        make_metadata(
                            rel_path, "java", body_start, fields_end,
                            symbol=f"{t['name']}<fields>", symbol_type="fields", parent=t["name"],
                        ),
                    )
                )

        # 3.3 方法本体
        for m in own:
            chunks.append(
                CodeChunk(
                    _slice(lines, m["start_line"], m["end_line"]),
                    make_metadata(
                        rel_path, "java", m["start_line"], m["end_line"],
                        symbol=f"{t['name']}.{m['name']}", symbol_type="method", parent=t["name"],
                    ),
                )
            )

    # ---- 4. 不在任何类里的方法（静态工具类之外的场景） ----
    for m in orphans:
        chunks.append(
            CodeChunk(
                _slice(lines, m["start_line"], m["end_line"]),
                make_metadata(
                    rel_path, "java", m["start_line"], m["end_line"],
                    symbol=m["name"], symbol_type="method",
                ),
            )
        )

    return _split_oversize(chunks, max_chunk_lines)


# --------------------------------------------------------------------------- #
# Markdown：按标题层级切分
# --------------------------------------------------------------------------- #

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.*?)\s*#*\s*$")


def split_markdown(source: str, rel_path: str, max_chunk_lines: int = 120) -> list[CodeChunk]:
    lines = source.splitlines()

    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        match = _MD_HEADING_RE.match(line)
        if match:
            headings.append((idx, len(match.group(1)), match.group("title").strip()))

    if not headings:
        return split_by_lines(source, rel_path, "markdown", max_chunk_lines)

    chunks: list[CodeChunk] = []

    # 第一个标题之前的内容（前言）
    if headings[0][0] > 0:
        preface = _slice(lines, 1, headings[0][0])
        if preface.strip():
            chunks.append(
                CodeChunk(
                    preface,
                    make_metadata(rel_path, "markdown", 1, headings[0][0], symbol_type="preface"),
                )
            )

    for pos, (idx, level, title) in enumerate(headings):
        # 一直取到下一个标题（不管层级）：这样各分块互不重叠，
        # 父章节不会把子章节的内容整段重复一遍，避免检索时出现重复召回
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)

        parents = [t for _, lv, t in headings[:pos] if lv < level]
        chunks.append(
            CodeChunk(
                _slice(lines, idx + 1, end),
                make_metadata(
                    rel_path, "markdown", idx + 1, max(idx + 1, end),
                    symbol=title, symbol_type=f"h{level}", parent=" > ".join(parents),
                ),
            )
        )

    return _split_oversize(chunks, max_chunk_lines)