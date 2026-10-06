"""源码文件扫描：找出所有需要建索引的文件。

只做「找文件」这一件事，不读内容，方便单测与复用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# 扩展名 -> 语言标识，新增语言只要在这里加一行
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".md": "markdown",
}


@dataclass(slots=True)
class SourceFile:
    path: Path
    rel_path: str      # 相对代码库根目录的路径，统一用 / 分隔，便于跨平台与展示
    language: str
    size: int


def detect_language(path: Path) -> str | None:
    return LANGUAGE_BY_EXT.get(path.suffix.lower())


def iter_source_files(
    root: str | Path,
    include_ext: list[str] | None = None,
    exclude_dirs: set[str] | None = None,
    max_file_size_kb: int = 512,
) -> Iterator[SourceFile]:
    """遍历代码库，产出需要建索引的文件。

    - 跳过 exclude_dirs 里的目录（.git / .venv / node_modules 等）
    - 跳过所有以 . 开头的目录
    - 跳过空文件和超过体积上限的文件
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"不是有效目录：{root_path}")

    allowed_ext = {e.lower() for e in include_ext} if include_ext else set(LANGUAGE_BY_EXT)
    excluded = exclude_dirs or set()
    max_bytes = max_file_size_kb * 1024

    for dirpath, dirnames, filenames in os.walk(root_path):
        # 原地修改 dirnames 可以阻止 os.walk 继续往下走，比事后过滤高效得多
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))

        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in allowed_ext:
                continue
            language = detect_language(path)
            if language is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_bytes:
                continue
            yield SourceFile(
                path=path,
                rel_path=path.relative_to(root_path).as_posix(),
                language=language,
                size=size,
            )