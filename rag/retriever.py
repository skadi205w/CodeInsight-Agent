"""检索器：把「建库」和「查询」两件事串起来。

职责边界：
- 建库：遍历源码 -> 代码分块 -> 向量化 -> 写入 Chroma（支持增量）
- 查询：问题向量化 -> 向量召回 -> 元数据/路径过滤 -> 截断成 Top-K
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag.embedder import BaseEmbedder
from app.rag.loader import iter_source_files
from app.rag.splitter import split_file
from app.rag.vectorstore import VectorStore


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def location(self) -> str:
        meta = self.metadata
        return f"{meta.get('file_path', '?')}:{meta.get('start_line', 0)}-{meta.get('end_line', 0)}"

    def to_reference(self) -> dict[str, Any]:
        meta = self.metadata
        return {
            "file_path": meta.get("file_path", ""),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "symbol": meta.get("symbol", ""),
            "symbol_type": meta.get("symbol_type", ""),
            "score": round(self.score, 4),
        }


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: VectorStore, settings) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings

    # ------------------------------------------------------------------ #
    # 建库
    # ------------------------------------------------------------------ #

    def index_directory(self, root: str | Path, rebuild: bool = False) -> dict[str, Any]:
        """对指定目录建索引。

        rebuild=False 时做增量：对比文件内容哈希，只有新增或改动过的文件才会重新索引。
        """
        started = time.time()
        root_path = Path(root).expanduser().resolve()

        if rebuild:
            self.store.reset()
        known_hashes = self.store.file_hashes()

        report: dict[str, Any] = {
            "root": str(root_path),
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "errors": [],
            "elapsed_sec": 0.0,
        }

        pending_chunks = []
        pending_texts: list[str] = []
        batch_size = max(1, self.settings.embedding_batch_size)

        def flush() -> None:
            nonlocal pending_chunks, pending_texts
            if not pending_chunks:
                return
            vectors = self.embedder.embed_documents(pending_texts)
            report["chunks_added"] += self.store.add_chunks(pending_chunks, vectors)
            pending_chunks = []
            pending_texts = []

        for source in iter_source_files(
            root_path,
            include_ext=self.settings.include_ext_list,
            exclude_dirs=self.settings.exclude_dir_set,
            max_file_size_kb=self.settings.max_file_size_kb,
        ):
            report["files_scanned"] += 1

            try:
                text = source.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                report["errors"].append(f"{source.rel_path}: 读取失败 {exc}")
                continue

            digest = content_hash(text)
            if known_hashes.get(source.rel_path) == digest:
                report["files_skipped"] += 1
                continue

            chunks = split_file(
                source.rel_path,
                text,
                source.language,
                max_chunk_lines=self.settings.max_chunk_lines,
            )
            if not chunks:
                continue

            # 重新索引前先清掉这个文件的旧分块，避免函数删掉之后残留孤儿分块
            self.store.delete_by_file(source.rel_path)

            for chunk in chunks:
                chunk.metadata["content_hash"] = digest
                pending_chunks.append(chunk)
                pending_texts.append(chunk.text)

            report["files_indexed"] += 1
            if len(pending_chunks) >= batch_size:
                flush()

        flush()

        report["elapsed_sec"] = round(time.time() - started, 2)
        report["total_chunks"] = self.store.count()
        report["total_files"] = len(self.store.file_hashes())
        return report

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
        path_prefix: str | None = None,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []

        k = max(1, top_k or self.settings.top_k)
        # 先多召回一些，再做过滤，避免过滤之后不够 K 条
        recall_n = max(self.settings.recall_n, k * 3 if path_prefix else k)

        vector = self.embedder.embed_query(query)
        raw_hits = self.store.query(vector, n_results=recall_n, where=self._build_where(language))

        results = [RetrievedChunk(h["text"], h["metadata"], float(h["score"])) for h in raw_hits]

        if path_prefix:
            # Chroma 的元数据算子不支持前缀匹配，所以放到 Python 侧过滤
            prefix = path_prefix.replace("\\", "/").lstrip("./")
            results = [r for r in results if str(r.metadata.get("file_path", "")).startswith(prefix)]

        return results[:k]

    @staticmethod
    def _build_where(language: str | None) -> dict[str, Any] | None:
        if not language:
            return None
        return {"language": language.strip().lower()}