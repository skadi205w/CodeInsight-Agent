"""Chroma 向量库封装。

选 Chroma 的理由：本地文件持久化，不用额外起一个服务，pip 装完就能用，
对个人项目和小团队来说是启动成本最低的方案。

需要知道的两个细节：
1. 元数据只支持字符串/数字/布尔，None 必须先过滤掉，否则写入会报错；
2. 元数据过滤算子有限（$eq/$in/$gt 这类），所以「按路径前缀过滤」放在检索器里用 Python 做。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.splitter import CodeChunk


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "code_chunks") -> None:
        path = Path(persist_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(path)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._create_collection()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _create_collection(self):
        # cosine 距离配合归一化向量，score = 1 - distance 就是余弦相似度
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def make_id(chunk: CodeChunk) -> str:
        raw = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}:{chunk.metadata.get('symbol', '')}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"{chunk.file_path}#{chunk.start_line}-{chunk.end_line}#{digest}"

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Chroma 只接受 str / int / float / bool，这里统一过滤。"""
        cleaned: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: Iterable[CodeChunk], embeddings: list[list[float]]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        seen: set[str] = set()

        for chunk in chunk_list:
            chunk_id = self.make_id(chunk)
            if chunk_id in seen:      # 同批次出现重复 id，Chroma 会直接报错
                continue
            seen.add(chunk_id)
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._clean_metadata(chunk.metadata))

        if not ids:
            return 0

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings[: len(ids)],
        )
        return len(ids)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def query(
        self,
        embedding: list[float],
        n_results: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        total = self.count()
        if total == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            hits.append(
                {
                    "text": document or "",
                    "metadata": dict(metadata or {}),
                    "distance": float(distance),
                    "score": max(0.0, 1.0 - float(distance)),
                }
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #

    def file_hashes(self) -> dict[str, str]:
        """{相对路径: 内容哈希}，用于增量索引时判断文件有没有变过。"""
        total = self.count()
        if total == 0:
            return {}
        data = self._collection.get(include=["metadatas"], limit=total)
        result: dict[str, str] = {}
        for metadata in data.get("metadatas") or []:
            if not metadata:
                continue
            file_path = metadata.get("file_path")
            content_hash = metadata.get("content_hash")
            if file_path and content_hash:
                result[str(file_path)] = str(content_hash)
        return result

    def delete_by_file(self, rel_path: str) -> None:
        try:
            self._collection.delete(where={"file_path": rel_path})
        except Exception:       # 删一个不存在的文件不应该中断整个建库流程
            pass

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._create_collection()

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "chunks": self.count(),
            "files": len(self.file_hashes()),
        }