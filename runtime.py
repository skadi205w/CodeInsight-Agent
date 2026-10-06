"""运行时容器：把 LLM、Embedding、向量库、检索器、工具、Agent 组装到一起。

做成单例的原因是构建成本很高：
- bge-small-zh 加载一次要几秒、占几百 MB 内存；
- Chroma 的 PersistentClient 也不适合反复创建。

另外这里维护一份进程内的会话记忆（session_id -> 消息列表）。
注意这是内存态、重启即丢；要持久化应该换成 Redis 之类的外部存储。
"""

from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.agent.react import ReActAgent
from app.agent.tools import Tool, build_tools, tool_descriptions
from app.config import Settings, get_settings
from app.llm.ollama_client import OllamaClient, OllamaError
from app.rag.embedder import BaseEmbedder, build_embedder
from app.rag.retriever import Retriever
from app.rag.vectorstore import VectorStore

# 单会话最多保留多少条历史消息，防止上下文无限膨胀
MAX_HISTORY_MESSAGES = 20


class Runtime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # 用可重入锁：tools / retriever / agent 都是懒加载，
        # 构建 tools 的过程中还会去取 retriever（取 retriever 又要取 embedder），
        # 普通 Lock 在这种嵌套获取时会自我死锁。
        self._lock = threading.RLock()

        self._embedder: BaseEmbedder | None = None
        self._store: VectorStore | None = None
        self._retriever: Retriever | None = None
        self._llm: OllamaClient | None = None
        self._tools: list[Tool] | None = None
        self._agent: ReActAgent | None = None

        self.repo_root: Path = Path(self.settings.repo_root).expanduser().resolve()
        self._tools_root: Path | None = None
        self.sessions: dict[str, list[dict[str, str]]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # 惰性构建
    # ------------------------------------------------------------------ #

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = VectorStore(
                        self.settings.chroma_path,
                        self.settings.collection_name,
                    )
        return self._store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = Retriever(self.embedder, self.store, self.settings)
        return self._retriever

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    self._llm = OllamaClient(
                        base_url=self.settings.ollama_base_url,
                        default_model=self.settings.llm_model,
                        timeout=self.settings.llm_timeout,
                        temperature=self.settings.llm_temperature,
                    )
        return self._llm

    @property
    def tools(self) -> list[Tool]:
        if self._tools is None or self._tools_root != self.repo_root:
            with self._lock:
                self._tools = build_tools(self.retriever, self.repo_root, self.settings)
                self._tools_root = self.repo_root
                self._agent = None      # 工具变了，Agent 的系统提示词也要重建
        return self._tools

    @property
    def agent(self) -> ReActAgent:
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._agent = ReActAgent(
                        llm=self.llm,
                        tools=self.tools,
                        max_steps=self.settings.max_agent_steps,
                        temperature=self.settings.llm_temperature,
                    )
        return self._agent

    # ------------------------------------------------------------------ #
    # 业务入口（同步实现，HTTP 层用 to_thread 包起来）
    # ------------------------------------------------------------------ #

    def set_repo_root(self, path: str | Path) -> Path:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"不是有效目录：{root}")
        self.repo_root = root
        return root

    def index(self, path: str | Path | None = None, rebuild: bool = False) -> dict[str, Any]:
        root = self.set_repo_root(path) if path else self.repo_root
        report = self.retriever.index_directory(root, rebuild=rebuild)
        # 目标库变了，工具要重建，否则 explain_code 还指向老目录
        self._tools = None
        self._agent = None
        self.tools
        return report

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        hits = self.retriever.search(
            query,
            top_k=top_k,
            language=language,
            path_prefix=path_prefix,
        )
        return [
            {
                "file_path": hit.metadata.get("file_path", ""),
                "start_line": hit.metadata.get("start_line", 0),
                "end_line": hit.metadata.get("end_line", 0),
                "symbol": hit.metadata.get("symbol", ""),
                "symbol_type": hit.metadata.get("symbol_type", ""),
                "language": hit.metadata.get("language", ""),
                "score": round(hit.score, 4),
                "code": hit.text,
            }
            for hit in hits
        ]

    async def achat(self, question: str, session_id: str = "default", top_k: int | None = None) -> dict[str, Any]:
        if top_k:
            # 临时覆盖本轮检索条数，不改全局配置
            original = self.settings.top_k
            self.settings.top_k = top_k
            try:
                return await self._run_agent(question, session_id)
            finally:
                self.settings.top_k = original
        return await self._run_agent(question, session_id)

    async def _run_agent(self, question: str, session_id: str) -> dict[str, Any]:
        history = self.sessions.get(session_id, [])
        result = await self.agent.run(question, history=history)

        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        ]
        self.sessions[session_id] = history[-MAX_HISTORY_MESSAGES:]

        payload = result.to_dict()
        payload["session_id"] = session_id
        return payload

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ------------------------------------------------------------------ #

    def describe_tools(self) -> list[dict[str, str]]:
        return tool_descriptions(self.tools)

    def stats(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "llm_model": self.settings.llm_model,
            "embedding_backend": self.settings.embedding_backend,
            "embedding_model": self.settings.embedding_model,
            "vector_store": self.store.stats(),
            "tools": [tool.name for tool in self.tools],
            "sessions": len(self.sessions),
        }

    async def health(self) -> dict[str, Any]:
        ollama_ok = await self.llm.is_available()
        models: list[str] = []
        error = ""
        if ollama_ok:
            try:
                models = await self.llm.list_models()
            except OllamaError as exc:
                error = str(exc)

        try:
            chunks = self.store.count()
            store_ok = True
        except Exception as exc:                     # noqa: BLE001
            chunks, store_ok, error = 0, False, str(exc)

        llm_model = self.settings.llm_model
        return {
            "status": "ok" if (ollama_ok and store_ok) else "degraded",
            "ollama": {"available": ollama_ok, "base_url": self.settings.ollama_base_url,
                       "model": llm_model, "model_ready": any(m.startswith(llm_model.split(":")[0]) for m in models),
                       "models": models[:20], "error": error},
            "vector_store": {"ok": store_ok, "chunks": chunks},
        }

    async def aclose(self) -> None:
        if self._llm is not None:
            await self._llm.aclose()


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime