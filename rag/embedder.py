"""Embedding 封装。

默认用 sentence-transformers 在本地跑 bge-small-zh（中文注释、中文提问场景效果好，
模型只有 100 MB 左右，纯 CPU 也能跑）；也可以切到 Ollama 的 /api/embed，
这样连 torch 都不用装。

一个容易踩的坑：bge 中文系列在**检索场景**下，query 侧需要加一句指令前缀，
文档侧不加。官方说明这样能明显拉开相关与不相关文本的距离，这里按这个规则实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# bge-zh 系列官方建议的检索指令前缀，只加在 query 上
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BaseEmbedder(ABC):
    """向量化接口。

    抽成接口是为了让后端可替换：本地模型 / Ollama / 以后换成在线 API 都不用改上层。
    """

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档侧向量化（不加指令前缀）。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """查询侧向量化（加指令前缀）。"""


class SentenceTransformerEmbedder(BaseEmbedder):
    """默认后端：本地跑 bge-small-zh。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        # 延迟导入：没装 sentence-transformers 时，只要不用这个后端就依然能启动
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = dim
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # 归一化后，余弦相似度可以直接用内积算
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [BGE_ZH_QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].tolist()


class OllamaEmbedder(BaseEmbedder):
    """备选后端：直接用 Ollama 提供的 embedding 接口，跳过 torch。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        dim: int = 0,
        timeout: float = 120.0,
    ) -> None:
        import httpx

        self.model = model
        self.name = f"ollama:{model}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.dim = dim or len(self.embed_query("维度探测"))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/api/embed", json={"model": self.model, "input": texts})
        if response.status_code == 404:
            # 兼容老版本 Ollama：只有 /api/embeddings，且一次只能传一个
            vectors = []
            for text in texts:
                single = self._client.post("/api/embeddings", json={"model": self.model, "prompt": text})
                single.raise_for_status()
                vectors.append(single.json()["embedding"])
            return vectors
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def build_embedder(settings) -> BaseEmbedder:
    """按配置创建 Embedding 后端。"""
    backend = (settings.embedding_backend or "").strip().lower()

    if backend in {"ollama",}:
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    if backend in {"sentence_transformers", "sentence-transformers", "st", "local"}:
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(
        f"不认识的 EMBEDDING_BACKEND：{settings.embedding_backend!r}，"
        "可选值：sentence_transformers / ollama"
    )