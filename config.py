"""全局配置：优先读环境变量，其次读项目根目录下的 .env。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 服务 ----------------
    app_name: str = "CodeInsight Agent"
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------------- LLM ----------------
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "deepseek-r1:7b"
    llm_temperature: float = 0.1
    # 本地大模型首个 token 可能要等很久，超时给足
    llm_timeout: float = 300.0

    # ---------------- Embedding ----------------
    # sentence_transformers：默认，本地跑 bge-small-zh
    # ollama：走 Ollama 的 /api/embed，省掉 torch 依赖
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_batch_size: int = 32

    # ---------------- 向量库 ----------------
    chroma_dir: str = "./data/chroma"
    collection_name: str = "code_chunks"

    # ---------------- 检索 ----------------
    recall_n: int = 20          # 先从向量库粗召回
    top_k: int = 5              # 再截断后交给 LLM

    # ---------------- Agent ----------------
    max_agent_steps: int = 8
    allow_write: bool = False   # 是否允许工具直接改写源文件（默认只给 diff 预览）

    # ---------------- 代码库 ----------------
    # 被索引、被 Agent 读取的目标代码库根目录（调用 /index 时会动态覆盖它）
    repo_root: str = "."

    # ---------------- 索引 ----------------
    include_ext: str = ".py,.java,.md"
    exclude_dirs: str = (
        ".git,.idea,.venv,venv,__pycache__,node_modules,dist,build,"
        "target,.pytest_cache,.mypy_cache,.ruff_cache,site-packages"
    )
    max_file_size_kb: int = 512
    max_chunk_lines: int = 120

    @property
    def include_ext_list(self) -> list[str]:
        return [e.strip().lower() for e in self.include_ext.split(",") if e.strip()]

    @property
    def exclude_dir_set(self) -> set[str]:
        return {d.strip() for d in self.exclude_dirs.split(",") if d.strip()}

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()