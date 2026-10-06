"""FastAPI 应用入口。

启动：
    uvicorn app.main:app --reload --port 8000

接口文档：
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.runtime import get_runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 启动时什么都不做：Embedding 模型和向量库都是按需惰性加载的，
    # 这样即使 Ollama 没起、模型没装，服务本身也能正常启动并给出可读的错误。
    yield
    await get_runtime().aclose()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="读取本地代码库构建知识库，并用自研 ReAct Agent 回答代码问题的服务。",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", summary="根路径")
async def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "hint": "先 POST /index 建库，再 POST /chat 提问",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)