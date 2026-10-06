"""HTTP 接口层。

接口设计上刻意把「检索」和「对话」分开：
- /search 不经过 LLM，纯检索。调分块策略、调 Embedding 效果时，
  直接打这个接口看召回结果，比每次都跑一遍大模型快得多；
- /chat 走完整 Agent 流程（多步推理 + 工具调用）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.runtime import get_runtime

router = APIRouter()


# --------------------------------------------------------------------------- #
# 请求体
# --------------------------------------------------------------------------- #


class IndexRequest(BaseModel):
    path: str | None = Field(default=None, description="目标代码库目录；留空则用配置里的 REPO_ROOT")
    rebuild: bool = Field(default=False, description="true=清空重建（全量）；false=增量索引")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID，用于多轮对话")
    top_k: int | None = Field(default=None, ge=1, le=50, description="本轮检索条数，可选")


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #


@router.post("/index", summary="对代码库建索引")
async def build_index(payload: IndexRequest) -> dict[str, Any]:
    runtime = get_runtime()
    try:
        # 建库是 CPU 密集 + 阻塞 IO，丢到线程池，别把事件循环堵死
        report = await asyncio.to_thread(runtime.index, payload.path, payload.rebuild)
    except (NotADirectoryError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:                          # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"建索引失败：{exc}") from exc
    return {"ok": True, "report": report}


@router.post("/chat", summary="与 Agent 对话（完整 ReAct 流程）")
async def chat(payload: ChatRequest) -> dict[str, Any]:
    runtime = get_runtime()
    try:
        return await runtime.achat(payload.question, payload.session_id, payload.top_k)
    except Exception as exc:                          # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{exc}") from exc


@router.get("/search", summary="纯检索（不经过 LLM）")
async def search(
    q: str = Query(..., min_length=1, description="查询语句"),
    top_k: int = Query(default=5, ge=1, le=50),
    language: str | None = Query(default=None, description="限定语言：python / java / markdown"),
    path_prefix: str | None = Query(default=None, description="限定目录前缀"),
) -> dict[str, Any]:
    runtime = get_runtime()
    hits = await asyncio.to_thread(runtime.search, q, top_k, language, path_prefix)
    return {"query": q, "count": len(hits), "hits": hits}


@router.get("/tools", summary="列出 Agent 可用的工具")
async def list_tools() -> dict[str, Any]:
    return {"tools": get_runtime().describe_tools()}


@router.get("/stats", summary="索引与运行状态")
async def stats() -> dict[str, Any]:
    return await asyncio.to_thread(get_runtime().stats)


@router.get("/health", summary="健康检查（Ollama + 向量库）")
async def health() -> dict[str, Any]:
    return await get_runtime().health()


@router.delete("/session/{session_id}", summary="清空某个会话的上下文")
async def clear_session(session_id: str) -> dict[str, Any]:
    get_runtime().reset_session(session_id)
    return {"ok": True, "session_id": session_id}