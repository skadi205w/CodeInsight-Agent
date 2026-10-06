"""代码检索工具 —— Agent 最常用的一个。

它本身不"理解"代码，只负责从向量库把最相关的片段捞出来并整理成模型能读的文本；
真正的解释和推理由大模型基于这些片段完成。
"""

from __future__ import annotations

from typing import Any

from app.agent.tools.base import Tool, ToolResult, as_int, indent_block, truncate
from app.rag.retriever import Retriever


class CodeSearchTool(Tool):
    name = "code_search"
    description = (
        "在代码知识库中做语义检索：输入一句自然语言描述，返回最相关的代码片段、"
        "文件路径与行号。当你不确定某个功能写在哪、叫什么名字时，先用这个工具。"
    )
    parameters = {
        "query": "自然语言查询，例如「订单创建后如何扣减库存」",
        "top_k": "返回条数，可选，默认 5",
        "language": "限定语言，可选：python / java / markdown",
        "path_prefix": "限定目录前缀，可选，例如 app/service",
    }

    def __init__(self, retriever: Retriever, default_top_k: int = 5) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k

    def execute(self, payload: dict[str, Any]) -> ToolResult:
        query = str(
            payload.get("query") or payload.get("q") or payload.get("input") or ""
        ).strip()
        if not query:
            return ToolResult(text='缺少 query 参数。正确用法：{"query": "想找的代码功能"}')

        top_k = as_int(payload.get("top_k"), self.default_top_k)
        language = payload.get("language") or None
        path_prefix = payload.get("path_prefix") or None

        hits = self.retriever.search(
            query,
            top_k=top_k,
            language=language,
            path_prefix=path_prefix,
        )
        if not hits:
            return ToolResult(
                text=f"没有检索到与「{query}」相关的代码。可以换个说法，或去掉 language / path_prefix 限制再试。"
            )

        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            meta = hit.metadata
            blocks.append(
                f"[{index}] {hit.location}  相似度={hit.score:.3f}\n"
                f"    符号：{meta.get('symbol') or '(未识别)'}  "
                f"类型：{meta.get('symbol_type') or '-'}\n"
                f"{indent_block(hit.text)}"
            )

        text = (
            f"与「{query}」最相关的 {len(hits)} 个代码片段：\n\n" + "\n\n".join(blocks)
        )
        return ToolResult(
            text=truncate(text),
            references=[hit.to_reference() for hit in hits],
        )