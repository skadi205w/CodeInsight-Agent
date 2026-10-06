"""命令行入口。

    python -m app.cli index  --path ./target-project   # 对目标代码库建索引
    python -m app.cli search "订单创建后怎么扣库存"      # 纯检索，不调大模型
    python -m app.cli ask    "订单创建后怎么扣库存"      # 完整 Agent 问答
    python -m app.cli stats                            # 查看索引状态
    python -m app.cli serve  --port 8000               # 启动 HTTP 服务
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from app.config import get_settings
from app.runtime import get_runtime


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_index(args: argparse.Namespace) -> int:
    runtime = get_runtime()
    report = runtime.index(args.path, rebuild=args.rebuild)
    print_json(report)
    if report["files_scanned"] == 0:
        print("提示：没有扫描到可索引的文件，确认 --path 指向的是代码目录。", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    runtime = get_runtime()
    hits = runtime.search(
        args.query,
        top_k=args.top_k,
        language=args.language,
        path_prefix=args.path_prefix,
    )
    if not hits:
        print("没有命中任何分块。可以换个说法，或先执行 index 建库。")
        return 1

    for index, hit in enumerate(hits, start=1):
        print(
            f"[{index}] {hit['file_path']}:{hit['start_line']}-{hit['end_line']}  "
            f"score={hit['score']}  symbol={hit['symbol'] or '-'}"
        )
        print("-" * 70)
        print(hit["code"])
        print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    runtime = get_runtime()
    result = asyncio.run(runtime.achat(args.question, session_id=args.session, top_k=args.top_k))

    print("=" * 70)
    print("Agent 推理过程")
    print("=" * 70)
    for step in result["steps"]:
        print(f"\n--- Step {step['index']} ---")
        if step["thought"]:
            print(f"Thought: {step['thought']}")
        print(f"Action: {step['action']}")
        print(f"Action Input: {step['action_input']}")
        print(f"Observation: {step['observation'][:400]}")

    print("\n" + "=" * 70)
    print("最终回答")
    print("=" * 70)
    print(result["answer"])

    if result["references"]:
        print("\n引用位置：")
        for ref in result["references"]:
            print(f"  - {ref['file_path']}:{ref['start_line']}-{ref['end_line']}  {ref['symbol']}")

    print(
        f"\n耗时 {result['elapsed_ms']} ms，"
        f"步数 {len(result['steps'])}，结束原因 {result['stopped_reason']}"
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print_json(get_runtime().stats())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
    return 0


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="CodeInsight Agent —— 代码知识库智能助手（命令行）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("index", help="对代码库建索引")
    p.add_argument("--path", default=None, help="目标代码库目录，留空则用配置里的 REPO_ROOT")
    p.add_argument("--rebuild", action="store_true", help="清空后全量重建（默认增量）")
    p.set_defaults(func=cmd_index)

    p = subparsers.add_parser("search", help="纯检索（不调用大模型，适合调试分块与向量化）")
    p.add_argument("query", help="查询语句")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--language", default=None, choices=["python", "java", "markdown"])
    p.add_argument("--path-prefix", default=None, help="限定目录前缀")
    p.set_defaults(func=cmd_search)

    p = subparsers.add_parser("ask", help="完整 Agent 问答")
    p.add_argument("question", help="你的问题")
    p.add_argument("--session", default="cli", help="会话 ID")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_ask)

    p = subparsers.add_parser("stats", help="查看索引与运行状态")
    p.set_defaults(func=cmd_stats)

    p = subparsers.add_parser("serve", help="启动 HTTP 服务")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())