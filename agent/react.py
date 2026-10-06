"""手写 ReAct 推理循环。

为什么不用 LangChain 的 AgentExecutor？
它把「拼提示词 -> 解析模型输出 -> 调工具 -> 回灌结果 -> 循环」全封装了，
用起来三行代码；但被追问「Agent 怎么决定调哪个工具」「工具报错怎么办」
「怎么防止无限循环」时就容易答不上来。这里把每一步都摊开写，逻辑完全可控。

四个关键设计：
1. 工具描述即提示词：模型完全靠 tool.spec() 决定何时调用哪个工具；
2. 异常不中断：工具报错也作为 Observation 回灌，让模型自己决定重试还是换工具；
3. 步数上限：防止模型在两个工具之间来回死循环；
4. 全程留痕：每个 Step 都记录 thought / action / observation，随响应返回，
   既是前端演示「思考过程」的数据，也是排错时最有用的东西。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.agent.prompt import (
    ParsedOutput,
    build_system_prompt,
    parse_agent_output,
)
from app.agent.tools.base import Tool, ToolResult


@dataclass(slots=True)
class AgentStep:
    index: int
    thought: str
    action: str
    action_input: str
    observation: str
    references: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "references": self.references,
        }


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: int = 0
    stopped_reason: str = "final_answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "references": self.references,
            "elapsed_ms": self.elapsed_ms,
            "stopped_reason": self.stopped_reason,
        }


def format_scratchpad_entry(step: AgentStep) -> str:
    """把一轮结果拼回提示词，作为下一轮的上下文。"""
    lines = []
    if step.thought:
        lines.append(f"Thought: {step.thought}")
    lines.append(f"Action: {step.action}")
    lines.append(f"Action Input: {step.action_input}")
    lines.append(f"Observation: {step.observation}")
    return "\n".join(lines) + "\n"


def _dedupe_references(references: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    for ref in references:
        key = (ref.get("file_path"), ref.get("start_line"), ref.get("end_line"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


class ReActAgent:
    def __init__(
        self,
        llm: Any,
        tools: Sequence[Tool],
        max_steps: int = 8,
        temperature: float = 0.1,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tools}
        self.max_steps = max(1, max_steps)
        self.temperature = temperature
        self.system_prompt = build_system_prompt(list(self.tools.values()))

    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        question: str,
        history: list[dict[str, str]],
        scratchpad: str,
    ) -> list[dict[str, str]]:
        content = f"Question: {question}"
        if scratchpad:
            content += "\n\n" + scratchpad

        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": content})
        return messages

    async def _invoke_tool(self, tool: Tool, raw_input: str) -> ToolResult:
        """工具大多是同步阻塞的（读文件、查向量库），丢到线程池里跑。"""
        try:
            return await asyncio.to_thread(tool.run, raw_input)
        except Exception as exc:                      # noqa: BLE001 - 工具异常必须兜住
            return ToolResult(text=f"工具执行失败：{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #

    async def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        started = time.perf_counter()
        history = history or []

        scratchpad = ""
        steps: list[AgentStep] = []
        all_references: list[dict[str, Any]] = []

        def finish(answer: str, reason: str) -> AgentResult:
            return AgentResult(
                answer=answer,
                steps=steps,
                references=_dedupe_references(all_references),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                stopped_reason=reason,
            )

        for step_index in range(1, self.max_steps + 1):
            messages = self._build_messages(question, history, scratchpad)

            try:
                raw = await self.llm.chat(
                    messages,
                    temperature=self.temperature,
                    # 防止模型自己编造 Observation
                    stop=["\nObservation:", "\nObservation：", "\n观察:"],
                )
            except Exception as exc:                  # noqa: BLE001
                return finish(f"调用大模型失败：{exc}", "llm_error")

            parsed: ParsedOutput = parse_agent_output(raw)

            # ① 模型给出了最终答案，收工
            if parsed.is_final:
                return finish(parsed.final_answer, "final_answer")

            # ② 模型没按格式输出，把原文当答案返回，避免空转
            if not parsed.has_action:
                return finish(raw.strip() or "模型没有返回有效内容。", "no_action")

            # ③ 执行工具
            tool = self.tools.get(parsed.action)
            if tool is None:
                result = ToolResult(
                    text=f"工具 {parsed.action!r} 不存在。可用工具：{', '.join(self.tools) or '无'}"
                )
            else:
                result = await self._invoke_tool(tool, parsed.action_input)

            step = AgentStep(
                index=step_index,
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input,
                observation=result.text,
                references=result.references,
            )
            steps.append(step)
            all_references.extend(result.references)
            scratchpad += format_scratchpad_entry(step)

        return finish(
            "已达到最大推理步数，仍未得出最终答案。可以换个更具体的问法，"
            "或调大环境变量 MAX_AGENT_STEPS。",
            "max_steps",
        )