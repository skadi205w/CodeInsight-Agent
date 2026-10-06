"""ReAct 提示词模板与输出解析。

这是本项目最核心、面试最容易被追问的部分：Agent 到底是怎么「思考」的。

链路是：
    拼提示词 -> 让模型输出 Thought / Action -> 解析 -> 真正调用工具 ->
    把结果作为 Observation 拼回提示词 -> 再问模型 -> 直到它给出 Final Answer

模型输出是不可信的：可能少写冒号、用中文冒号、把参数写成字符串、
或者干脆用 `Action: code_search("xxx")` 这种调用式写法。所以这里的解析器
做了多层兜底，这也是「自己写 Agent」比「套框架」更值得讲的地方。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT_TEMPLATE = """你是一个代码助手，可以调用工具来阅读、检索一个代码仓库。

你可以使用的工具（每轮只能用一个）：

__TOOL_SPECS__

请严格按下面的格式回答：

Question: 用户的问题
Thought: 你的思考——现在缺什么信息、为什么选这个工具
Action: 工具名（必须是上面列表里的名字）
Action Input: 工具参数的 JSON 对象，例如 {"query": "扣减库存", "top_k": 5}

系统会执行工具，并把结果作为 Observation 返回给你；然后你继续下一轮 Thought / Action，
可以连续调用多个工具。

当你已经掌握足够信息时，用下面的格式结束：

Thought: 我已经掌握了足够信息
Final Answer: 你的最终回答

硬性要求：
1. 所有关于代码的结论必须来自工具返回的内容，不允许凭空编造文件名、函数名或行号；
2. 一次只输出一个 Action，不要自己虚构 Observation；
3. Action Input 必须是合法 JSON，不要附加解释文字；
4. 最终回答用中文，引用代码时写明「文件路径:行号」。
"""


def build_system_prompt(tools: list[Any]) -> str:
    specs = "\n".join(tool.spec() for tool in tools) or "（当前没有任何可用工具）"
    return SYSTEM_PROMPT_TEMPLATE.replace("__TOOL_SPECS__", specs)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

_FINAL_RE = re.compile(
    r"(?:Final\s*Answer|最终答案|最终回答)\s*[:：]\s*(?P<answer>.+)",
    re.IGNORECASE | re.DOTALL,
)

_THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(?P<thought>.+?)"
    r"(?=(?:\n\s*(?:Action|Final\s*Answer|Observation)\s*[:：])|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_ACTION_RE = re.compile(
    r"(?:^|\n)\s*Action\s*[:：]\s*(?P<action>[A-Za-z_][A-Za-z0-9_\-]*)",
    re.IGNORECASE,
)

_ACTION_INPUT_RE = re.compile(
    r"(?:^|\n)\s*Action\s*Input\s*[:：]\s*(?P<input>.+?)"
    r"(?=(?:\n\s*(?:Thought|Observation|Action)\s*[:：])|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# 兜底：模型写成 Action: code_search({"query": "x"}) 这种函数调用样式
_CALL_STYLE_RE = re.compile(
    r"(?:^|\n)\s*Action\s*[:：]\s*(?P<action>[A-Za-z_][A-Za-z0-9_\-]*)\s*[（(](?P<input>.*)[）)]\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class ParsedOutput:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer.strip())

    @property
    def has_action(self) -> bool:
        return bool(self.action.strip())


def strip_fences(text: str) -> str:
    """模型整体用 ``` 包起来时，把围栏去掉。"""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_agent_output(text: str) -> ParsedOutput:
    cleaned = strip_fences(text)
    if not cleaned:
        return ParsedOutput()

    parsed = ParsedOutput()

    thought_match = _THOUGHT_RE.search(cleaned)
    if thought_match:
        parsed.thought = thought_match.group("thought").strip()

    final_match = _FINAL_RE.search(cleaned)
    if final_match:
        parsed.final_answer = final_match.group("answer").strip()
        return parsed

    # 先判断是不是 Action: tool({"k": "v"}) 这种函数调用写法。
    # 必须放在普通 Action 之前，否则正则只能匹配到工具名、参数会被丢掉。
    call_match = _CALL_STYLE_RE.search(cleaned)
    if call_match:
        parsed.action = call_match.group("action").strip()
        parsed.action_input = call_match.group("input").strip()
        return parsed

    action_match = _ACTION_RE.search(cleaned)
    if action_match:
        parsed.action = action_match.group("action").strip()
        input_match = _ACTION_INPUT_RE.search(cleaned)
        if input_match:
            parsed.action_input = input_match.group("input").strip()

    return parsed


def extract_json_payload(raw: str) -> dict[str, Any] | None:
    """尽最大努力从 Action Input 里解析出 JSON 对象。"""
    if not raw:
        return None
    candidate = strip_fences(raw)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None