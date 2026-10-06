"""工具基类。

所有工具统一走「模板方法」：
    run(raw_input) -> parse_input() -> execute(payload) -> ToolResult

子类只需要实现 execute()，输入解析的兼容逻辑放在基类统一处理。
理由：LLM 输出的 Action Input 是「自然语言里夹 JSON」，不可能 100% 规整，
把「尽力解析」收敛到一处，工具本身只关心业务逻辑，新增工具的成本就很低。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.prompt import extract_json_payload


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。

    text       ：给模型看的文本，会作为 Observation 回灌进提示词
    references ：结构化引用（文件 + 行号），用于接口返回与前端跳转
    """

    text: str
    references: list[dict[str, Any]] = field(default_factory=list)


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 子类实现
    # ------------------------------------------------------------------ #

    @abstractmethod
    def execute(self, payload: dict[str, Any]) -> ToolResult:
        """真正干活的地方。"""

    # ------------------------------------------------------------------ #
    # 基类统一处理
    # ------------------------------------------------------------------ #

    def parse_input(self, raw_input: str) -> dict[str, Any]:
        payload = extract_json_payload(raw_input)
        if payload is not None:
            return payload

        # 不是 JSON：当成「第一个声明的参数」的字符串值
        text = (raw_input or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
            text = text[1:-1]
        first_param = next(iter(self.parameters), "input")
        return {first_param: text} if text else {}

    def run(self, raw_input: str) -> ToolResult:
        try:
            payload = self.parse_input(raw_input)
        except Exception as exc:                      # noqa: BLE001
            return ToolResult(text=f"{self.name} 参数解析失败：{exc}")
        return self.execute(payload)

    # ------------------------------------------------------------------ #
    # 提示词渲染
    # ------------------------------------------------------------------ #

    def spec(self) -> str:
        params = "、".join(f"{k}（{v}）" for k, v in self.parameters.items()) or "无参数"
        return f"- {self.name}：{self.description}\n  参数：{params}"


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "是", "真"}
    if value is None:
        return default
    return bool(value)


def indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def truncate(text: str, limit: int = 6000) -> str:
    """工具输出会直接进提示词，必须限长，否则一次检索就能把上下文撑爆。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（输出过长已截断，原长 {len(text)} 字符）"


def resolve_repo_path(repo_root: str | Path, rel_path: str) -> Path:
    """把相对路径解析到代码库根目录下，并阻止 ../ 越界访问。"""
    root = Path(repo_root).expanduser().resolve()
    candidate = (root / str(rel_path).replace("\\", "/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"路径越界，拒绝访问：{rel_path}")
    return candidate