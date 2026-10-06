# CodeInsight Agent —— 代码知识库智能助手

> 把整个代码仓库变成可检索的知识库，用自然语言问「这段代码在哪、干什么、有没有坑」。

## 1. 项目简介

面向**本地代码仓库**的 RAG + Agent 问答系统。它读取项目中的 Python / Java / Markdown 文件，按语法结构切块并向量化；之后你用中文提问，它会先检索出真实相关的代码片段，再给出**带文件路径和行号**的回答。LLM 与向量库全部本地运行，代码不出本机，私有仓库也能用。

**解决什么问题**

| 痛点 | 常见做法 | 本项目 |
| --- | --- | --- |
| 接手陌生项目，调用链一长就迷路 | Ctrl+F + 肉眼翻文件 | 全库语义索引，一句话定位到函数 |
| 通用大模型不懂你的代码 | 直接问，得到幻觉答案 | 先检索真实代码再作答，结论可溯源 |
| 补全类工具只管「写」 | Copilot 解决不了「读懂」 | 专注代码理解：解释、查 Bug、补注释 |

**核心功能**：代码库索引（支持增量）· 语义检索 · 代码解释 · Bug 排查 · 自动注释 · 多轮追问

**技术选型**

| 模块 | 选型 | 理由 |
| --- | --- | --- |
| 后端 | FastAPI | 异步、简洁、自带 Swagger 文档 |
| LLM | Ollama + qwen2.5:7b | 全程本地推理；指令跟随稳定，能可靠输出 ReAct 格式 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文语义效果好，约 100 MB，纯 CPU 可跑 |
| 向量库 | Chroma | 本地文件持久化，不需要额外起服务 |
| Agent | 手写 ReAct 循环 | 不套壳，推理流程完全可控、可解释 |

> 本项目**没有使用 LangChain**，分块、检索、Agent 循环均为自行实现——这也是能在面试中讲清每一处设计取舍的前提。

## 2. 系统架构

```
用户提问 ──POST /chat──▶ FastAPI 接口层
                              │
                              ▼
                  Agent 调度器（手写 ReAct 循环）
                     ▲ Thought / Action │ Observation
                     └──────────────────┘
                              │
                              ▼
                       工具集 Toolset
      code_search · explain_code · symbol_lookup · write_comment
                              │
                              ▼
                         RAG 检索模块
            向量化 → Chroma 相似度检索 → 元数据过滤 → Top-K
                              │
                              ▼
                  Chroma 向量库（本地文件持久化）
                              ▲
                              │ 建库 / 增量更新
                        索引模块 Indexer
              扫描 py/java/md → 代码专用分块 → 写元数据
```

**一次请求的完整链路**

1. 用户 `POST /chat` 提交问题；
2. 调度器把「问题 + 工具说明 + 历史对话」组装成 ReAct Prompt，交给 LLM；
3. LLM 输出 `Thought` 与 `Action`，调度器解析后调用对应工具；
4. 检索类工具进入 RAG 模块，向量化 → Chroma 检索 → 元数据过滤 → 返回 Top-K 代码片段；
5. 结果作为 `Observation` 回灌给 LLM，回到第 3 步继续循环；
6. LLM 输出 `Final Answer`，调度器整理答案与**引用的文件 / 行号**返回。

## 3. 快速开始

**前置条件**：Python 3.10+（建议 3.11）、[Ollama](https://ollama.com/download)、内存 16 GB 以上（跑 7B 模型）

```bash
# ① 拉取模型
ollama pull qwen2.5:7b

# ② 创建环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

# ③ 配置：复制 .env.example 为 .env，按需修改 LLM_MODEL / CHROMA_DIR 等
```

> 国内网络下 `huggingface.co` 可能连不上，首次运行（会下载 Embedding 模型）前先设镜像：
> Windows `$env:HF_ENDPOINT='https://hf-mirror.com'`，macOS / Linux `export HF_ENDPOINT=https://hf-mirror.com`

```bash
# ④ 对目标代码库建索引（--path 指向你想被「读懂」的项目）
python -m app.cli index --path ./your-target-project

# ⑤ 启动服务
python -m app.cli serve --port 8000
```

启动后访问 **http://127.0.0.1:8000/docs** 查看自动生成的接口文档。

命令行还提供 `search`（纯检索，不调用 LLM，秒级返回，调分块策略时最常用）和 `ask`（完整 Agent 问答）两个子命令。

> 注意 `CHROMA_DIR=./data/chroma` 是相对路径，**所有命令都要在项目根目录执行**。

## 4. 核心模块

### 4.1 RAG 模块

**代码专用分块**——本项目与通用 RAG 最大的区别。通用 RAG 按固定字符数切分，会把一个函数拦腰截断，检索出来的片段既看不懂也跑不通。本项目按语法结构切分：

| 文件类型 | 切分依据 |
| --- | --- |
| `.py` | `ast` 解析，以函数 / 类 / 方法为最小单元，保留装饰器与顶部 import |
| `.java` | 正则匹配类 / 接口 / 枚举声明，配合括号配对定位方法边界 |
| `.md` | 按 `#` / `##` 标题层级分块 |
| 其他 | 按固定行数兜底；超长分块再二次切分 |

每个分块附带一组元数据，供检索过滤、供回答溯源：

```python
{
    "file_path": "app/service/stock_service.py",
    "start_line": 12, "end_line": 40,
    "language": "python",
    "symbol": "StockService.deduct_stock", "symbol_type": "method",
}
```

**检索流程**

```
Query → 向量化 → Chroma 相似度检索（Top-N）→ 元数据过滤（语言 / 目录）→ 截断 Top-K → 附带路径与行号返回
```

向量库使用 cosine 距离，返回的 `score = 1 - distance`，即余弦相似度（0~1，越大越相关）。

### 4.2 Agent 工作流

**为什么自己写 ReAct 循环**：LangChain 的 `AgentExecutor` 把「拼提示词 → 解析输出 → 调工具 → 回灌 → 循环」全部封装了，用起来三行代码，但被追问「怎么判断该调哪个工具」「工具调用失败怎么办」「怎么防止无限循环」时容易答不上来。自己实现一遍，这些问题就都变成了明确的设计决策。

```python
def run(self, question, history):
    scratchpad = ""
    for step in range(self.max_steps):        # ① 步数上限，防死循环
        output = self.llm.generate(build_react_prompt(question, self.tools, history, scratchpad))
        if "Final Answer:" in output:         # ② 终止条件
            return parse_final_answer(output)
        thought, action, action_input = parse(output)
        if action not in self.tools:          # ③ 幻觉工具名兜底
            scratchpad += f"Observation: 工具 {action} 不存在，可用：{list(self.tools)}\n"
            continue
        try:
            observation = self.tools[action].run(action_input)
        except Exception as e:
            observation = f"工具执行失败：{e}"  # ④ 异常也回灌给模型
        scratchpad += f"Thought: {thought}\nAction: {action}\nObservation: {observation}\n"
    return "已达到最大推理步数，未能得出最终答案。"
```

**四个关键设计**

- **工具描述即提示词**：Agent 完全靠每个工具的 `description` 决定何时调用，描述写得准不准直接决定效果。
- **异常不中断**：工具报错也作为 Observation 回灌，让模型自己决定重试还是换工具，而不是整个请求失败。
- **步数上限**：`MAX_AGENT_STEPS`（默认 8）兜底，防止模型在两个工具之间来回死循环。
- **可观测**：完整 scratchpad 随响应返回，可以展示 Agent 的思考过程，也是排错时最有用的东西。

**工具清单**

| 工具 | 作用 | 关键参数 |
| --- | --- | --- |
| `code_search` | 语义检索代码片段 | `query`、`top_k`、`language` |
| `explain_code` | 读取并解释指定文件 / 函数 | `path`、`lines` |
| `symbol_lookup` | 按符号名精确查找定义与引用 | `symbol_name` |
| `write_comment` | 生成注释，默认只返回 diff 预览 | `path`、`lines`、`content` |

### 4.3 后端接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/index` | 对指定目录建索引，支持 `force` 全量重建 |
| `POST` | `/chat` | Agent 对话主入口，返回答案 + 引用 + 思考过程 |
| `GET` | `/search` | 纯检索，不经过 LLM，便于调试 |
| `GET` | `/tools` | 列出当前可用的工具 |
| `GET` | `/stats` | 索引与运行状态 |
| `GET` | `/health` | 健康检查（Ollama + 向量库） |

**`POST /chat` 示例**

```json
// 请求
{ "question": "订单创建后是怎么扣库存的？", "session_id": "demo-001", "top_k": 5 }

// 响应
{
  "answer": "订单创建的扣库存逻辑在 app/service/stock_service.py:12 的 deduct_stock()……",
  "references": [
    { "file_path": "app/service/stock_service.py", "start_line": 12, "end_line": 40, "symbol": "deduct_stock" }
  ],
  "steps": [
    { "thought": "需要先检索与订单创建、扣库存相关的代码", "action": "code_search", "action_input": "订单创建 扣减库存" }
  ],
  "elapsed_ms": 8421
}
```

`references` 单独返回，前端可点击跳转到对应代码位置；`steps` 暴露推理轨迹，方便调试与演示。

### 4.4 目录结构

```
CodeInsightAgent/
├── app/
│   ├── cli.py               # 命令行入口：index / search / ask / stats / serve
│   ├── main.py              # FastAPI 应用
│   ├── config.py            # 配置加载（.env + 环境变量）
│   ├── runtime.py           # 运行时装配：把 RAG、工具、Agent 串起来
│   ├── api/routes.py        # HTTP 路由层
│   ├── rag/                 # loader / splitter / embedder / vectorstore / retriever
│   ├── agent/               # react.py（推理循环）、prompt.py、tools/
│   └── llm/ollama_client.py # Ollama 调用封装
├── tests/                   # 单元测试（pytest）
├── data/chroma/             # 向量库，首次运行自动生成（已 gitignore）
├── requirements.txt / requirements-dev.txt
├── .env.example
└── LICENSE (MIT)
```

## 5. 可优化方向

**检索质量**

- 混合检索：向量 + BM25 并行召回后融合，解决「函数名精确匹配」场景下纯向量反而找不到的问题
- Rerank 重排：召回 Top-20 后用交叉编码器精排，提升 Top-5 命中率
- 查询改写：口语化提问先由 LLM 改写成更贴近代码用语的查询再检索

**索引**

- 跨文件上下文：基于调用图把「调用方 + 被调方」拼在一起，让 Agent 一次拿到完整链路
- 索引一致性：删除或重命名文件时同步清理向量库中的旧分块
- 语言扩展：目前覆盖 Python / Java / Markdown，可扩展 Go / TypeScript / C++

**Agent**

- 结构化输出：改用原生 function calling 替代文本解析，减少格式解析失败
- 反思机制：对检索结果做相关性自评，不满意则自动改写查询重试
- 并行工具调用：多路检索互相独立时可并发执行，降低响应延迟

**工程与体验**

- 流式输出：答案与思考过程用 SSE 流式返回，减少长时间等待的焦虑感
- 前端界面：用 Streamlit 或 Vue 展示答案 + 可点击的代码引用
- 效果评测：人工标注 50-100 个问答对（问题 + 正确文件/行号 + 参考答案），量化 Recall@k 与引用准确率，用数据说明设计决策的价值