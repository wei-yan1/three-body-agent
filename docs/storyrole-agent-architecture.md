# StoryRole 最终 Agent 架构

## 对话主链路

```text
用户消息
  ↓
Query Planner Agent
  ├─ 判断用户意图
  ├─ 判断是否需要小说检索、记忆、关系和时间线检查
  └─ 判断是否触发时间线检查
  ↓
Context Curator Agent
  ├─ 角色画像
  ├─ 小说证据
  ├─ 最近对话
  ├─ 用户记忆
  ├─ 关系状态
  └─ 时间线边界
  ↓
Character Reasoning Agent（ReAct）
  ├─ read_character_profile
  ├─ read_novel_evidence
  ├─ read_timeline_boundary
  └─ 输出有限的回应决策摘要，不输出详细思维链
  ↓
Character Conversation Agent
  └─ 以角色身份生成草稿
  ↓
Consistency Guard Agent
  ├─ 检查角色一致性
  ├─ 检查证据边界
  ├─ 检查未来剧情泄露
  └─ 必要时最小幅度重写
  ↓
Memory Decision Agent
  └─ 判断本轮是否包含值得长期保存的用户信息
  ↓
最终回复
```

## 当前已注册 Agent

- `character-resolver`：判断用户输入角色是否存在
- `nuwa-profiler`：生成并持久化角色画像
- `query-planner`：规划本轮请求
- `context-curator`：组装最小相关上下文
- `character-reasoning`：ReAct 风格内部决策
- `relationship-state`：提供人物关系上下文
- `timeline-guard`：提供知识边界和防剧透判断
- `character-conversation`：角色表达
- `consistency-guard`：回答质量审查和最小重写
- `memory-decision`：判断长期记忆候选

Agent 之间统一通过 A2A task/message/artifact 结构通信。默认本地进程调用，配置 `STORYROLE_A2A_TRANSPORT=http` 后可改为 HTTP A2A；每个 Agent 都有 Agent Card。

## MCP 工具层

MCP 不负责 Agent 编排，而负责提供稳定的工具能力。当前服务器：

```text
app/mcp/server.py
```

工具：

- `get_character_profile`
- `get_relationship_state`
- `check_timeline_boundary`
- `search_novel`
- `get_import_status`

stdio 启动：

```powershell
uv run python -m app.mcp.server
```

MCP 与 A2A 的边界：

```text
Agent：决定做什么
MCP：提供可以做什么
A2A：Agent 之间如何传递任务
```

## 设计原则

1. 不把详细隐藏思维链写入数据库或返回前端，只保留结构化决策摘要。
2. 角色画像、时间线和小说证据优先于模型自由发挥。
3. Context Curator 控制上下文大小，避免每轮注入整本小说。
4. Consistency Guard 最多做有限次最小重写，避免无限 Agent 循环。
5. Memory Decision 只产生记忆候选，持久化仍通过既有 MemoryTool 完成。
6. MCP 先提供只读能力；写入记忆、关系状态等能力需要增加权限和审计后再开放。

## 已落地的下一阶段能力

- 关系状态已写入 PostgreSQL，并保存最近互动与证据；普通对话不改变关系，只有明显正负向表达才会小幅变化。
- 长期记忆先保存为 `pending` 候选，不再由对话 Agent 直接写入长期记忆。审核通过后才调用 `MemoryTool` 持久化。
- 记忆候选支持去重、审核、拒绝、持久化状态，并保留原始线程名称。
- Timeline Guard 继续保持轻量规则版。
- 远程 A2A 拆分暂不启用，等 Agent 行为稳定后优先拆分 Reasoning 和 Consistency Guard。


## 当前实现取舍

### Relationship State

动态关系状态已迁移到 PostgreSQL `storyrole_relationship_states`，当前保存角色与用户之间的：

- trust
- intimacy
- tension
- dependency
- state
- evidence

只有明显的正向或负向互动才会小幅更新状态，普通对话不改变关系，避免关系漂移。

### Memory Decision

长期记忆不再由 Agent 直接写入。候选先进入 `storyrole_memory_candidates`，状态为 `pending`：

```text
Memory Decision
  ↓
pending candidate
  ↓
人工审核或后续审核 Agent
  ↓
approved / rejected
  ↓
approved 才写入 MemoryTool
```

审核接口：

```text
GET  /api/v1/storyrole/memory-candidates
POST /api/v1/storyrole/memory-candidates/{candidate_id}/review?status=approved
```

### Timeline Guard

当前保留轻量版本：只基于角色画像中的 `knowledge_boundary` 和用户问题中的未来提示词进行提醒，不建立完整事件图。等小说章节事件抽取稳定后再增强。

### A2A 拆分

当前仍然默认进程内 A2A。由于所有 Agent 已有 Agent Card 和统一任务协议，后续可以优先拆分 `character-reasoning` 或 `consistency-guard`，不需要改业务接口。


## Laya 决策后端

动态角色的 `query-planner` 已支持三种模式：

```env
STORYROLE_DECISION_BACKEND=laya        # 默认，主动决策并在失败时回退规则
STORYROLE_DECISION_BACKEND=laya_shadow  # Laya 只参与对比，不改变实际结果
STORYROLE_DECISION_BACKEND=rule        # 强制使用规则
```

Laya 通过 `app/agents/decision/laya_backend.py` 懒加载，不需要 Ollama。默认启用 Laya；未安装或加载/预测失败时会自动退回规则后端。

```env
STORYROLE_LAYA_MODEL=convaiinnovations/laya-multilingual
STORYROLE_LAYA_DEVICE=auto
```

Laya 只负责意图、检索、记忆、关系和时间线等快速结构化决策；深度推理不由 Laya 决定，而由用户请求中的 `deep_reasoning` 开关决定。角色回答仍由 Character Conversation Agent 使用生成模型完成。


### 深度推理开关

动态角色聊天请求支持：

```json
{
  "message": "请认真分析这个选择的代价",
  "deep_reasoning": true
}
```

默认 `false`，普通对话不会额外调用 Reasoning Agent 的模型推理；用户明确开启后才执行深度 ReAct 决策。


## 快速回答与深度思考

对话支持 `quick`、`deep`、`auto` 三种模式。`auto` 只使用规则规划，不额外调用路由模型。

快速回答：

```text
规则规划 → 必要时少量混合检索 → 一次角色模型生成
```

深度思考：

```text
深度问题规划 → 多查询并行检索 → BM25+dense 融合 → Rerank
→ 证据分析 → 角色认知决策 → 角色模型生成 → 一致性复核
```

深度模式只返回最终角色回答，不返回隐藏思维链；任一深度步骤失败时仍使用已有上下文生成回答。

API 请求示例：

```json
{"message":"请分析他做出这个选择的动机和代价","mode":"deep","period_id":12}
```

旧客户端的 `deep_reasoning=true` 仍兼容。


### 本轮编排完善

- 深度模式最多使用 3 个子查询，先并行 Dense+BM25 召回，再合并去重，统一 Rerank 一次。
- 快速模式只在规划结果要求时读取关系或时间线，不再每轮固定调用。
- 深度阶段分别设置规划、检索、证据分析、角色认知、复核和最终回答预算；可选阶段失败时使用结构化降级结果。
- 记忆候选和关系更新改为回答返回后的异步任务，不阻塞用户看到回答。
- 回答结果增加 `mode`、`evidence_count`、`latency_ms`，用于前端提示和后续质量评测。
- 一致性复核输入现在包含时间线、证据分析和检索原文，不只检查静态角色画像。
