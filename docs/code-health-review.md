# StoryRole 代码健康整理记录

本轮针对现有工程做了稳定性、边界和可维护性整理。

## 已处理

- PostgreSQL Schema 初始化增加进程级幂等保护，避免每个请求重复建表和重复开启初始化连接。
- A2A 任务注册表增加有界 LRU 风格淘汰，避免长期运行进程内存无限增长。
- A2A HTTP 通信支持可选 `X-StoryRole-A2A-Key` 服务间密钥。
- 动态角色 API 不再把 Planner、Reasoning、Guard、Memory Decision 内部结果返回给前端。
- 动态角色检索记忆只在 Query Planner 判断需要时执行，减少无意义向量检索。
- 记忆写入移动到回答审查之后，保存最终回答而不是未经审查的草稿。
- A2A 同步 Agent Handler 放入线程执行，避免同步数据库/文件操作阻塞 FastAPI 事件循环。
- ReAct、画像分析、Consistency Guard 的模型调用使用异步接口。
- 清理多个未使用导入和废弃局部变量，确保应用代码无 Ruff F 级错误。
- 修复固定角色 API 的 Persona 路径解析，移除悬空的 `paths` 变量。
- 小说导入创建工作区时，如果 PostgreSQL 写入失败，会清理已创建文件并返回 503，避免出现“文件任务存在但数据库没有工作区”的半成功状态。

## 暂未删除的模块

部分旧模块仍是空壳，但没有被当前运行链路引用，例如旧的 archive/timeline/tool registry 和 legacy memory 文件。它们暂时保留，避免破坏已有外部脚本；后续可在一次独立清理提交中删除。

## 验证

```text
uv run python -m compileall -q app
uv run --with pytest --with pytest-asyncio pytest tests/unit -q
10 passed
uv run --with ruff ruff check app --select F
All checks passed
```
