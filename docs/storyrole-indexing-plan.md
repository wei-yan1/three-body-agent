# StoryRole 导入索引闭环

当前导入任务不做 Chunk 数据库生命周期、版本回滚、软删除或 Chunk CRUD；只解决在线导入后的可检索闭环。

```text
上传
  ↓
解析 / 结构化分块
  ↓
chunks.jsonl
  ↓
Embedding（batch=40，有界并发=5，最多重试=5，指数退避）
  ↓
Chroma 增量 upsert
  +
持久化 BM25 增量 upsert
  ↓
ready
```

## 运行时配置

```env
STORYROLE_AUTO_INDEX=1
STORYROLE_CHROMA_DIR=data/indexes/chroma
STORYROLE_INDEX_COLLECTION_PREFIX=storyrole_novel_
STORYROLE_EMBED_BATCH_SIZE=40
STORYROLE_EMBED_WORKERS=5
STORYROLE_EMBED_MAX_RETRIES=5
STORYROLE_EMBED_RATE_PER_SECOND=0
```

## 增量规则

- 每个导入任务使用独立 collection：`storyrole_novel_<import_id>`，避免不同小说互相污染。
- chunk_id 已包含内容 hash；相同 chunk 不重复 embedding。
- 同一 import/source 的旧 chunk ID 会从 Chroma 和 BM25 清理。
- 新 chunk 才进入 embedding 队列。
- BM25 存在 `data/indexes/bm25/<collection>.json`，不是查询时临时在 Dense 候选上重建。
- 检索时 Dense 与持久化 BM25 独立召回，再融合和 rerank。

如果没有配置 embedding API，任务会在 embedding/vector_indexing 阶段失败并保留 chunks.jsonl；配置密钥后可以重试任务。


