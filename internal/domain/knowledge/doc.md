Package knowledge 知识图谱（RAG 文档实体关系）。完整 KG V2 说明见 `docs/kg-v2-mvp.md`。

由 LLM 从文档抽取实体和关系，按 `KGChunk → KGEntityMention → KGEntity` 保存可回溯证据。
检索时组合名称、别名、Entity Embedding 种子，只做受控 1~2 跳扩散，并返回真实证据 `pg_id`。
文件分组：

types.go       KG V2 领域类型与向量接口
normalizer.go  NFKC 实体归一化
ids.go         Entity / Mention / Relation 稳定 ID
extractor.go   严格 JSON 抽取与类型白名单
model.go       Chunk 图构建、种子与证据评分
kgstore.go     Neo4j 摄入、删除、受控扩散和 Evidence 回溯
vector.go      Milvus kg_entities 索引

与 memory/graph 的区别：
- knowledge 处理文档 KG V2（KG 前缀 Label）和证据回溯
- memory/graph 处理"记忆条目"（Memory 节点）—— 时序 + 相似度连接
- 二者共享同一 Neo4j 实例，但 schema 不同（节点 label 不同）
