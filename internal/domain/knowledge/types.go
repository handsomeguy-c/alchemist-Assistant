// Package knowledge 实现基于 LightRAG 范式（arXiv:2410.05779）的知识图谱功能：
//   - 双层次索引：实体（精确名称 key）+ 关系（LLM 生成的多个主题 keyword keys）
//   - 双层次检索：查询 → HL/LL 关键词提取 → 向量匹配 → 一跳子图扩展
//   - Gleaning 迭代抽取：首轮抽取 + 遗漏补全 + 循环检查
//   - 增量更新：新 chunk → 局部子图 → 集合并入现有图
//
// 所有操作在 Neo4j / Milvus 不可用时均优雅降级（返回空结果，不阻塞主流程）。
package knowledge

// EntityType 实体类型枚举
type EntityType string

const (
	EntityPerson   EntityType = "Person"
	EntityOrg      EntityType = "Organization"
	EntityLocation EntityType = "Location"
	EntityConcept  EntityType = "Concept"
	EntityEvent    EntityType = "Event"
	EntityProduct  EntityType = "Product"
	EntityUnknown  EntityType = "Unknown"
)

// Entity 是知识图谱中的一个节点
type Entity struct {
	Name    string     `json:"name"`
	Type    EntityType `json:"type"`
	DocHash string     `json:"doc_hash,omitempty"`
	ChunkID int        `json:"chunk_id,omitempty"` // 文档内的 chunk idx（0-based）
	PGID    int64      `json:"pg_id,omitempty"`    // PG 自增 ID（用于 RAG 检索 join 回真实 chunk）
}

// Relation 是两个实体之间的有向边
type Relation struct {
	FromName string  `json:"from"`
	ToName   string  `json:"to"`
	RelType  string  `json:"rel_type"` // RELATES_TO / PART_OF / CAUSES / DESCRIBES / MENTIONS
	Weight   float64 `json:"weight,omitempty"`
	DocHash  string  `json:"doc_hash,omitempty"`
	ChunkID  int     `json:"chunk_id,omitempty"`
	PGID     int64   `json:"pg_id,omitempty"`
}

// GraphSearchResult 是一次图检索的单条结果
type GraphSearchResult struct {
	ChunkID  int      `json:"chunk_id"` // 文档内 idx（兼容字段）
	PGID     int64    `json:"pg_id"`    // PG 自增 ID，用于 RAG RRF 融合
	Score    float64  `json:"score"`    // 基于向量相似度 + 图距离的综合分
	Entities []string `json:"entities"` // 命中的实体名称
	HopPath  []string `json:"hop_path"` // 遍历路径（可解释性）
}

// ExtractResult 是 Extractor.Extract 的输出（向后兼容）
type ExtractResult struct {
	Entities  []Entity   `json:"entities"`
	Relations []Relation `json:"relations"`
}

// ─────────────────────── LightRAG 扩展类型 ─────────────────────────────────

// EntityKV 是实体的键值对画像（LightRAG P(·) 函数输出）。
// Key 为实体名称（精确匹配），Value 为描述 + 关联文本片段。
type EntityKV struct {
	Key        string    `json:"key"`         // entity name (exact match key)
	Value      string    `json:"value"`       // description + associated snippet
	EntityType string    `json:"entity_type"` // Person / Org / ...
	DocHash    string    `json:"doc_hash,omitempty"`
	ChunkID    int       `json:"chunk_id,omitempty"`
	PGID       int64     `json:"pg_id,omitempty"`
	Embedding  []float64 `json:"embedding,omitempty"` // computed at index time
}

// RelationKV 是关系的键值对画像。
// Keys 为 LLM 生成的多个关键词（捕获全局主题），Value 为关系描述 + 关联文本。
// 这是 LightRAG 的核心创新：关系拥有 MULTIPLE keys（而非单一类型字符串），
// 使得高层抽象查询可以通过向量匹配找到相关关系。
type RelationKV struct {
	Keys     []string `json:"keys"`     // LLM-generated keyword keys (multiple!)
	Value    string   `json:"value"`    // description + snippets
	SrcID    string   `json:"src_id"`   // source entity name
	TgtID    string   `json:"tgt_id"`   // target entity name
	Keywords string   `json:"keywords"` // original keyword string (for display)
	Weight   float64  `json:"weight"`
	DocHash  string   `json:"doc_hash,omitempty"`
	ChunkID  int      `json:"chunk_id,omitempty"`
	PGID     int64    `json:"pg_id,omitempty"`
}

// QueryKeywords 是查询时的双层次关键词提取结果（LightRAG 检索 Step 1）。
// HighLevel → 匹配关系 keys（抽象主题）；LowLevel → 匹配实体 keys（具体实体）。
type QueryKeywords struct {
	HighLevel []string `json:"high_level"` // abstract topic keywords → relation vector DB
	LowLevel  []string `json:"low_level"`  // specific entity keywords → entity vector DB
}

// ExtractionRecord 是单个 chunk 的结构化抽取结果（LightRAG R(·) 函数输出）。
// 包含实体、关系、以及 chunk 级别的高层主题关键词。
type ExtractionRecord struct {
	Entities        []Entity   `json:"entities"`
	Relations       []Relation `json:"relations"`
	ContentKeywords []string   `json:"content_keywords"` // high-level themes of this chunk
}

// ─────────────────────── 向量存储接口 ───────────────────────────────────────

// EntityVectorHit 实体向量检索命中
type EntityVectorHit struct {
	EntityName string  `json:"entity_name"`
	PGID       int64   `json:"pg_id"`
	Score      float64 `json:"score"`
}

// RelationVectorHit 关系向量检索命中
type RelationVectorHit struct {
	SrcID   string  `json:"src_id"`
	TgtID   string  `json:"tgt_id"`
	PGID    int64   `json:"pg_id"`
	Score   float64 `json:"score"`
	Keyword string  `json:"keyword"` // which keyword matched
}

// KGVectorStore 是 KG 专用向量存储接口。
// 生产实现用 Milvus，降级时用 nil（跳过向量路径，回退到精确名称匹配）。
type KGVectorStore interface {
	// IndexEntity 将实体 key（名称）索引到向量库
	IndexEntity(ctx interface{}, name, desc string, pgID int64) error

	// IndexRelation 将关系的每个 keyword key 索引到向量库
	IndexRelation(ctx interface{}, srcID, tgtID string, keywords []string, desc string, pgID int64) error

	// SearchEntities 用查询向量搜索实体
	SearchEntities(ctx interface{}, queryVec []float32, topK int) ([]EntityVectorHit, error)

	// SearchRelations 用查询向量搜索关系
	SearchRelations(ctx interface{}, queryVec []float32, topK int) ([]RelationVectorHit, error)

	// DeleteByDocHash 删除某个文档的全部实体/关系向量
	DeleteByDocHash(ctx interface{}, docHash string) error

	// Available 报告向量存储是否可用
	Available() bool
}
