package knowledge

import (
	"alchemist-Assistant/config"
	"alchemist-Assistant/internal/infrastructure/platform/neo4j"
	"context"
	"fmt"
	"log"
	"sort"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
)

// KGStore 在 platform/neo4j.Client 之上封装 RAG 专用的图操作。
//
// LightRAG 范式实现（arXiv:2410.05779）：
//   - IndexDocument：文档摄入时抽取实体/关系 + 键值对画像 + 向量索引（P(·) + D(·)）
//   - Search：双层次检索（HL/LL 关键词 → 向量匹配 → 一跳子图扩展）
//   - DeleteDocument：级联删除节点 + 向量
//
// 所有操作在 Neo4j / Milvus 不可用时均优雅降级。
type KGStore struct {
	neo4j        *neo4j.Client
	maxHops      int
	kgWeight     float64
	vectorWeight float64
	extractor    *Extractor
	vectorIdx    *milvusVectorStore
	embedFn      func(text string) ([]float64, error)
}

// KGStoreConfig is the constructor parameter bag for NewKGStore.
type KGStoreConfig struct {
	Config       *config.APIConfig
	LLMFn        func(systemPrompt, userMsg string) string
	MilvusClient milvusClient.Client
	EmbedFn      func(text string) ([]float64, error)
}

// NewKGStore 创建知识图谱存储
func NewKGStore(cfg *config.APIConfig, llmFn func(systemPrompt, userMsg string) string) *KGStore {
	return NewKGStoreWithDeps(KGStoreConfig{
		Config: cfg,
		LLMFn:  llmFn,
	})
}

// NewKGStoreWithDeps 创建带完整依赖的知识图谱存储（含向量索引）
func NewKGStoreWithDeps(cfg KGStoreConfig) *KGStore {
	// 构建 Extractor 配置
	extractorCfg := DefaultExtractorConfig()
	if len(cfg.Config.KGEntityTypes) > 0 {
		extractorCfg.EntityTypes = cfg.Config.KGEntityTypes
	}
	if len(cfg.Config.KGRelationTypes) > 0 {
		extractorCfg.RelationTypes = cfg.Config.KGRelationTypes
	}
	if cfg.Config.KGMaxGleaning >= 0 {
		extractorCfg.MaxGleaning = cfg.Config.KGMaxGleaning
	}
	if !cfg.Config.KGEnableGleaning {
		extractorCfg.MaxGleaning = 0
	}

	ks := &KGStore{
		neo4j:        neo4j.Connect(cfg.Config.Neo4jConfig),
		maxHops:      cfg.Config.KGMaxHops,
		kgWeight:     cfg.Config.KGWeight,
		vectorWeight: cfg.Config.KGVectorWeight,
		extractor:    NewExtractorWithConfig(cfg.LLMFn, extractorCfg),
		embedFn:      cfg.EmbedFn,
	}

	// 初始化向量索引
	if cfg.MilvusClient != nil && cfg.EmbedFn != nil {
		dim := cfg.Config.KGVectorDim
		if dim <= 0 {
			dim = cfg.Config.RAGMilvusDim
		}
		if dim <= 0 {
			dim = 1024
		}
		ks.vectorIdx = newMilvusVectorStore(cfg.MilvusClient, cfg.EmbedFn, dim)
		ks.vectorIdx.Init()
	}

	return ks
}

// Available 图存储是否可用
func (ks *KGStore) Available() bool { return ks.neo4j.Available() }

// Close 关闭底层连接
func (ks *KGStore) Close() { ks.neo4j.Close() }

// Client 暴露底层 Neo4j 客户端，供 memory 包共享同一连接驱动记忆图
func (ks *KGStore) Client() *neo4j.Client { return ks.neo4j }

// VectorAvailable 报告向量索引是否可用
func (ks *KGStore) VectorAvailable() bool {
	return ks.vectorIdx != nil && ks.vectorIdx.Available()
}

// ─────────────────────────────── 文档摄入 ────────────────────────────────────

// IndexDocument 为一批 chunks 抽取实体关系并写入图 + 向量索引。
// 以异步方式调用，不阻塞主 Ingest 流程。
//
// LightRAG 流程：
//  1. R(·): ExtractWithGleaning 抽取实体 + 关系 + content_keywords
//  2. P(·): 生成键值对 — 实体 key=名称, 关系 key=LLM 生成的关键词
//  3. D(·): 去重合并（MERGE 幂等）
//  4. 向量索引: 实体名称 + 关系关键词 → embedding → Milvus
func (ks *KGStore) IndexDocument(docHash string, chunks []ChunkRef) {
	if !ks.neo4j.Available() {
		return
	}

	for _, c := range chunks {
		record := ks.extractor.ExtractWithGleaning(c.Content)
		if len(record.Entities) == 0 && len(record.Relations) == 0 {
			continue
		}

		ctx := context.Background()

		// 写入实体节点
		for _, ent := range record.Entities {
			ent.DocHash = docHash
			ent.ChunkID = c.ID
			ent.PGID = c.PGID
			ks.upsertEntity(ctx, ent)

			// 向量索引：实体名称
			if ks.VectorAvailable() {
				ks.vectorIdx.IndexEntity(ctx, ent.Name, string(ent.Type), ent.PGID)
			}
		}

		// 写入关系边 + 关系关键词向量索引
		for _, rel := range record.Relations {
			rel.DocHash = docHash
			rel.ChunkID = c.ID
			rel.PGID = c.PGID
			if rel.Weight <= 0 {
				rel.Weight = 0.5
			}
			ks.upsertRelation(ctx, rel)

			// 向量索引：关系关键词（LightRAG 的核心创新：关系有多个 key）
			if ks.VectorAvailable() && len(record.ContentKeywords) > 0 {
				// 使用 content_keywords + 关系描述作为关系向量
				allKeywords := append([]string{}, record.ContentKeywords...)
				ks.vectorIdx.IndexRelation(ctx, rel.FromName, rel.ToName, allKeywords,
					fmt.Sprintf("%s - %s", rel.RelType, rel.FromName+"→"+rel.ToName), rel.PGID)
			}
		}
	}
	log.Printf("🕸️  知识图谱索引完成（LightRAG 范式）：docHash=%s，chunks=%d", docHash, len(chunks))
}

// ChunkRef 是 KGStore 摄入时需要的 chunk 信息（避免直接依赖 rag 包形成循环）
type ChunkRef struct {
	ID      int   // 文档内 chunk idx（0-based）
	PGID    int64 // PostgreSQL 自增 ID，KG 节点上同时持久化以支持 RAG RRF 融合
	Content string
}

// upsertEntity MERGE 实体节点（幂等）
func (ks *KGStore) upsertEntity(ctx context.Context, ent Entity) {
	sess := ks.neo4j.Session()
	defer sess.Close(ctx)
	query := `MERGE (e:Entity {name: $name})
	          SET e.type = $type, e.doc_hash = $doc_hash, e.chunk_id = $chunk_id, e.pg_id = $pg_id`
	_, err := sess.Run(ctx, query, map[string]any{
		"name":     ent.Name,
		"type":     string(ent.Type),
		"doc_hash": ent.DocHash,
		"chunk_id": ent.ChunkID,
		"pg_id":    ent.PGID,
	})
	if err != nil {
		log.Printf("⚠️  Neo4j upsertEntity 失败 (%s): %v", ent.Name, err)
	}
}

// upsertRelation MERGE 关系边（幂等）
func (ks *KGStore) upsertRelation(ctx context.Context, rel Relation) {
	sess := ks.neo4j.Session()
	defer sess.Close(ctx)
	// 动态关系类型无法用参数传递，必须拼入查询字符串
	// 安全性由 isValidRelType 保证（extractor 已过滤非法类型）
	query := `MERGE (a:Entity {name: $from})
	          MERGE (b:Entity {name: $to})
	          MERGE (a)-[r:` + rel.RelType + ` {doc_hash: $doc_hash}]->(b)
	          SET r.chunk_id = $chunk_id, r.pg_id = $pg_id, r.weight = $weight`
	_, err := sess.Run(ctx, query, map[string]any{
		"from":     rel.FromName,
		"to":       rel.ToName,
		"doc_hash": rel.DocHash,
		"chunk_id": rel.ChunkID,
		"pg_id":    rel.PGID,
		"weight":   rel.Weight,
	})
	if err != nil {
		log.Printf("⚠️  Neo4j upsertRelation 失败 (%s→%s): %v", rel.FromName, rel.ToName, err)
	}
}

// ─────────────────────────────── 文档删除 ────────────────────────────────────

// DeleteDocument 删除与 docHash 关联的所有关系，并清理孤立实体节点
func (ks *KGStore) DeleteDocument(docHash string) {
	if !ks.neo4j.Available() {
		return
	}
	ctx := context.Background()
	sess := ks.neo4j.Session()
	defer sess.Close(ctx)

	// 删除所有归属此文档的关系
	_, err := sess.Run(ctx,
		`MATCH ()-[r {doc_hash: $doc_hash}]-() DELETE r`,
		map[string]any{"doc_hash": docHash})
	if err != nil {
		log.Printf("⚠️  Neo4j 删除文档关系失败: %v", err)
	}
	// 清理孤立实体节点
	_, err = sess.Run(ctx,
		`MATCH (e:Entity) WHERE NOT (e)--() AND e.doc_hash = $doc_hash DELETE e`,
		map[string]any{"doc_hash": docHash})
	if err != nil {
		log.Printf("⚠️  Neo4j 清理孤立节点失败: %v", err)
	}

	// 向量索引删除（Milvus 中按 doc_hash 删除依赖 pg_id，当前由 PG/Neo4j 主导）
	if ks.VectorAvailable() {
		ks.vectorIdx.DeleteByDocHash(ctx, docHash)
	}
}

// ─────────────────────────────── 图检索（双层次）──────────────────────────────

// Search 执行 LightRAG 双层次检索：
//
//  1. 查询关键词提取：LLM → HL（高层主题）+ LL（低层实体）关键词
//  2. 向量匹配：LL 关键词 → entity vector DB；HL 关键词 → relation vector DB
//  3. 子图扩展：从匹配到的候选实体做 1-hop 邻居遍历
//  4. 综合评分：向量相似度 × vectorWeight + 图结构分 × (1-vectorWeight)
//
// 降级路径：
//   - 向量索引不可用 → 回退到精确名称匹配（旧 Search 行为）
//   - Neo4j 不可用 → 返回 nil
func (ks *KGStore) Search(queryText string, topK int) []GraphSearchResult {
	if !ks.neo4j.Available() {
		return nil
	}

	// ── 路径 A：双层次检索（向量可用时）─────────────────────────────────
	if ks.VectorAvailable() {
		return ks.searchDualLevel(queryText, topK)
	}

	// ── 路径 B：精确名称匹配降级（向量不可用时）─────────────────────────
	return ks.searchExact(queryText, topK)
}

// searchDualLevel LightRAG 双层次检索
func (ks *KGStore) searchDualLevel(queryText string, topK int) []GraphSearchResult {
	// Step 1: 提取双层次关键词
	keywords := ks.extractor.ExtractQueryKeywords(queryText)

	// Step 2: 对 HL/LL 关键词做向量搜索
	candidateNames := make(map[string]float64) // entity name → vector score
	candidatePGIDs := make(map[int64]float64)  // pg_id → max vector score

	// 嵌入 LL 关键词 → 搜索实体向量库
	if len(keywords.LowLevel) > 0 {
		llText := joinKeywords(keywords.LowLevel)
		if llEmb, err := ks.embedFn(llText); err == nil {
			entityHits, _ := ks.vectorIdx.SearchEntities(nil, float64To32(llEmb), topK*2)
			for _, hit := range entityHits {
				if hit.EntityName != "" {
					if existing, ok := candidateNames[hit.EntityName]; !ok || hit.Score > existing {
						candidateNames[hit.EntityName] = hit.Score
					}
				}
				if hit.PGID > 0 {
					if existing, ok := candidatePGIDs[hit.PGID]; !ok || hit.Score > existing {
						candidatePGIDs[hit.PGID] = hit.Score
					}
				}
			}
		}
	}

	// 嵌入 HL 关键词 → 搜索关系向量库
	if len(keywords.HighLevel) > 0 {
		hlText := joinKeywords(keywords.HighLevel)
		if hlEmb, err := ks.embedFn(hlText); err == nil {
			relHits, _ := ks.vectorIdx.SearchRelations(nil, float64To32(hlEmb), topK*2)
			for _, hit := range relHits {
				if hit.SrcID != "" {
					candidateNames[hit.SrcID] = 0.5 // 关系端点得分略低
				}
				if hit.TgtID != "" {
					candidateNames[hit.TgtID] = 0.5
				}
				if hit.PGID > 0 {
					if _, ok := candidatePGIDs[hit.PGID]; !ok {
						candidatePGIDs[hit.PGID] = hit.Score * 0.7
					}
				}
			}
		}
	}

	// 如果没有向量命中，回退到精确匹配
	if len(candidateNames) == 0 && len(candidatePGIDs) == 0 {
		return ks.searchExact(queryText, topK)
	}

	// Step 3: 子图扩展 — 从候选实体出发做 1-hop 遍历
	graphResults := ks.expandFromCandidates(candidateNames, topK*2)

	// Step 4: 综合评分（向量 + 图结构）
	results := make([]GraphSearchResult, 0, len(graphResults))
	for _, gr := range graphResults {
		// 图结构分（归一化）
		graphScore := float64(len(gr.Entities))*0.3 + 0.3 // 命中实体数 + 基础分

		// 向量分（如果有）
		vecScore := 0.0
		for _, ent := range gr.Entities {
			if s, ok := candidateNames[ent]; ok && s > vecScore {
				vecScore = s
			}
		}
		if s, ok := candidatePGIDs[gr.PGID]; ok && s > vecScore {
			vecScore = s
		}

		// 融合：向量权重 + 图权重
		vw := ks.vectorWeight
		if vw <= 0 {
			vw = 0.5
		}
		gr.Score = (vw*vecScore + (1-vw)*graphScore) * ks.kgWeight

		results = append(results, gr)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

// searchExact 精确名称匹配降级（向量索引不可用时的回退路径）。
// 与旧 Search 行为一致：LLM 抽取查询实体 → Neo4j 精确名称匹配 → APOC 路径扩展。
func (ks *KGStore) searchExact(queryText string, topK int) []GraphSearchResult {
	// 抽取查询中的实体
	extracted := ks.extractor.Extract(queryText)
	if len(extracted.Entities) == 0 {
		return nil
	}

	// 构建实体名列表
	names := make([]string, 0, len(extracted.Entities))
	for _, e := range extracted.Entities {
		names = append(names, e.Name)
	}

	return ks.expandFromCandidatesList(names, topK)
}

// expandFromCandidates 从候选实体名称 + 分数出发做子图扩展
func (ks *KGStore) expandFromCandidates(candidates map[string]float64, topK int) []GraphSearchResult {
	names := make([]string, 0, len(candidates))
	for name := range candidates {
		names = append(names, name)
	}
	return ks.expandFromCandidatesList(names, topK)
}

// expandFromCandidatesList 从实体名称列表出发做 Neo4j 子图扩展（1~maxHops 跳）
func (ks *KGStore) expandFromCandidatesList(names []string, topK int) []GraphSearchResult {
	if len(names) == 0 {
		return nil
	}

	ctx := context.Background()

	hops := ks.maxHops
	if hops <= 0 {
		hops = 2
	}
	if hops > 3 {
		hops = 3
	}

	// 优先使用 APOC 路径扩展
	sess := ks.neo4j.Session()
	defer sess.Close(ctx)

	query := `
		MATCH (e:Entity) WHERE e.name IN $names
		CALL apoc.path.subgraphNodes(e, {
		  maxLevel: $hops,
		  relationshipFilter: "RELATES_TO|PART_OF|CAUSES|DESCRIBES|MENTIONS|WORKS_FOR|LOCATED_IN"
		})
		YIELD node AS neighbor
		WHERE neighbor:Entity AND neighbor.chunk_id IS NOT NULL
		WITH e.name AS seed, neighbor.name AS nb, neighbor.chunk_id AS cid,
		     COALESCE(neighbor.pg_id, 0) AS pgid,
		     toInteger(apoc.node.degree(neighbor)) AS degree
		RETURN cid, pgid, collect(DISTINCT seed) AS seeds, collect(DISTINCT nb) AS neighbors, max(degree) AS deg
		ORDER BY size(seeds) DESC, deg DESC
		LIMIT $limit`

	records, err := sess.Run(ctx, query, map[string]any{
		"names": names,
		"hops":  int64(hops),
		"limit": int64(topK * 3),
	})
	if err != nil {
		// APOC 不可用时降级为直接节点匹配
		return ks.searchDirect(ctx, names, topK)
	}

	// 收集结果
	type rawResult struct {
		chunkID   int
		pgID      int64
		seeds     []string
		neighbors []string
		degree    int64
	}
	var raw []rawResult
	for records.Next(ctx) {
		rec := records.Record()
		cid, _ := rec.Get("cid")
		pgid, _ := rec.Get("pgid")
		seeds, _ := rec.Get("seeds")
		nbs, _ := rec.Get("neighbors")
		deg, _ := rec.Get("deg")

		chunkID := toInt(cid)
		if chunkID < 0 {
			continue
		}
		r := rawResult{
			chunkID:   chunkID,
			pgID:      toInt64(pgid),
			seeds:     toStringSlice(seeds),
			neighbors: toStringSlice(nbs),
			degree:    toInt64(deg),
		}
		raw = append(raw, r)
	}
	if err := records.Err(); err != nil {
		log.Printf("⚠️  Neo4j 图检索 records 错误: %v", err)
	}

	// 计算分数：命中种子越多 + 图中心度越高 → 分越高
	seen := make(map[int64]bool)
	var results []GraphSearchResult
	for _, r := range raw {
		if r.pgID == 0 || seen[r.pgID] {
			continue
		}
		seen[r.pgID] = true
		score := float64(len(r.seeds))*0.6 + float64(r.degree)*0.01
		results = append(results, GraphSearchResult{
			ChunkID:  r.chunkID,
			PGID:     r.pgID,
			Score:    score,
			Entities: r.seeds,
			HopPath:  r.neighbors,
		})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

// searchDirect APOC 不可用时的降级版本：直接匹配实体所在 chunk
func (ks *KGStore) searchDirect(ctx context.Context, names []string, topK int) []GraphSearchResult {
	sess := ks.neo4j.Session()
	defer sess.Close(ctx)

	records, err := sess.Run(ctx,
		`MATCH (e:Entity) WHERE e.name IN $names AND e.chunk_id IS NOT NULL
		 RETURN e.chunk_id AS cid, COALESCE(e.pg_id, 0) AS pgid, e.name AS name ORDER BY cid LIMIT $limit`,
		map[string]any{"names": names, "limit": int64(topK)})
	if err != nil {
		return nil
	}

	seen := make(map[int64]bool)
	var results []GraphSearchResult
	for records.Next(ctx) {
		rec := records.Record()
		cid := toInt(rec.Values[0])
		pgid := toInt64(rec.Values[1])
		name := toString(rec.Values[2])
		if pgid == 0 || seen[pgid] {
			continue
		}
		seen[pgid] = true
		results = append(results, GraphSearchResult{
			ChunkID:  cid,
			PGID:     pgid,
			Score:    ks.kgWeight,
			Entities: []string{name},
		})
	}
	return results
}

// ─────────────────────────────── 内部工具 ────────────────────────────────────

func toInt(v any) int {
	switch x := v.(type) {
	case int64:
		return int(x)
	case int:
		return x
	case float64:
		return int(x)
	}
	return -1
}

func toInt64(v any) int64 {
	if x, ok := v.(int64); ok {
		return x
	}
	return 0
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func toStringSlice(v any) []string {
	if arr, ok := v.([]any); ok {
		s := make([]string, 0, len(arr))
		for _, a := range arr {
			if str, ok := a.(string); ok {
				s = append(s, str)
			}
		}
		return s
	}
	return nil
}

func intStr(n int) string {
	return fmt.Sprintf("%d", n)
}

func joinKeywords(kws []string) string {
	result := ""
	for i, kw := range kws {
		if i > 0 {
			result += ", "
		}
		result += kw
	}
	return result
}
