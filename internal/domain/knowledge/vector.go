// Package knowledge — KG 专用向量存储（LightRAG 向量索引层）。
//
// 为实体名称和关系关键词建立独立的向量索引，支撑双层次检索：
//   - 实体向量索引：entity name → embedding → Milvus "kg_entities" collection
//   - 关系向量索引：relation keyword → embedding → Milvus "kg_relations" collection
//
// 所有操作在 Milvus 不可用时优雅降级（返回空结果/跳过索引）。
package knowledge

import (
	"context"
	"fmt"
	"log"
	"strings"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ─────────────────────────────── 常量 ──────────────────────────────────────

const (
	kgEntityColName   = "kg_entities"
	kgRelationColName = "kg_relations"
	kgVecDim          = 1024 // 默认向量维度，与 RAG chunk 维度对齐
)

// ─────────────────────────────── milvusVectorStore ─────────────────────────

// milvusVectorStore 是 KGVectorStore 的 Milvus 实现。
// nil milvus client 时所有方法变成空操作（降级）。
type milvusVectorStore struct {
	milvus  milvusClient.Client
	embedFn func(text string) ([]float64, error)
	dim     int
}

// newMilvusVectorStore 创建 Milvus 向量存储；milvus 为 nil 时降级为空操作。
// embedFn 用于将实体名称/关键词文本转为向量。
func newMilvusVectorStore(milvus milvusClient.Client, embedFn func(text string) ([]float64, error), dim int) *milvusVectorStore {
	if dim <= 0 {
		dim = kgVecDim
	}
	return &milvusVectorStore{milvus: milvus, embedFn: embedFn, dim: dim}
}

func (vs *milvusVectorStore) Available() bool {
	return vs.milvus != nil && vs.embedFn != nil
}

// Init 创建 collection 和索引（幂等）
func (vs *milvusVectorStore) Init() {
	if !vs.Available() {
		return
	}
	vs.ensureCollection(kgEntityColName, "entity_name")
	vs.ensureCollection(kgRelationColName, "keyword")
}

func (vs *milvusVectorStore) ensureCollection(colName, keyFieldName string) {
	ctx := context.Background()
	has, err := vs.milvus.HasCollection(ctx, colName)
	if err != nil {
		log.Printf("⚠️  KG vector: 检查 %s collection 失败: %v", colName, err)
		return
	}
	if has {
		// 检查维度是否匹配
		coll, err := vs.milvus.DescribeCollection(ctx, colName)
		if err == nil {
			for _, f := range coll.Schema.Fields {
				if f.Name == "embedding" && f.DataType == entity.FieldTypeFloatVector {
					if f.TypeParams["dim"] != fmt.Sprintf("%d", vs.dim) {
						log.Printf("⚠️  KG vector: %s 维度不匹配，重建 collection", colName)
						vs.milvus.DropCollection(ctx, colName)
						has = false
					}
				}
			}
		}
		if has {
			_ = vs.milvus.LoadCollection(ctx, colName, false)
			return
		}
	}

	schema := &entity.Schema{
		CollectionName: colName,
		Fields: []*entity.Field{
			{Name: "pg_id", DataType: entity.FieldTypeInt64, PrimaryKey: true, AutoID: false},
			{Name: keyFieldName, DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "512"}},
			{Name: "content", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "4096"}},
			{Name: "embedding", DataType: entity.FieldTypeFloatVector, TypeParams: map[string]string{"dim": fmt.Sprintf("%d", vs.dim)}},
		},
	}
	if err := vs.milvus.CreateCollection(ctx, schema, 1); err != nil {
		log.Printf("⚠️  KG vector: 创建 %s collection 失败: %v", colName, err)
		return
	}
	idx, _ := entity.NewIndexIvfFlat(entity.IP, 128)
	if err := vs.milvus.CreateIndex(ctx, colName, "embedding", idx, false); err != nil {
		log.Printf("⚠️  KG vector: %s 索引创建失败: %v", colName, err)
	}
	if err := vs.milvus.LoadCollection(ctx, colName, false); err != nil {
		log.Printf("⚠️  KG vector: %s 加载失败: %v", colName, err)
	}
	log.Printf("✅ KG vector: %s collection 已就绪 (dim=%d)", colName, vs.dim)
}

// ─────────────────────────────── 索引操作 ──────────────────────────────────

func (vs *milvusVectorStore) IndexEntity(ctx interface{}, name, desc string, pgID int64) error {
	if !vs.Available() {
		return nil
	}
	// 向量化：实体名称 + 描述
	vecText := name
	if desc != "" {
		vecText = name + ": " + desc
	}
	emb, err := vs.embedFn(vecText)
	if err != nil {
		return fmt.Errorf("embed entity %s: %w", name, err)
	}
	emb32 := float64To32(emb)

	_, err = vs.milvus.Insert(
		context.Background(), kgEntityColName, "",
		entity.NewColumnInt64("pg_id", []int64{pgID}),
		entity.NewColumnVarChar("entity_name", []string{name}),
		entity.NewColumnVarChar("content", []string{vecText}),
		entity.NewColumnFloatVector("embedding", vs.dim, [][]float32{emb32}),
	)
	if err != nil {
		return fmt.Errorf("insert entity vector for %s: %w", name, err)
	}
	return nil
}

func (vs *milvusVectorStore) IndexRelation(ctx interface{}, srcID, tgtID string, keywords []string, desc string, pgID int64) error {
	if !vs.Available() {
		return nil
	}
	// 为每个 keyword 插入一条向量记录
	for _, kw := range keywords {
		kw = strings.TrimSpace(kw)
		if kw == "" {
			continue
		}
		vecText := kw
		if desc != "" {
			vecText = kw + ": " + desc
		}
		emb, err := vs.embedFn(vecText)
		if err != nil {
			log.Printf("⚠️  KG vector: embed relation keyword %q: %v", kw, err)
			continue
		}
		emb32 := float64To32(emb)

		// 用 pg_id + keyword hash 做复合主键（Milvus 不支持字符串主键）
		// 实际用 pg_id 组合：pgID*10000 + hash(kw) % 10000
		compositeID := pgID*10000 + int64(hashKeyword(kw)%10000)

		_, err = vs.milvus.Insert(
			context.Background(), kgRelationColName, "",
			entity.NewColumnInt64("pg_id", []int64{compositeID}),
			entity.NewColumnVarChar("keyword", []string{kw}),
			entity.NewColumnVarChar("content", []string{vecText}),
			entity.NewColumnFloatVector("embedding", vs.dim, [][]float32{emb32}),
		)
		if err != nil {
			log.Printf("⚠️  KG vector: insert relation vector for %q: %v", kw, err)
		}
	}
	return nil
}

// ─────────────────────────────── 检索操作 ──────────────────────────────────

func (vs *milvusVectorStore) SearchEntities(ctx interface{}, queryVec []float32, topK int) ([]EntityVectorHit, error) {
	if !vs.Available() || len(queryVec) == 0 {
		return nil, nil
	}
	sp, _ := entity.NewIndexFlatSearchParam()
	results, err := vs.milvus.Search(
		context.Background(), kgEntityColName, []string{},
		"", []string{"entity_name", "pg_id"},
		[]entity.Vector{entity.FloatVector(queryVec)},
		"embedding", entity.IP,
		topK, sp,
	)
	if err != nil {
		return nil, fmt.Errorf("search entities: %w", err)
	}

	var hits []EntityVectorHit
	for _, r := range results {
		// r.IDs 包含主键（pg_id）
		var ids []int64
		if r.IDs != nil {
			ids = r.IDs.FieldData().GetScalars().GetLongData().Data
		}
		// 从 output fields 提取 entity_name
		entityNameCol := r.Fields.GetColumn("entity_name")
		for i := range r.Scores {
			entityName := ""
			if entityNameCol != nil && i < entityNameCol.Len() {
				entityName, _ = entityNameCol.GetAsString(i)
			}
			pgID := int64(0)
			if i < len(ids) {
				pgID = ids[i]
			}
			hits = append(hits, EntityVectorHit{
				EntityName: entityName,
				PGID:       pgID,
				Score:      float64(r.Scores[i]),
			})
		}
	}
	return hits, nil
}

func (vs *milvusVectorStore) SearchRelations(ctx interface{}, queryVec []float32, topK int) ([]RelationVectorHit, error) {
	if !vs.Available() || len(queryVec) == 0 {
		return nil, nil
	}
	sp, _ := entity.NewIndexFlatSearchParam()
	results, err := vs.milvus.Search(
		context.Background(), kgRelationColName, []string{},
		"", []string{"keyword", "pg_id"},
		[]entity.Vector{entity.FloatVector(queryVec)},
		"embedding", entity.IP,
		topK, sp,
	)
	if err != nil {
		return nil, fmt.Errorf("search relations: %w", err)
	}

	var hits []RelationVectorHit
	for _, r := range results {
		keywordCol := r.Fields.GetColumn("keyword")
		pgIDCol := r.Fields.GetColumn("pg_id")
		for i := range r.Scores {
			keyword := ""
			if keywordCol != nil && i < keywordCol.Len() {
				keyword, _ = keywordCol.GetAsString(i)
			}
			pgID := int64(0)
			if pgIDCol != nil && i < pgIDCol.Len() {
				pgID, _ = pgIDCol.GetAsInt64(i)
			}
			// 反解 composite ID 得到真实 pg_id
			realPGID := pgID / 10000
			if realPGID <= 0 {
				// 如果无法反解，使用 ID 列
				if r.IDs != nil {
					ids := r.IDs.FieldData().GetScalars().GetLongData().Data
					if i < len(ids) {
						realPGID = ids[i] / 10000
					}
				}
			}
			hits = append(hits, RelationVectorHit{
				PGID:    realPGID,
				Score:   float64(r.Scores[i]),
				Keyword: keyword,
			})
		}
	}
	return hits, nil
}

func (vs *milvusVectorStore) DeleteByDocHash(ctx interface{}, docHash string) error {
	if !vs.Available() {
		return nil
	}
	// Milvus 不支持直接按 doc_hash 删除（collection 里没有这个字段）。
	// 由调用方在 Neo4j 中找到 pgIDs 后调用。当前为空操作，
	// 实际删除发生在 Neo4j DeleteDocument 路径中。
	return nil
}

// ─────────────────────────────── 内部辅助 ──────────────────────────────────

func float64To32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, f := range v {
		out[i] = float32(f)
	}
	return out
}

func hashKeyword(kw string) int {
	h := 0
	for _, r := range kw {
		h = h*31 + int(r)
	}
	if h < 0 {
		h = -h
	}
	return h
}
