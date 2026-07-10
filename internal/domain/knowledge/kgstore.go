package knowledge

import (
	"alchemist-Assistant/config"
	platformneo4j "alchemist-Assistant/internal/infrastructure/platform/neo4j"
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"time"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	driver "github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type KGStore struct {
	neo4j        *platformneo4j.Client
	extractor    *Extractor
	normalizer   EntityNormalizer
	vector       EntityVectorStore
	embedFn      func(string) ([]float64, error)
	seedTopK     int
	vectorTopK   int
	vectorMin    float64
	maxHops      int
	hopDecay     float64
	evidenceTopK int
	relWeights   map[string]float64
}

type KGStoreConfig struct {
	Config       *config.APIConfig
	LLMFn        func(systemPrompt, userMsg string) string
	MilvusClient milvusClient.Client
	EmbedFn      func(string) ([]float64, error)
}

func NewKGStore(cfg *config.APIConfig, llmFn func(systemPrompt, userMsg string) string) *KGStore {
	return NewKGStoreWithDeps(KGStoreConfig{Config: cfg, LLMFn: llmFn})
}

func NewKGStoreWithDeps(deps KGStoreConfig) *KGStore {
	cfg := deps.Config
	if cfg == nil {
		cfg = &config.APIConfig{}
	}
	ks := &KGStore{
		neo4j: platformneo4j.Connect(cfg.Neo4jConfig), extractor: NewExtractor(deps.LLMFn),
		normalizer: NewEntityNormalizer(), embedFn: deps.EmbedFn,
		seedTopK: cfg.KGEntitySeedTopK, vectorTopK: cfg.KGEntityVectorTopK,
		vectorMin: cfg.KGEntityVectorMinScore, maxHops: cfg.KGGraphMaxHops,
		hopDecay: cfg.KGGraphHopDecay, evidenceTopK: cfg.KGGraphEvidenceTopK,
		relWeights: cfg.KGRelationTypeWeights,
	}
	if ks.seedTopK <= 0 {
		ks.seedTopK = 5
	}
	if ks.vectorTopK <= 0 {
		ks.vectorTopK = 10
	}
	if ks.vectorMin <= 0 {
		ks.vectorMin = .65
	}
	if ks.maxHops <= 0 || ks.maxHops > 2 {
		ks.maxHops = 2
	}
	if ks.hopDecay <= 0 || ks.hopDecay > 1 {
		ks.hopDecay = .7
	}
	if ks.evidenceTopK <= 0 {
		ks.evidenceTopK = 20
	}
	if len(ks.relWeights) == 0 {
		ks.relWeights = defaultRelationWeights()
	}

	if deps.MilvusClient != nil && deps.EmbedFn != nil {
		vectorStore := newMilvusEntityVectorStore(deps.MilvusClient, deps.EmbedFn, cfg.RAGMilvusDim, cfg.KGEntityVectorMetric)
		if err := vectorStore.Init(); err != nil {
			logVectorInitError(err)
		} else {
			ks.vector = vectorStore
		}
	}
	return ks
}

func defaultRelationWeights() map[string]float64 {
	return map[string]float64{
		RelationCauses: 1, RelationDependsOn: 1, RelationUses: 1, RelationReplaces: 1,
		RelationProduces: .95, RelationPartOf: .95, RelationIsA: .95,
		RelationWorksFor: .9, RelationLocatedIn: .9, RelationDescribes: .8,
		RelationMentions: .7, RelationRelatesTo: .7,
	}
}

func (ks *KGStore) Available() bool { return ks != nil && ks.neo4j != nil && ks.neo4j.Available() }
func (ks *KGStore) VectorAvailable() bool {
	return ks != nil && ks.vector != nil && ks.vector.Available()
}
func (ks *KGStore) Close() {
	if ks != nil && ks.neo4j != nil {
		ks.neo4j.Close()
	}
}
func (ks *KGStore) Client() *platformneo4j.Client {
	if ks == nil {
		return nil
	}
	return ks.neo4j
}

type ChunkRef struct {
	ID      int
	PGID    int64
	Content string
}

func (ks *KGStore) IndexDocument(docHash string, chunks []ChunkRef) {
	if !ks.Available() {
		return
	}
	for _, chunk := range chunks {
		if chunk.PGID <= 0 {
			continue
		}
		extracted := ks.extractor.Extract(chunk.Content)
		if len(extracted.Entities) == 0 {
			continue
		}
		graph := buildChunkGraph(docHash, chunk, extracted, ks.normalizer)
		dirty, err := ks.writeChunkGraph(context.Background(), graph)
		if err != nil {
			log.Printf("⚠️  KG V2 chunk 写入失败 (doc=%s chunk=%d pg=%d): %v", docHash, chunk.ID, chunk.PGID, err)
			continue
		}
		if ks.VectorAvailable() {
			for _, entity := range dirty {
				ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				err := ks.vector.UpsertEntityVector(ctx, entity)
				cancel()
				if err != nil {
					log.Printf("⚠️  KG Entity Vector 写入失败 (%s): %v", entity.ID, err)
				}
			}
		}
	}
}

func (ks *KGStore) writeChunkGraph(ctx context.Context, graph chunkGraph) ([]CanonicalEntity, error) {
	session := ks.neo4j.Session()
	defer session.Close(ctx)
	return driver.ExecuteWrite(ctx, session, func(tx driver.ManagedTransaction) ([]CanonicalEntity, error) {
		resolved, remap, dirty, err := ks.resolveEntities(ctx, tx, graph.Entities)
		if err != nil {
			return nil, err
		}
		for i := range graph.Mentions {
			old := graph.Mentions[i].EntityID
			if id := remap[old]; id != "" {
				graph.Mentions[i].EntityID = id
				graph.Mentions[i].ID = MentionID(graph.Mentions[i].DocHash, graph.Mentions[i].ChunkID,
					graph.Mentions[i].PGID, id, graph.Mentions[i].NormalizedSurface)
			}
		}
		for i := range graph.Relations {
			rel := &graph.Relations[i]
			if id := remap[rel.FromEntityID]; id != "" {
				rel.FromEntityID = id
			}
			if id := remap[rel.ToEntityID]; id != "" {
				rel.ToEntityID = id
			}
			rel.ID = RelationID(rel.FromEntityID, rel.ToEntityID, rel.RelType, rel.DocHash, rel.ChunkID, rel.PGID)
		}

		if _, err := tx.Run(ctx, `MERGE (c:KGChunk {id:$id})
			ON CREATE SET c.pg_id=$pg_id, c.doc_hash=$doc_hash, c.chunk_id=$chunk_id, c.created_at=datetime()
			SET c.pg_id=$pg_id, c.doc_hash=$doc_hash, c.chunk_id=$chunk_id`, map[string]any{
			"id": graph.Chunk.ID, "pg_id": graph.Chunk.PGID, "doc_hash": graph.Chunk.DocHash, "chunk_id": graph.Chunk.ChunkID,
		}); err != nil {
			return nil, err
		}
		if _, err := tx.Run(ctx, `UNWIND $entities AS row
			MERGE (e:KGEntity {id:row.id})
			ON CREATE SET e.canonical_name=row.canonical_name, e.normalized_name=row.normalized_name,
			 e.type=row.type, e.aliases=row.aliases, e.alias_norms=row.alias_norms, e.created_at=datetime()
			SET e.aliases=reduce(acc=coalesce(e.aliases,[]), x IN row.aliases | CASE WHEN x IN acc THEN acc ELSE acc+x END),
			 e.alias_norms=reduce(acc=coalesce(e.alias_norms,[]), x IN row.alias_norms | CASE WHEN x IN acc THEN acc ELSE acc+x END),
			 e.updated_at=datetime()`, map[string]any{"entities": entityRows(resolved)}); err != nil {
			return nil, err
		}
		if _, err := tx.Run(ctx, `UNWIND $mentions AS row
			MATCH (c:KGChunk {id:row.chunk_id}), (e:KGEntity {id:row.entity_id})
			MERGE (m:KGEntityMention {id:row.id})
			ON CREATE SET m.surface=row.surface, m.normalized_surface=row.normalized_surface,
			 m.extracted_type=row.extracted_type, m.doc_hash=row.doc_hash, m.chunk_id=row.chunk_idx,
			 m.pg_id=row.pg_id, m.created_at=datetime()
			MERGE (c)-[:HAS_MENTION]->(m)
			MERGE (m)-[:OF_ENTITY]->(e)`, map[string]any{"mentions": mentionRows(graph.Chunk.ID, graph.Mentions)}); err != nil {
			return nil, err
		}
		if _, err := tx.Run(ctx, `UNWIND $relations AS row
			MATCH (a:KGEntity {id:row.from_id}), (b:KGEntity {id:row.to_id})
			MERGE (a)-[r:KG_RELATION {id:row.id}]->(b)
			ON CREATE SET r.rel_type=row.rel_type, r.doc_hash=row.doc_hash, r.chunk_id=row.chunk_id,
			 r.pg_id=row.pg_id, r.confidence=row.confidence, r.created_at=datetime()`,
			map[string]any{"relations": relationRows(graph.Relations)}); err != nil {
			return nil, err
		}
		return dirty, nil
	})
}

func (ks *KGStore) resolveEntities(ctx context.Context, tx driver.ManagedTransaction, inputs []CanonicalEntity) ([]CanonicalEntity, map[string]string, []CanonicalEntity, error) {
	resolved := make([]CanonicalEntity, 0, len(inputs))
	remap := make(map[string]string, len(inputs))
	dirty := make([]CanonicalEntity, 0, len(inputs))
	for _, input := range inputs {
		surfaces := append([]string{input.NormalizedName}, input.AliasNorms...)
		records, err := tx.Run(ctx, `MATCH (e:KGEntity) WHERE e.type=$type AND
			(e.normalized_name=$normalized OR any(x IN coalesce(e.alias_norms,[]) WHERE x IN $surfaces))
			RETURN e.id AS id, e.canonical_name AS canonical_name, e.normalized_name AS normalized_name,
			 e.type AS type, coalesce(e.aliases,[]) AS aliases, coalesce(e.alias_norms,[]) AS alias_norms,
			 e.normalized_name=$normalized AS exact ORDER BY exact DESC, e.id`, map[string]any{
			"type": string(input.Type), "normalized": input.NormalizedName, "surfaces": surfaces,
		})
		if err != nil {
			return nil, nil, nil, err
		}
		var candidates []CanonicalEntity
		var exact []bool
		for records.Next(ctx) {
			rec := records.Record()
			candidates = append(candidates, CanonicalEntity{
				ID: stringValue(rec, "id"), CanonicalName: stringValue(rec, "canonical_name"),
				NormalizedName: stringValue(rec, "normalized_name"), Type: EntityType(stringValue(rec, "type")),
				Aliases: stringSliceValue(rec, "aliases"), AliasNorms: stringSliceValue(rec, "alias_norms"),
			})
			exact = append(exact, boolValue(rec, "exact"))
		}
		if err := records.Err(); err != nil {
			return nil, nil, nil, err
		}
		chosen := input
		isDirty := true
		if len(candidates) > 0 && exact[0] {
			chosen = mergeEntityAliases(candidates[0], input)
			isDirty = aliasesChanged(candidates[0], chosen)
		} else if len(candidates) == 1 {
			chosen = mergeEntityAliases(candidates[0], input)
			isDirty = aliasesChanged(candidates[0], chosen)
		} else if len(candidates) > 1 {
			log.Printf("⚠️  KG alias 存在多个同类型候选，保守创建独立实体: %s (%s)", input.CanonicalName, input.Type)
		}
		remap[input.ID] = chosen.ID
		resolved = append(resolved, chosen)
		if isDirty {
			dirty = append(dirty, chosen)
		}
	}
	return resolved, remap, dirty, nil
}

func mergeEntityAliases(existing, incoming CanonicalEntity) CanonicalEntity {
	aliases := append([]string{}, existing.Aliases...)
	norms := append([]string{}, existing.AliasNorms...)
	seen := make(map[string]bool)
	for _, value := range norms {
		seen[value] = true
	}
	add := func(name, normalized string) {
		if normalized == "" || normalized == existing.NormalizedName || seen[normalized] {
			return
		}
		seen[normalized] = true
		aliases = append(aliases, name)
		norms = append(norms, normalized)
	}
	add(incoming.CanonicalName, incoming.NormalizedName)
	for i, name := range incoming.Aliases {
		if i < len(incoming.AliasNorms) {
			add(name, incoming.AliasNorms[i])
		}
	}
	existing.Aliases, existing.AliasNorms = aliases, norms
	return existing
}

func aliasesChanged(before, after CanonicalEntity) bool {
	return len(before.AliasNorms) != len(after.AliasNorms)
}

func (ks *KGStore) DeleteDocument(docHash string) {
	if !ks.Available() {
		return
	}
	ctx := context.Background()
	session := ks.neo4j.Session()
	defer session.Close(ctx)
	orphans, err := driver.ExecuteWrite(ctx, session, func(tx driver.ManagedTransaction) ([]string, error) {
		queries := []string{
			`MATCH ()-[r:KG_RELATION {doc_hash:$doc_hash}]->() DELETE r`,
			`MATCH (m:KGEntityMention {doc_hash:$doc_hash}) DETACH DELETE m`,
			`MATCH (c:KGChunk {doc_hash:$doc_hash}) DETACH DELETE c`,
		}
		for _, query := range queries {
			if _, err := tx.Run(ctx, query, map[string]any{"doc_hash": docHash}); err != nil {
				return nil, err
			}
		}
		records, err := tx.Run(ctx, `MATCH (e:KGEntity)
			WHERE NOT (e)<-[:OF_ENTITY]-(:KGEntityMention) AND NOT (e)-[:KG_RELATION]-()
			WITH collect(e.id) AS ids, collect(e) AS entities FOREACH (e IN entities | DELETE e) RETURN ids`, nil)
		if err != nil {
			return nil, err
		}
		if records.Next(ctx) {
			return stringSliceValue(records.Record(), "ids"), nil
		}
		return nil, records.Err()
	})
	if err != nil {
		log.Printf("⚠️  KG V2 删除文档失败 (%s): %v", docHash, err)
		return
	}
	if ks.VectorAvailable() && len(orphans) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		if err := ks.vector.DeleteEntityVectors(ctx, orphans); err != nil {
			log.Printf("⚠️  KG Entity Vector 删除失败: %v", err)
		}
		cancel()
	}
}

func (ks *KGStore) Search(queryText string, topK int) []GraphSearchResult {
	if !ks.Available() || strings.TrimSpace(queryText) == "" || topK <= 0 {
		return nil
	}
	extracted := ks.extractor.Extract(queryText)
	seeds := ks.exactSeeds(extracted.Entities)
	if ks.VectorAvailable() && ks.embedFn != nil {
		texts := []string{queryText}
		for _, entity := range extracted.Entities {
			texts = append(texts, entity.Name)
		}
		for _, text := range texts {
			emb, err := ks.embedFn(text)
			if err != nil || len(emb) == 0 {
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			hits, err := ks.vector.SearchEntityVectors(ctx, toFloat32(emb), ks.vectorTopK)
			cancel()
			if err != nil {
				log.Printf("⚠️  KG Entity Vector 查询失败: %v", err)
				continue
			}
			for _, hit := range hits {
				if hit.Score < ks.vectorMin {
					continue
				}
				score := applyTypeScore(hit.Score, hit.Type, extracted.Entities)
				seeds = append(seeds, EntitySeed{EntityID: hit.EntityID, CanonicalName: hit.CanonicalName, Type: hit.Type, Score: score, MatchType: "vector"})
			}
		}
	}
	seeds = mergeSeeds(seeds, ks.seedTopK)
	if len(seeds) == 0 {
		return nil
	}
	return ks.expandEvidence(seeds, topK)
}

func (ks *KGStore) exactSeeds(inputs []Entity) []EntitySeed {
	if len(inputs) == 0 {
		return nil
	}
	rows := make([]map[string]any, 0, len(inputs))
	for _, input := range inputs {
		rows = append(rows, map[string]any{"normalized": ks.normalizer.Normalize(input.Name), "type": string(input.Type)})
	}
	ctx := context.Background()
	session := ks.neo4j.ReadSession()
	defer session.Close(ctx)
	records, err := session.Run(ctx, `UNWIND $inputs AS input MATCH (e:KGEntity)
		WHERE e.normalized_name=input.normalized OR any(x IN coalesce(e.alias_norms,[]) WHERE x=input.normalized)
		WITH e,input, CASE WHEN e.normalized_name=input.normalized THEN 1.0 ELSE 0.95 END AS base
		RETURN e.id AS id, e.canonical_name AS name, e.type AS type,
		CASE WHEN input.type='Unknown' THEN base WHEN e.type=input.type THEN CASE WHEN base+0.05>1 THEN 1 ELSE base+0.05 END ELSE base*0.8 END AS score,
		CASE WHEN e.normalized_name=input.normalized THEN 'canonical' ELSE 'alias' END AS match_type`, map[string]any{"inputs": rows})
	if err != nil {
		log.Printf("⚠️  KG 名称/别名种子查询失败: %v", err)
		return nil
	}
	var seeds []EntitySeed
	for records.Next(ctx) {
		rec := records.Record()
		seeds = append(seeds, EntitySeed{
			EntityID: stringValue(rec, "id"), CanonicalName: stringValue(rec, "name"), Type: EntityType(stringValue(rec, "type")),
			Score: floatValue(rec, "score"), MatchType: stringValue(rec, "match_type"),
		})
	}
	return seeds
}

type evidenceAccumulator struct {
	chunkID      int
	scores       []float64
	entities     []string
	path         []string
	evidenceType string
	matchedSeed  string
}

func (ks *KGStore) expandEvidence(seeds []EntitySeed, topK int) []GraphSearchResult {
	ctx := context.Background()
	session := ks.neo4j.ReadSession()
	defer session.Close(ctx)
	seedRows := make([]map[string]any, 0, len(seeds))
	seedByID := make(map[string]EntitySeed)
	for _, seed := range seeds {
		seedRows = append(seedRows, map[string]any{"id": seed.EntityID})
		seedByID[seed.EntityID] = seed
	}
	acc := make(map[int64]*evidenceAccumulator)
	add := func(pgID int64, chunkID int, score float64, evidenceType, seed string, entities, path []string) {
		if pgID <= 0 {
			return
		}
		item := acc[pgID]
		if item == nil {
			item = &evidenceAccumulator{chunkID: chunkID}
			acc[pgID] = item
		}
		item.scores = append(item.scores, clamp01(score))
		if evidencePriority(evidenceType) > evidencePriority(item.evidenceType) {
			item.evidenceType = evidenceType
			item.matchedSeed = seed
			item.entities = entities
			item.path = path
		}
	}

	mentionRecords, err := session.Run(ctx, `UNWIND $seeds AS input MATCH (e:KGEntity {id:input.id})<-[:OF_ENTITY]-(m:KGEntityMention)<-[:HAS_MENTION]-(c:KGChunk)
		RETURN input.id AS seed_id, c.pg_id AS pg_id, c.chunk_id AS chunk_id`, map[string]any{"seeds": seedRows})
	if err == nil {
		for mentionRecords.Next(ctx) {
			rec := mentionRecords.Record()
			seed := seedByID[stringValue(rec, "seed_id")]
			add(int64Value(rec, "pg_id"), intValue(rec, "chunk_id"), seed.Score, "seed_mention", seed.CanonicalName, []string{seed.CanonicalName}, nil)
		}
	}

	pathRecords, err := session.Run(ctx, `UNWIND $seeds AS input MATCH (seed:KGEntity {id:input.id})
		MATCH p=(seed)-[:KG_RELATION*1..2]-(neighbor:KGEntity) WHERE length(p)<=$hops
		RETURN input.id AS seed_id, [n IN nodes(p)|n.canonical_name] AS names,
		 [r IN relationships(p)|{id:r.id,pg_id:r.pg_id,chunk_id:r.chunk_id,rel_type:r.rel_type,confidence:coalesce(r.confidence,1.0)}] AS edges`,
		map[string]any{"seeds": seedRows, "hops": int64(ks.maxHops)})
	if err == nil {
		seenEvidence := make(map[string]bool)
		for pathRecords.Next(ctx) {
			rec := pathRecords.Record()
			seed := seedByID[stringValue(rec, "seed_id")]
			names := stringSliceValue(rec, "names")
			edges := mapSliceValue(rec, "edges")
			score := ks.scorePath(seed.Score, edges)
			for _, edge := range edges {
				evidenceID := stringAny(edge["id"])
				key := seed.EntityID + "|" + evidenceID + "|" + strings.Join(names, "->")
				if seenEvidence[key] {
					continue
				}
				seenEvidence[key] = true
				add(int64Any(edge["pg_id"]), intAny(edge["chunk_id"]), score, "relation", seed.CanonicalName, names, names)
			}
		}
	}

	neighborRecords, err := session.Run(ctx, `UNWIND $seeds AS input MATCH (seed:KGEntity {id:input.id})
		MATCH p=(seed)-[:KG_RELATION*1..2]-(neighbor:KGEntity) WHERE length(p)<=$hops
		MATCH (neighbor)<-[:OF_ENTITY]-(m:KGEntityMention)<-[:HAS_MENTION]-(c:KGChunk)
		RETURN input.id AS seed_id, c.pg_id AS pg_id, c.chunk_id AS chunk_id,
		 [n IN nodes(p)|n.canonical_name] AS names,
		 [r IN relationships(p)|{rel_type:r.rel_type,confidence:coalesce(r.confidence,1.0)}] AS edges
		 LIMIT $limit`, map[string]any{"seeds": seedRows, "hops": int64(ks.maxHops), "limit": int64(ks.evidenceTopK * 3)})
	if err == nil {
		for neighborRecords.Next(ctx) {
			rec := neighborRecords.Record()
			seed := seedByID[stringValue(rec, "seed_id")]
			names := stringSliceValue(rec, "names")
			edges := mapSliceValue(rec, "edges")
			add(int64Value(rec, "pg_id"), intValue(rec, "chunk_id"), ks.scorePath(seed.Score, edges)*.8, "neighbor_mention", seed.CanonicalName, names, names)
		}
	}

	results := make([]GraphSearchResult, 0, len(acc))
	for pgID, item := range acc {
		results = append(results, GraphSearchResult{ChunkID: item.chunkID, PGID: pgID, Score: aggregateEvidenceScore(item.scores), Entities: item.entities, HopPath: item.path, EvidenceType: item.evidenceType, MatchedSeed: item.matchedSeed})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].PGID < results[j].PGID
		}
		return results[i].Score > results[j].Score
	})
	limit := topK
	if limit > ks.evidenceTopK {
		limit = ks.evidenceTopK
	}
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

func applyTypeScore(score float64, hitType EntityType, inputs []Entity) float64 {
	hasKnown, matched := false, false
	for _, input := range inputs {
		if input.Type == EntityUnknown {
			continue
		}
		hasKnown = true
		if input.Type == hitType {
			matched = true
		}
	}
	if matched {
		score += .05
	} else if hasKnown {
		score *= .8
	}
	return clamp01(score)
}
func (ks *KGStore) relationWeight(relType string) float64 {
	if w := ks.relWeights[relType]; w > 0 {
		return w
	}
	return 1
}

func (ks *KGStore) scorePath(seedScore float64, edges []map[string]any) float64 {
	score := seedScore * math.Pow(ks.hopDecay, float64(len(edges)))
	for _, edge := range edges {
		score *= ks.relationWeight(stringAny(edge["rel_type"])) * clamp01(floatAny(edge["confidence"]))
	}
	return clamp01(score)
}
func evidencePriority(kind string) int {
	switch kind {
	case "relation":
		return 3
	case "seed_mention":
		return 2
	case "neighbor_mention":
		return 1
	}
	return 0
}

func entityRows(values []CanonicalEntity) []map[string]any {
	out := make([]map[string]any, 0, len(values))
	for _, v := range values {
		out = append(out, map[string]any{"id": v.ID, "canonical_name": v.CanonicalName, "normalized_name": v.NormalizedName, "type": string(v.Type), "aliases": v.Aliases, "alias_norms": v.AliasNorms})
	}
	return out
}
func mentionRows(chunkID string, values []EntityMention) []map[string]any {
	out := make([]map[string]any, 0, len(values))
	for _, v := range values {
		out = append(out, map[string]any{"id": v.ID, "entity_id": v.EntityID, "surface": v.Surface, "normalized_surface": v.NormalizedSurface, "extracted_type": string(v.ExtractedType), "doc_hash": v.DocHash, "chunk_id": chunkID, "chunk_idx": v.ChunkID, "pg_id": v.PGID})
	}
	return out
}
func relationRows(values []RelationEvidence) []map[string]any {
	out := make([]map[string]any, 0, len(values))
	for _, v := range values {
		out = append(out, map[string]any{"id": v.ID, "from_id": v.FromEntityID, "to_id": v.ToEntityID, "rel_type": v.RelType, "doc_hash": v.DocHash, "chunk_id": v.ChunkID, "pg_id": v.PGID, "confidence": v.Confidence})
	}
	return out
}

func recordValue(rec *driver.Record, key string) any    { value, _ := rec.Get(key); return value }
func stringValue(rec *driver.Record, key string) string { return stringAny(recordValue(rec, key)) }
func boolValue(rec *driver.Record, key string) bool {
	value, _ := recordValue(rec, key).(bool)
	return value
}
func floatValue(rec *driver.Record, key string) float64 { return floatAny(recordValue(rec, key)) }
func intValue(rec *driver.Record, key string) int       { return intAny(recordValue(rec, key)) }
func int64Value(rec *driver.Record, key string) int64   { return int64Any(recordValue(rec, key)) }
func stringSliceValue(rec *driver.Record, key string) []string {
	value := recordValue(rec, key)
	switch x := value.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, v := range x {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
func mapSliceValue(rec *driver.Record, key string) []map[string]any {
	value := recordValue(rec, key)
	switch x := value.(type) {
	case []map[string]any:
		return x
	case []any:
		out := make([]map[string]any, 0, len(x))
		for _, v := range x {
			if m, ok := v.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	return nil
}
func stringAny(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}
func floatAny(value any) float64 {
	switch x := value.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int64:
		return float64(x)
	case int:
		return float64(x)
	}
	return 0
}
func intAny(value any) int {
	switch x := value.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	}
	return 0
}
func int64Any(value any) int64 {
	switch x := value.(type) {
	case int64:
		return x
	case int:
		return int64(x)
	case float64:
		return int64(x)
	}
	return 0
}
func intStr(n int) string { return fmt.Sprintf("%d", n) }
