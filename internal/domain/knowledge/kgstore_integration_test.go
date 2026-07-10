//go:build integration

package knowledge

import (
	"alchemist-Assistant/config"
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
)

func TestKGV2Neo4jIntegration(t *testing.T) {
	if os.Getenv("KG_INTEGRATION") != "1" {
		t.Skip("set KG_INTEGRATION=1 to run Neo4j/Milvus integration tests")
	}
	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	docHash := "kg-v2-integration-" + suffix
	docHashB := docHash + "-b"
	systemName, productName := "系统A-"+suffix, "Kafka-"+suffix
	llmFn := func(_, _ string) string {
		return fmt.Sprintf(`{"entities":[{"name":%q,"type":"System"},{"name":%q,"type":"Product","aliases":["Apache Kafka %s"]}],"relations":[{"from":%q,"to":%q,"rel_type":"USES","confidence":0.9}]}`,
			systemName, productName, suffix, systemName, productName)
	}
	cfg := &config.APIConfig{StorageConfig: config.StorageConfig{Neo4jConfig: config.Neo4jConfig{
		Neo4jURI: envOr("NEO4J_URI", "bolt://localhost:7687"), Neo4jUser: envOr("NEO4J_USER", "neo4j"),
		Neo4jPassword: envOr("NEO4J_PASSWORD", "password123"), KGEnabled: true, KGGraphMaxHops: 2,
		KGEntitySeedTopK: 5, KGGraphHopDecay: .7, KGGraphEvidenceTopK: 20,
	}}}
	store := NewKGStore(cfg, llmFn)
	if !store.Available() {
		t.Skip("Neo4j is unavailable")
	}
	defer store.Close()
	defer store.DeleteDocument(docHash)
	defer store.DeleteDocument(docHashB)

	store.IndexDocument(docHash, []ChunkRef{{ID: 0, PGID: 910001, Content: "a"}, {ID: 1, PGID: 910002, Content: "b"}, {ID: 2, PGID: 910003, Content: "c"}})
	ctx := context.Background()
	session := store.Client().ReadSession()
	defer session.Close(ctx)
	record, err := session.Run(ctx, `MATCH (c:KGChunk {doc_hash:$doc})
		OPTIONAL MATCH (c)-[:HAS_MENTION]->(m:KGEntityMention)
		WITH count(DISTINCT c) AS chunks, count(DISTINCT m) AS mentions
		MATCH ()-[r:KG_RELATION {doc_hash:$doc}]->()
		RETURN chunks, mentions, count(r) AS relations`, map[string]any{"doc": docHash})
	if err != nil || !record.Next(ctx) {
		t.Fatalf("count KG V2 graph: %v", err)
	}
	rec := record.Record()
	if int64Value(rec, "chunks") != 3 || int64Value(rec, "mentions") != 6 || int64Value(rec, "relations") != 3 {
		t.Fatalf("unexpected graph counts: chunks=%d mentions=%d relations=%d", int64Value(rec, "chunks"), int64Value(rec, "mentions"), int64Value(rec, "relations"))
	}
	hits := store.Search(productName, 10)
	if len(hits) == 0 || hits[0].PGID < 910001 || hits[0].PGID > 910003 {
		t.Fatalf("evidence retrieval did not return relation pg_id: %#v", hits)
	}

	store.IndexDocument(docHashB, []ChunkRef{{ID: 0, PGID: 910004, Content: "shared entity in document B"}})
	store.DeleteDocument(docHash)
	preserved, err := session.Run(ctx, `MATCH (e:KGEntity {canonical_name:$name})
		MATCH (e)<-[:OF_ENTITY]-(m:KGEntityMention {doc_hash:$doc})
		OPTIONAL MATCH ()-[r:KG_RELATION {doc_hash:$doc}]->()
		RETURN count(DISTINCT e) AS entities, count(DISTINCT m) AS mentions, count(DISTINCT r) AS relations`,
		map[string]any{"name": productName, "doc": docHashB})
	if err != nil || !preserved.Next(ctx) {
		t.Fatalf("verify shared entity after deleting document A: %v", err)
	}
	preservedRecord := preserved.Record()
	if int64Value(preservedRecord, "entities") != 1 || int64Value(preservedRecord, "mentions") != 1 || int64Value(preservedRecord, "relations") != 1 {
		t.Fatalf("document B evidence was not preserved: entities=%d mentions=%d relations=%d",
			int64Value(preservedRecord, "entities"), int64Value(preservedRecord, "mentions"), int64Value(preservedRecord, "relations"))
	}
}

func TestKGV2MilvusIntegration(t *testing.T) {
	if os.Getenv("KG_INTEGRATION") != "1" {
		t.Skip("set KG_INTEGRATION=1 to run Neo4j/Milvus integration tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client, err := milvusClient.NewClient(ctx, milvusClient.Config{Address: envOr("MILVUS_ADDR", "localhost:19530")})
	if err != nil {
		t.Skipf("Milvus unavailable: %v", err)
	}
	defer client.Close()
	vectorStore := newMilvusEntityVectorStore(client, func(string) ([]float64, error) { return []float64{1, 0, 0, 0}, nil }, 4, "COSINE")
	vectorStore.collection = "kg_entities_integration_" + fmt.Sprintf("%d", time.Now().UnixNano())
	defer client.DropCollection(context.Background(), vectorStore.collection)
	if err := vectorStore.Init(); err != nil {
		t.Skipf("kg_entities is unavailable or belongs to another schema: %v", err)
	}
	entityID := CanonicalEntityID(EntityProduct, fmt.Sprintf("integration-%d", time.Now().UnixNano()))
	entity := CanonicalEntity{ID: entityID, CanonicalName: "Integration Kafka", NormalizedName: "integration kafka", Type: EntityProduct, Aliases: []string{"Kafka"}}
	if err := vectorStore.UpsertEntityVector(ctx, entity); err != nil {
		t.Fatal(err)
	}
	defer vectorStore.DeleteEntityVectors(context.Background(), []string{entityID})
	hits, err := vectorStore.SearchEntityVectors(ctx, []float32{1, 0, 0, 0}, 20)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, hit := range hits {
		if hit.EntityID == entityID {
			found = true
		}
	}
	if !found {
		t.Fatalf("upserted entity %s was not found: %#v", entityID, hits)
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
