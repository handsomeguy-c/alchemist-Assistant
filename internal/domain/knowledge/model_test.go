package knowledge

import (
	"alchemist-Assistant/config"
	"math"
	"testing"
	"unicode/utf8"
)

func TestBuildChunkGraphProducesStableMentionsAndEvidence(t *testing.T) {
	result := ExtractResult{
		Entities: []Entity{
			{Name: "系统A", Type: EntitySystem},
			{Name: "Kafka", Type: EntityProduct, Aliases: []string{"Apache Kafka"}},
		},
		Relations: []Relation{{FromName: "系统A", ToName: "Kafka", RelType: RelationUses, Confidence: .8}},
	}
	g := buildChunkGraph("doc", ChunkRef{ID: 2, PGID: 101, Content: "x"}, result, NewEntityNormalizer())
	if g.Chunk.ID != "pg:101" || len(g.Entities) != 2 || len(g.Mentions) != 2 || len(g.Relations) != 1 {
		t.Fatalf("unexpected chunk graph: %#v", g)
	}
	if g.Relations[0].PGID != 101 || g.Relations[0].ID == "" {
		t.Fatalf("relation evidence missing: %#v", g.Relations[0])
	}

	again := buildChunkGraph("doc", ChunkRef{ID: 2, PGID: 101, Content: "x"}, result, NewEntityNormalizer())
	if again.Mentions[0].ID != g.Mentions[0].ID || again.Relations[0].ID != g.Relations[0].ID {
		t.Fatal("repeated ingestion must generate identical IDs")
	}
}

func TestTruncateUTF8DoesNotSplitRune(t *testing.T) {
	got := truncateUTF8("中文abc", 4)
	if got != "中" || !utf8.ValidString(got) {
		t.Fatalf("truncateUTF8 returned %q", got)
	}
}

func TestBuildChunkGraphSeparatesSameNameDifferentTypes(t *testing.T) {
	g := buildChunkGraph("doc", ChunkRef{ID: 0, PGID: 1}, ExtractResult{Entities: []Entity{
		{Name: "Apple", Type: EntityOrg},
		{Name: "Apple", Type: EntityProduct},
	}}, NewEntityNormalizer())
	if len(g.Entities) != 2 || g.Entities[0].ID == g.Entities[1].ID {
		t.Fatalf("same name with different types must remain separate: %#v", g.Entities)
	}
}

func TestEvidenceAggregationIsBounded(t *testing.T) {
	if got := aggregateEvidenceScore([]float64{0.6, 0.6, 0.6}); got <= 0.6 || got > 1 {
		t.Fatalf("aggregate score=%v, want bounded multi-evidence bonus", got)
	}
}

func TestNormalizeVectorScore(t *testing.T) {
	if got := normalizeVectorScore("L2", 1); got != 0.5 {
		t.Fatalf("L2 score=%v", got)
	}
	if got := normalizeVectorScore("COSINE", -1); got != 0 {
		t.Fatalf("cosine score=%v", got)
	}
	if got := normalizeVectorScore("COSINE", 1); got != 1 {
		t.Fatalf("cosine score=%v", got)
	}
}

func TestMergeSeedsDeduplicatesAndKeepsBestScore(t *testing.T) {
	got := mergeSeeds([]EntitySeed{
		{EntityID: "a", Score: .7, MatchType: "vector"},
		{EntityID: "a", Score: 1, MatchType: "canonical"},
		{EntityID: "b", Score: .8, MatchType: "alias"},
	}, 2)
	if len(got) != 2 || got[0].EntityID != "a" || got[0].Score != 1 || got[0].MatchType != "canonical" {
		t.Fatalf("unexpected merged seeds: %#v", got)
	}
}

func TestApplyTypeScoreBonusesMatchAndDownweightsConflict(t *testing.T) {
	queryEntities := []Entity{{Name: "Kafka", Type: EntityProduct}}
	if got := applyTypeScore(.8, EntityProduct, queryEntities); math.Abs(got-.85) > 1e-12 {
		t.Fatalf("matching type score=%v", got)
	}
	if got := applyTypeScore(.8, EntitySystem, queryEntities); math.Abs(got-.64) > 1e-12 {
		t.Fatalf("conflicting type score=%v", got)
	}
}

func TestPathScoreUsesHopDecayRelationWeightAndConfidence(t *testing.T) {
	store := &KGStore{hopDecay: .7, relWeights: map[string]float64{RelationMentions: .5}}
	edges := []map[string]any{{"rel_type": RelationMentions, "confidence": .8}}
	want := .9 * .7 * .5 * .8
	if got := store.scorePath(.9, edges); math.Abs(got-want) > 1e-12 {
		t.Fatalf("path score=%v, want %v", got, want)
	}
}

func TestKGStoreGracefullyDegradesWithoutNeo4jOrMilvus(t *testing.T) {
	store := NewKGStore(&config.APIConfig{}, nil)
	if store.Available() || store.VectorAvailable() {
		t.Fatal("disabled infrastructure must be unavailable")
	}
	store.IndexDocument("doc", []ChunkRef{{ID: 0, PGID: 1, Content: "Kafka"}})
	store.DeleteDocument("doc")
	if got := store.Search("Kafka", 5); len(got) != 0 {
		t.Fatalf("degraded search returned results: %#v", got)
	}
}
