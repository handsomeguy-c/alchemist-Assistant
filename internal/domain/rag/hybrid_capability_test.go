package rag

import (
	"alchemist-Assistant/config"
	"alchemist-Assistant/internal/domain/knowledge"
	"alchemist-Assistant/internal/infrastructure/persistence/ragchunk"
	"errors"
	"math"
	"testing"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
)

type capabilityRepo struct {
	pg, milvus, es bool
	rows           map[int64]ragchunk.Row
}

func (r *capabilityRepo) SavePG(string, int, string, []byte) (int64, error) { return 0, nil }
func (r *capabilityRepo) SavePGWithParent(string, int, string, string, []byte) (int64, error) {
	return 0, nil
}
func (r *capabilityRepo) IndexES(int64, string, string, int) error          { return nil }
func (r *capabilityRepo) InsertMilvus([]int64, []string, [][]float32) error { return nil }
func (r *capabilityRepo) LoadAll() ([]ragchunk.Row, error)                  { return nil, nil }
func (r *capabilityRepo) LoadByIDs(ids []int64) ([]ragchunk.Row, error) {
	out := make([]ragchunk.Row, 0, len(ids))
	for _, id := range ids {
		if row, ok := r.rows[id]; ok {
			out = append(out, row)
		}
	}
	return out, nil
}
func (r *capabilityRepo) SearchES(string, int) ([]ragchunk.ESHit, error) {
	if !r.es {
		return nil, errors.New("no es")
	}
	return []ragchunk.ESHit{{PGID: 1, Score: 10}}, nil
}
func (r *capabilityRepo) SearchMilvus([]float32, int) ([]ragchunk.MilvusHit, error) {
	if !r.milvus {
		return nil, errors.New("no milvus")
	}
	return []ragchunk.MilvusHit{{ID: 1, Distance: .1}}, nil
}
func (r *capabilityRepo) Delete(string) error               { return nil }
func (r *capabilityRepo) Init(int)                          {}
func (r *capabilityRepo) PGAvailable() bool                 { return r.pg }
func (r *capabilityRepo) MilvusAvailable() bool             { return r.milvus }
func (r *capabilityRepo) ESAvailable() bool                 { return r.es }
func (r *capabilityRepo) MilvusClient() milvusClient.Client { return nil }

type capabilityKG struct{ hits []knowledge.GraphSearchResult }

func (k capabilityKG) Available() bool                                  { return true }
func (k capabilityKG) Search(string, int) []knowledge.GraphSearchResult { return k.hits }

func TestSearchOneCombinesKeywordAndKGWithoutMilvus(t *testing.T) {
	repo := &capabilityRepo{pg: true, es: true, rows: map[int64]ragchunk.Row{1: {ID: 1, Content: "es"}, 2: {ID: 2, Content: "kg"}}}
	hs := NewHybridStore(&config.APIConfig{RAGConfig: config.RAGConfig{RRFConstantK: 60}, StorageConfig: config.StorageConfig{Neo4jConfig: config.Neo4jConfig{KGWeight: .3}}}, repo)
	hs.kg = capabilityKG{hits: []knowledge.GraphSearchResult{{PGID: 2, Score: .99}}}
	got := hs.searchOne("q", 10, false)
	if len(got) != 2 {
		t.Fatalf("got %d results, want ES+KG", len(got))
	}
}

func TestSearchOneSupportsKGOnly(t *testing.T) {
	repo := &capabilityRepo{pg: true, rows: map[int64]ragchunk.Row{2: {ID: 2, Content: "kg"}}}
	hs := NewHybridStore(&config.APIConfig{RAGConfig: config.RAGConfig{RRFConstantK: 60}, StorageConfig: config.StorageConfig{Neo4jConfig: config.Neo4jConfig{KGWeight: .3}}}, repo)
	hs.kg = capabilityKG{hits: []knowledge.GraphSearchResult{{PGID: 2, Score: .01}}}
	got := hs.searchOne("q", 10, false)
	if len(got) != 1 || got[0].Chunk.Content != "kg" {
		t.Fatalf("unexpected KG-only results: %#v", got)
	}
	want := .3 / 61.0
	if math.Abs(got[0].Score-want) > 1e-12 {
		t.Fatalf("KG RRF score=%v, want %v; KG internal score must not be weighted", got[0].Score, want)
	}
}
