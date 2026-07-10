package knowledge

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
	"unicode/utf8"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const kgEntityCollection = "kg_entities"

type milvusEntityVectorStore struct {
	client     milvusClient.Client
	embedFn    func(string) ([]float64, error)
	dim        int
	metric     string
	collection string
	ready      bool
}

func newMilvusEntityVectorStore(client milvusClient.Client, embedFn func(string) ([]float64, error), dim int, metric string) *milvusEntityVectorStore {
	if dim <= 0 {
		dim = 1024
	}
	metric = strings.ToUpper(metric)
	if metric != "COSINE" && metric != "IP" && metric != "L2" {
		metric = "COSINE"
	}
	return &milvusEntityVectorStore{client: client, embedFn: embedFn, dim: dim, metric: metric, collection: kgEntityCollection}
}

func (s *milvusEntityVectorStore) Available() bool {
	return s != nil && s.client != nil && s.embedFn != nil && s.ready
}

func (s *milvusEntityVectorStore) Init() error {
	if s == nil || s.client == nil || s.embedFn == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	has, err := s.client.HasCollection(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("check KG entity collection: %w", err)
	}
	if has {
		collection, err := s.client.DescribeCollection(ctx, s.collection)
		if err != nil {
			return fmt.Errorf("describe KG entity collection: %w", err)
		}
		compatible := false
		for _, field := range collection.Schema.Fields {
			if field.Name == "entity_id" && field.PrimaryKey && field.DataType == entity.FieldTypeVarChar {
				compatible = true
			}
			if field.Name == "embedding" && field.TypeParams["dim"] != fmt.Sprintf("%d", s.dim) {
				return fmt.Errorf("kg_entities embedding dim is %s, expected %d", field.TypeParams["dim"], s.dim)
			}
		}
		if !compatible {
			return fmt.Errorf("kg_entities exists with a non-KG-V2 primary key; refusing to overwrite it")
		}
		if err := s.client.LoadCollection(ctx, s.collection, false); err != nil {
			return fmt.Errorf("load KG entity collection: %w", err)
		}
		s.ready = true
		return nil
	}

	schema := &entity.Schema{CollectionName: s.collection, Fields: []*entity.Field{
		{Name: "entity_id", DataType: entity.FieldTypeVarChar, PrimaryKey: true, AutoID: false, TypeParams: map[string]string{"max_length": "64"}},
		{Name: "canonical_name", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "512"}},
		{Name: "entity_type", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "64"}},
		{Name: "aliases_text", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "4096"}},
		{Name: "embedding", DataType: entity.FieldTypeFloatVector, TypeParams: map[string]string{"dim": fmt.Sprintf("%d", s.dim)}},
	}}
	if err := s.client.CreateCollection(ctx, schema, 1); err != nil {
		return fmt.Errorf("create KG entity collection: %w", err)
	}
	index, err := entity.NewIndexIvfFlat(metricType(s.metric), 128)
	if err != nil {
		return fmt.Errorf("create KG vector index config: %w", err)
	}
	if err := s.client.CreateIndex(ctx, s.collection, "embedding", index, false); err != nil {
		return fmt.Errorf("create KG vector index: %w", err)
	}
	if err := s.client.LoadCollection(ctx, s.collection, false); err != nil {
		return fmt.Errorf("load KG entity collection: %w", err)
	}
	s.ready = true
	return nil
}

func (s *milvusEntityVectorStore) UpsertEntityVector(ctx context.Context, value CanonicalEntity) error {
	if !s.Available() {
		return nil
	}
	text := fmt.Sprintf("名称: %s\n类型: %s\n别名: %s", value.CanonicalName, value.Type, strings.Join(value.Aliases, ", "))
	embedding, err := s.embedFn(text)
	if err != nil {
		return fmt.Errorf("embed KG entity %s: %w", value.ID, err)
	}
	if len(embedding) != s.dim {
		return fmt.Errorf("embed KG entity %s: dimension %d, expected %d", value.ID, len(embedding), s.dim)
	}
	aliases := truncateUTF8(strings.Join(value.Aliases, "\n"), 4096)
	_, err = s.client.Upsert(ctx, s.collection, "",
		entity.NewColumnVarChar("entity_id", []string{value.ID}),
		entity.NewColumnVarChar("canonical_name", []string{truncateUTF8(value.CanonicalName, 512)}),
		entity.NewColumnVarChar("entity_type", []string{string(value.Type)}),
		entity.NewColumnVarChar("aliases_text", []string{aliases}),
		entity.NewColumnFloatVector("embedding", s.dim, [][]float32{toFloat32(embedding)}),
	)
	if err != nil {
		return fmt.Errorf("upsert KG entity vector %s: %w", value.ID, err)
	}
	return nil
}

func (s *milvusEntityVectorStore) SearchEntityVectors(ctx context.Context, vector []float32, topK int) ([]EntityVectorHit, error) {
	if !s.Available() || len(vector) == 0 || topK <= 0 {
		return nil, nil
	}
	param, _ := entity.NewIndexFlatSearchParam()
	results, err := s.client.Search(ctx, s.collection, nil, "",
		[]string{"canonical_name", "entity_type"}, []entity.Vector{entity.FloatVector(vector)},
		"embedding", metricType(s.metric), topK, param)
	if err != nil {
		return nil, fmt.Errorf("search KG entity vectors: %w", err)
	}
	hits := make([]EntityVectorHit, 0, topK)
	for _, result := range results {
		ids := result.IDs
		nameCol := result.Fields.GetColumn("canonical_name")
		typeCol := result.Fields.GetColumn("entity_type")
		for i, rawScore := range result.Scores {
			id, err := ids.GetAsString(i)
			if err != nil {
				continue
			}
			name, _ := nameCol.GetAsString(i)
			typeName, _ := typeCol.GetAsString(i)
			hits = append(hits, EntityVectorHit{EntityID: id, CanonicalName: name,
				Type: NormalizeEntityType(EntityType(typeName)), Score: normalizeVectorScore(s.metric, float64(rawScore))})
		}
	}
	return hits, nil
}

func (s *milvusEntityVectorStore) DeleteEntityVectors(ctx context.Context, ids []string) error {
	if !s.Available() || len(ids) == 0 {
		return nil
	}
	quoted := make([]string, 0, len(ids))
	for _, id := range ids {
		quoted = append(quoted, fmt.Sprintf("%q", id))
	}
	if err := s.client.Delete(ctx, s.collection, "", "entity_id in ["+strings.Join(quoted, ",")+"]"); err != nil {
		return fmt.Errorf("delete KG entity vectors: %w", err)
	}
	return nil
}

func metricType(metric string) entity.MetricType {
	switch strings.ToUpper(metric) {
	case "IP":
		return entity.IP
	case "L2":
		return entity.L2
	default:
		return entity.COSINE
	}
}

func toFloat32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = float32(value)
	}
	return out
}

func truncateUTF8(value string, maxBytes int) string {
	if maxBytes <= 0 || len(value) <= maxBytes {
		return value
	}
	for maxBytes > 0 && !utf8.ValidString(value[:maxBytes]) {
		maxBytes--
	}
	return value[:maxBytes]
}

func logVectorInitError(err error) {
	if err != nil {
		log.Printf("⚠️  KG Entity Vector 不可用，保留名称/别名检索: %v", err)
	}
}
