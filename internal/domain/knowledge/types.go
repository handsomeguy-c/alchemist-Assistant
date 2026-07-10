// Package knowledge implements the document knowledge graph used by RAG.
// It deliberately uses KG-prefixed labels so it cannot collide with Memory Graph.
package knowledge

import (
	"context"
	"time"
)

type EntityType string

const (
	EntityPerson     EntityType = "Person"
	EntityOrg        EntityType = "Organization"
	EntityLocation   EntityType = "Location"
	EntityConcept    EntityType = "Concept"
	EntityEvent      EntityType = "Event"
	EntityProduct    EntityType = "Product"
	EntityTechnology EntityType = "Technology"
	EntitySystem     EntityType = "System"
	EntityVersion    EntityType = "Version"
	EntityMetric     EntityType = "Metric"
	EntityUnknown    EntityType = "Unknown"
)

const (
	RelationRelatesTo = "RELATES_TO"
	RelationIsA       = "IS_A"
	RelationPartOf    = "PART_OF"
	RelationUses      = "USES"
	RelationDependsOn = "DEPENDS_ON"
	RelationCauses    = "CAUSES"
	RelationReplaces  = "REPLACES"
	RelationProduces  = "PRODUCES"
	RelationDescribes = "DESCRIBES"
	RelationMentions  = "MENTIONS"
	RelationWorksFor  = "WORKS_FOR"
	RelationLocatedIn = "LOCATED_IN"
)

// Entity is the extractor wire type. Provenance is filled by KGStore.
type Entity struct {
	Name       string     `json:"name"`
	Type       EntityType `json:"type"`
	Aliases    []string   `json:"aliases,omitempty"`
	Confidence float64    `json:"confidence,omitempty"`
	DocHash    string     `json:"doc_hash,omitempty"`
	ChunkID    int        `json:"chunk_id,omitempty"`
	PGID       int64      `json:"pg_id,omitempty"`
}

type Relation struct {
	FromName   string  `json:"from"`
	ToName     string  `json:"to"`
	RelType    string  `json:"rel_type"`
	Confidence float64 `json:"confidence,omitempty"`
	Weight     float64 `json:"weight,omitempty"` // retained for wire compatibility
	DocHash    string  `json:"doc_hash,omitempty"`
	ChunkID    int     `json:"chunk_id,omitempty"`
	PGID       int64   `json:"pg_id,omitempty"`
}

type CanonicalEntity struct {
	ID             string
	CanonicalName  string
	NormalizedName string
	Type           EntityType
	Aliases        []string
	AliasNorms     []string
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

type EntityMention struct {
	ID                string
	EntityID          string
	Surface           string
	NormalizedSurface string
	ExtractedType     EntityType
	DocHash           string
	ChunkID           int
	PGID              int64
}

type KGChunkRef struct {
	ID      string
	PGID    int64
	DocHash string
	ChunkID int
}

type RelationEvidence struct {
	ID           string
	FromEntityID string
	ToEntityID   string
	RelType      string
	DocHash      string
	ChunkID      int
	PGID         int64
	Confidence   float64
}

type EntitySeed struct {
	EntityID      string
	CanonicalName string
	Type          EntityType
	Score         float64
	MatchType     string
}

type EntityVectorHit struct {
	EntityID      string
	CanonicalName string
	Type          EntityType
	Score         float64
}

type GraphSearchResult struct {
	ChunkID      int      `json:"chunk_id"`
	PGID         int64    `json:"pg_id"`
	Score        float64  `json:"score"`
	Entities     []string `json:"entities"`
	HopPath      []string `json:"hop_path"`
	EvidenceType string   `json:"evidence_type,omitempty"`
	MatchedSeed  string   `json:"matched_seed,omitempty"`
}

type ExtractResult struct {
	Entities  []Entity   `json:"entities"`
	Relations []Relation `json:"relations"`
}

type EntityVectorStore interface {
	UpsertEntityVector(context.Context, CanonicalEntity) error
	SearchEntityVectors(context.Context, []float32, int) ([]EntityVectorHit, error)
	DeleteEntityVectors(context.Context, []string) error
	Available() bool
}
