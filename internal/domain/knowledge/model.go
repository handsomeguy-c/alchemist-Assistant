package knowledge

import (
	"math"
	"sort"
	"strings"
)

type chunkGraph struct {
	Chunk     KGChunkRef
	Entities  []CanonicalEntity
	Mentions  []EntityMention
	Relations []RelationEvidence
}

func buildChunkGraph(docHash string, chunk ChunkRef, extracted ExtractResult, normalizer EntityNormalizer) chunkGraph {
	g := chunkGraph{Chunk: KGChunkRef{
		ID: KGChunkID(chunk.PGID), PGID: chunk.PGID, DocHash: docHash, ChunkID: chunk.ID,
	}}
	entityBySurface := make(map[string]CanonicalEntity)
	for _, input := range extracted.Entities {
		canonical := normalizer.DisplayName(input.Name)
		normalized := normalizer.Normalize(canonical)
		if normalized == "" {
			continue
		}
		entityType := NormalizeEntityType(input.Type)
		aliases, aliasNorms := normalizedAliases(normalizer, canonical, input.Aliases)
		entity := CanonicalEntity{
			ID: CanonicalEntityID(entityType, normalized), CanonicalName: canonical,
			NormalizedName: normalized, Type: entityType, Aliases: aliases, AliasNorms: aliasNorms,
		}
		g.Entities = append(g.Entities, entity)
		g.Mentions = append(g.Mentions, EntityMention{
			ID:       MentionID(docHash, chunk.ID, chunk.PGID, entity.ID, normalized),
			EntityID: entity.ID, Surface: canonical, NormalizedSurface: normalized,
			ExtractedType: entityType, DocHash: docHash, ChunkID: chunk.ID, PGID: chunk.PGID,
		})
		entityBySurface[normalized] = entity
		for _, aliasNorm := range aliasNorms {
			entityBySurface[aliasNorm] = entity
		}
	}
	for _, rel := range extracted.Relations {
		from, fromOK := entityBySurface[normalizer.Normalize(rel.FromName)]
		to, toOK := entityBySurface[normalizer.Normalize(rel.ToName)]
		if !fromOK || !toOK {
			continue
		}
		relType := NormalizeRelationType(rel.RelType)
		confidence := clampConfidence(rel.Confidence)
		g.Relations = append(g.Relations, RelationEvidence{
			ID:           RelationID(from.ID, to.ID, relType, docHash, chunk.ID, chunk.PGID),
			FromEntityID: from.ID, ToEntityID: to.ID, RelType: relType,
			DocHash: docHash, ChunkID: chunk.ID, PGID: chunk.PGID, Confidence: confidence,
		})
	}
	return g
}

func normalizedAliases(n EntityNormalizer, canonical string, inputs []string) ([]string, []string) {
	canonicalNorm := n.Normalize(canonical)
	seen := map[string]bool{canonicalNorm: true}
	aliases := make([]string, 0, len(inputs))
	norms := make([]string, 0, len(inputs))
	for _, input := range inputs {
		alias := n.DisplayName(input)
		normalized := n.Normalize(alias)
		if normalized == "" || seen[normalized] {
			continue
		}
		seen[normalized] = true
		aliases = append(aliases, alias)
		norms = append(norms, normalized)
	}
	return aliases, norms
}

func aggregateEvidenceScore(scores []float64) float64 {
	remaining := 1.0
	for _, score := range scores {
		remaining *= 1 - clamp01(score)
	}
	return clamp01(1 - remaining)
}

func normalizeVectorScore(metric string, raw float64) float64 {
	switch strings.ToUpper(metric) {
	case "L2":
		if raw < 0 {
			raw = 0
		}
		return 1 / (1 + raw)
	case "IP":
		return clamp01(1 / (1 + math.Exp(-raw)))
	default: // COSINE
		return clamp01((raw + 1) / 2)
	}
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func mergeSeeds(seeds []EntitySeed, topK int) []EntitySeed {
	byID := make(map[string]EntitySeed)
	for _, seed := range seeds {
		seed.Score = clamp01(seed.Score)
		if old, ok := byID[seed.EntityID]; !ok || seed.Score > old.Score {
			byID[seed.EntityID] = seed
		}
	}
	out := make([]EntitySeed, 0, len(byID))
	for _, seed := range byID {
		out = append(out, seed)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].EntityID < out[j].EntityID
		}
		return out[i].Score > out[j].Score
	})
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out
}
