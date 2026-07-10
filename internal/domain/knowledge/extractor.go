package knowledge

import (
	"encoding/json"
	"log"
	"strings"
)

type Extractor struct {
	llmFn      func(systemPrompt, userMsg string) string
	normalizer EntityNormalizer
}

func NewExtractor(llmFn func(systemPrompt, userMsg string) string) *Extractor {
	return &Extractor{llmFn: llmFn, normalizer: NewEntityNormalizer()}
}

const extractSystemPrompt = `你是信息抽取专家。只返回严格 JSON，不要输出解释或 Markdown。
实体 type 只能是 Person, Organization, Location, Concept, Event, Product, Technology, System, Version, Metric, Unknown。
关系 rel_type 只能是 RELATES_TO, IS_A, PART_OF, USES, DEPENDS_ON, CAUSES, REPLACES, PRODUCES, DESCRIBES, MENTIONS, WORKS_FOR, LOCATED_IN。
每个关系的 from 和 to 必须出现在 entities 中。confidence 可选，范围 0 到 1。
格式：{"entities":[{"name":"实体名","type":"System","aliases":["别名"],"confidence":0.9}],"relations":[{"from":"实体A","to":"实体B","rel_type":"USES","confidence":0.8}]}
没有结果时返回 {"entities":[],"relations":[]}`

func (e *Extractor) Extract(text string) ExtractResult {
	if e.llmFn == nil || strings.TrimSpace(text) == "" {
		return ExtractResult{}
	}
	raw := strings.TrimSpace(e.llmFn(extractSystemPrompt, "文本：\n"+text))
	raw = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(raw, "```json"), "```"), "```"))
	var result ExtractResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		log.Printf("⚠️  实体关系抽取解析失败: %v（原始输出: %.100s）", err, raw)
		return ExtractResult{}
	}

	cleaned := ExtractResult{}
	bySurface := make(map[string]string)
	seenEntities := make(map[string]bool)
	for _, ent := range result.Entities {
		ent.Name = e.normalizer.DisplayName(ent.Name)
		if ent.Name == "" {
			continue
		}
		ent.Type = NormalizeEntityType(ent.Type)
		key := string(ent.Type) + "|" + e.normalizer.Normalize(ent.Name)
		if seenEntities[key] {
			continue
		}
		ent.Aliases = cleanAliases(e.normalizer, ent.Name, ent.Aliases)
		ent.Confidence = clampConfidence(ent.Confidence)
		cleaned.Entities = append(cleaned.Entities, ent)
		seenEntities[key] = true
		bySurface[e.normalizer.Normalize(ent.Name)] = ent.Name
		for _, alias := range ent.Aliases {
			bySurface[e.normalizer.Normalize(alias)] = ent.Name
		}
	}

	seenRelations := make(map[string]bool)
	for _, rel := range result.Relations {
		from, fromOK := bySurface[e.normalizer.Normalize(rel.FromName)]
		to, toOK := bySurface[e.normalizer.Normalize(rel.ToName)]
		if !fromOK || !toOK {
			continue
		}
		rel.FromName, rel.ToName = from, to
		rel.RelType = NormalizeRelationType(rel.RelType)
		rel.Confidence = clampConfidence(rel.Confidence)
		key := from + "|" + to + "|" + rel.RelType
		if seenRelations[key] {
			continue
		}
		seenRelations[key] = true
		cleaned.Relations = append(cleaned.Relations, rel)
	}
	return cleaned
}

func cleanAliases(n EntityNormalizer, canonical string, aliases []string) []string {
	canonicalNorm := n.Normalize(canonical)
	seen := map[string]bool{canonicalNorm: true}
	out := make([]string, 0, len(aliases))
	for _, alias := range aliases {
		alias = n.DisplayName(alias)
		normAlias := n.Normalize(alias)
		if normAlias == "" || seen[normAlias] {
			continue
		}
		seen[normAlias] = true
		out = append(out, alias)
	}
	return out
}

func clampConfidence(v float64) float64 {
	if v <= 0 {
		return 1
	}
	if v > 1 {
		return 1
	}
	return v
}

func NormalizeRelationType(value string) string {
	value = strings.ToUpper(strings.TrimSpace(value))
	value = strings.NewReplacer("-", "_", " ", "_").Replace(value)
	switch value {
	case RelationRelatesTo, RelationIsA, RelationPartOf, RelationUses,
		RelationDependsOn, RelationCauses, RelationReplaces, RelationProduces,
		RelationDescribes, RelationMentions, RelationWorksFor, RelationLocatedIn:
		return value
	case "BELONGS_TO", "COMPONENT_OF":
		return RelationPartOf
	case "UTILIZES", "USE":
		return RelationUses
	case "REQUIRES", "RELY_ON", "RELIES_ON":
		return RelationDependsOn
	case "LEADS_TO", "RESULTS_IN":
		return RelationCauses
	case "EMPLOYED_BY":
		return RelationWorksFor
	default:
		return RelationRelatesTo
	}
}

func isValidEntityType(t EntityType) bool { return NormalizeEntityType(t) == t }
func isValidRelType(r string) bool        { return NormalizeRelationType(r) == r }
