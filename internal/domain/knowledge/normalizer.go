package knowledge

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

type EntityNormalizer struct{}

func NewEntityNormalizer() EntityNormalizer { return EntityNormalizer{} }

func (EntityNormalizer) DisplayName(input string) string {
	value := norm.NFKC.String(input)
	value = strings.Join(strings.Fields(value), " ")
	return strings.TrimFunc(value, trimEdgePunctuation)
}

func (n EntityNormalizer) Normalize(input string) string {
	return strings.ToLower(n.DisplayName(input))
}

func trimEdgePunctuation(r rune) bool {
	if unicode.IsSpace(r) {
		return true
	}
	switch r {
	case '\'', '"', '“', '”', '‘', '’', '「', '」', '『', '』',
		'(', ')', '[', ']', '{', '}', '<', '>', '，', ',', '。',
		';', '；', ':', '：', '!', '！', '?', '？':
		return true
	}
	return false
}

func NormalizeEntityType(t EntityType) EntityType {
	switch t {
	case EntityPerson, EntityOrg, EntityLocation, EntityConcept, EntityEvent,
		EntityProduct, EntityTechnology, EntitySystem, EntityVersion, EntityMetric:
		return t
	default:
		return EntityUnknown
	}
}
