package knowledge

import "testing"

func TestExtractorCleansTypesAliasesAndRelations(t *testing.T) {
	raw := `{
  "entities": [
    {"name":" Apache Kafka ","type":"Technology","aliases":["Kafka","Ｋａｆｋａ"],"confidence":0.9},
    {"name":"系统A","type":"System"}
  ],
  "relations": [
    {"from":"系统A","to":"Apache Kafka","rel_type":"utilizes","confidence":0.8},
    {"from":"系统A","to":"不存在","rel_type":"CAUSES"}
  ]
}`
	e := NewExtractor(func(_, _ string) string { return "```json\n" + raw + "\n```" })
	got := e.Extract("text")
	if len(got.Entities) != 2 {
		t.Fatalf("entities=%d, want 2", len(got.Entities))
	}
	if got.Entities[0].Type != EntityTechnology || len(got.Entities[0].Aliases) != 1 || got.Entities[0].Aliases[0] != "Kafka" {
		t.Fatalf("unexpected entity: %#v", got.Entities[0])
	}
	if len(got.Relations) != 1 || got.Relations[0].RelType != RelationUses {
		t.Fatalf("unexpected relations: %#v", got.Relations)
	}
}

func TestExtractorFallsBackOnInvalidJSON(t *testing.T) {
	e := NewExtractor(func(_, _ string) string { return "not json" })
	got := e.Extract("text")
	if len(got.Entities) != 0 || len(got.Relations) != 0 {
		t.Fatalf("invalid JSON must degrade to empty result: %#v", got)
	}
}

func TestNormalizeRelationType(t *testing.T) {
	tests := map[string]string{
		"BELONGS_TO": RelationPartOf,
		"requires":   RelationDependsOn,
		"LEADS-TO":   RelationCauses,
		"unknown":    RelationRelatesTo,
	}
	for input, want := range tests {
		if got := NormalizeRelationType(input); got != want {
			t.Errorf("NormalizeRelationType(%q)=%q, want %q", input, got, want)
		}
	}
}
