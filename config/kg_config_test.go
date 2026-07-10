package config

import "testing"

func TestApplyDefaultsKGV2(t *testing.T) {
	c := &APIConfig{}
	applyDefaults(c)
	if c.KGEntitySeedTopK != 5 || c.KGEntityVectorTopK != 10 || c.KGGraphMaxHops != 2 || c.KGGraphEvidenceTopK != 20 {
		t.Fatalf("unexpected KG integer defaults: %#v", c.Neo4jConfig)
	}
	if c.KGEntityVectorMinScore != 0.65 || c.KGGraphHopDecay != 0.7 || c.KGEntityVectorMetric != "COSINE" {
		t.Fatalf("unexpected KG defaults: %#v", c.Neo4jConfig)
	}
	if c.KGRelationTypeWeights[RelationWeightMentions] >= c.KGRelationTypeWeights[RelationWeightCauses] {
		t.Fatalf("generic mention edges should be lower weighted: %#v", c.KGRelationTypeWeights)
	}
}
