package neo4j

import (
	"strings"
	"testing"
)

func TestKGV2SchemaQueries(t *testing.T) {
	joined := strings.Join(KGV2SchemaQueries(), "\n")
	for _, required := range []string{
		"KGEntity", "KGEntityMention", "KGChunk", "REQUIRE e.id IS UNIQUE",
		"REQUIRE m.id IS UNIQUE", "REQUIRE c.id IS UNIQUE", "REQUIRE c.pg_id IS UNIQUE",
		"normalized_name", "doc_hash", "pg_id",
	} {
		if !strings.Contains(joined, required) {
			t.Errorf("schema does not contain %q", required)
		}
	}
}
