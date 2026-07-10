package knowledge

import "testing"

func TestEntityNormalizerNormalize(t *testing.T) {
	tests := map[string]string{
		"Kafka":          "kafka",
		" kafka ":        "kafka",
		"KAFKA":          "kafka",
		"Ｋａｆｋａ":          "kafka",
		"Kafka\t 消息  队列": "kafka 消息 队列",
		"“Kafka”":        "kafka",
		"C++":            "c++",
		"C#":             "c#",
		".NET":           ".net",
		"PostgreSQL 16":  "postgresql 16",
		"v2.0":           "v2.0",
	}
	for input, want := range tests {
		if got := NewEntityNormalizer().Normalize(input); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestStableIDs(t *testing.T) {
	n := NewEntityNormalizer()
	entityA := CanonicalEntityID(EntityProduct, n.Normalize("Kafka"))
	entityB := CanonicalEntityID(EntityProduct, n.Normalize(" KAFKA "))
	if entityA == "" || entityA != entityB {
		t.Fatalf("entity IDs are not stable: %q != %q", entityA, entityB)
	}
	if entityA == CanonicalEntityID(EntityTechnology, n.Normalize("Kafka")) {
		t.Fatal("different entity types must produce different IDs")
	}

	mentionA := MentionID("doc", 3, 42, entityA, "kafka")
	mentionB := MentionID("doc", 3, 42, entityA, "kafka")
	if mentionA == "" || mentionA != mentionB {
		t.Fatal("mention ID must be stable")
	}

	relA := RelationID(entityA, "other", "USES", "doc", 3, 42)
	relB := RelationID(entityA, "other", "USES", "doc", 3, 42)
	if relA == "" || relA != relB {
		t.Fatal("relation ID must be stable")
	}
	if relA == RelationID(entityA, "other", "USES", "doc", 4, 43) {
		t.Fatal("different evidence must produce different relation IDs")
	}
}
