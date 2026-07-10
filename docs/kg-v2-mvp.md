# KG V2 MVP

KG V2 fixes the evidence-overwrite problem in the original document knowledge graph while keeping PostgreSQL as the only source of truth for full Chunk content.

## Data model

```text
(KGChunk)-[:HAS_MENTION]->(KGEntityMention)-[:OF_ENTITY]->(KGEntity)
                                                        |
                                  (KGEntity)-[:KG_RELATION]->(KGEntity)
```

- `KGEntity.id` is SHA-256 of `type|normalized_name`. `canonical_name` is fixed on first creation; later surfaces are appended to `aliases` and `alias_norms`.
- `KGEntityMention.id` includes `doc_hash`, `chunk_id`, `pg_id`, canonical entity ID and normalized surface. The MVP stores one mention for each normalized entity surface in each Chunk; it does not store character offsets.
- `KGChunk.id` is `pg:<pg_id>`. It stores only `pg_id`, `doc_hash`, `chunk_id` and timestamps, never full text.
- Every `KG_RELATION` has an evidence ID containing both entity IDs, `rel_type`, `doc_hash`, `chunk_id` and `pg_id`. The actual Neo4j relationship type is always `KG_RELATION`; the semantic type is a property.

Legacy nodes use `:Entity` and dynamic relationship types. KG V2 uses only labels beginning with `KG`, and new search queries never read legacy nodes. The `:Memory` graph is unchanged.

## Ingestion and retrieval

For every PostgreSQL-backed Chunk, the extractor returns strict JSON. The application normalizes entity surfaces, resolves exact canonical and unique alias matches, and writes the complete Chunk graph in one Neo4j transaction. Duplicate ingestion produces the same IDs and therefore does not create duplicate nodes or relationships.

The `kg_entities` Milvus collection uses `entity_id` as its VarChar primary key. Its embedding text is:

```text
名称: {canonical_name}
类型: {type}
别名: {aliases}
```

If a collection named `kg_entities` exists with the old LightRAG `pg_id` schema or a different vector dimension, KG V2 refuses to overwrite it and falls back to canonical/alias search. Remove or rename that old collection deliberately before enabling KG V2 vectors.

Query retrieval combines canonical-name, alias and entity-vector seeds, keeps at most five seeds, and expands only one or two `KG_RELATION` hops. Relation evidence `pg_id` is preferred, followed by seed Mention evidence and neighbor Mention evidence. Results return to the existing RRF layer and are finally loaded from PostgreSQL.

## Rebuilding existing documents

Old `:Entity` rows cannot be migrated safely because their earlier `pg_id` values were overwritten. Rebuild KG V2 by re-uploading or re-running the normal ingestion operation for every source document already stored in PostgreSQL. New labels can coexist with old labels during the rebuild.

Useful checks:

```cypher
MATCH (e:KGEntity) RETURN count(e);
MATCH (m:KGEntityMention) RETURN count(m);
MATCH (c:KGChunk) RETURN count(c);
MATCH ()-[r:KG_RELATION]->() RETURN count(r), count(DISTINCT r.pg_id);
```

Inspect one evidence chain:

```cypher
MATCH (c:KGChunk)-[:HAS_MENTION]->(m:KGEntityMention)-[:OF_ENTITY]->(e:KGEntity)
RETURN c.pg_id, c.doc_hash, m.surface, e.canonical_name, e.type
LIMIT 50;
```

## Rollback and cleanup

Rolling application code back to `main@6ec5ee3` resumes the legacy `:Entity` query path because legacy nodes are not removed. To remove only KG V2 data after rollback:

```cypher
MATCH ()-[r:KG_RELATION]->() DELETE r;
MATCH (n) WHERE n:KGEntityMention OR n:KGChunk OR n:KGEntity DETACH DELETE n;
```

Drop `kg_entities` from Milvus separately. Do not delete `:Entity` or `:Memory` as part of KG V2 cleanup.

## Configuration

KG V2 settings live under `neo4j` in both YAML files:

- `entity_seed_top_k: 5`
- `entity_vector_top_k: 10`
- `entity_vector_min_score: 0.65`
- `entity_vector_metric: COSINE` (`COSINE`, `IP`, and `L2` are supported)
- `graph_max_hops: 2`
- `graph_hop_decay: 0.7`
- `graph_evidence_top_k: 20`
- `relation_type_weights`: bounded weights for the relationship whitelist

## Tests

```bash
go test ./...
go vet ./...
```

With local Neo4j and Milvus running:

```bash
docker compose up -d postgres etcd minio milvus neo4j elasticsearch
KG_INTEGRATION=1 go test -tags=integration ./internal/domain/knowledge -run KGV2 -v
```

Integration tests use unique entity names and document hashes and delete only the data they create.

## Known limitations

- Same name, same type and different meaning can still be merged.
- Alias resolution is rule-based; ambiguous aliases are deliberately not auto-merged.
- Entity vectors contain only name, type and aliases, without Chunk context or LLM summaries.
- There are no Fact/Event relation-instance nodes, Mention offsets, PPR, communities, Topic/Summary entry points, Text-to-Cypher, complete tenant isolation, or document-version governance.

