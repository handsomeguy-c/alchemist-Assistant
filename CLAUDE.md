# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

alchemist-Assistant (炼丹师助手) is a multi-modal AI agent system in Go 1.24 with a 6-layer clean architecture and zero circular dependencies. It combines RAG (Milvus+ES+Neo4j hybrid retrieval), a three-layer memory system (short/long-term + preferences with Neo4j graph enhancement), sandboxed tool execution, and ReAct multi-step reasoning.

**LLM Backend**: DeepSeek API for chat/embedding via `internal/infrastructure/llm/`. Embedding models come from Volcengine/Doubao.

## Build & Run

```bash
# Local dev (needs Docker Desktop for infrastructure)
go mod tidy
docker compose up -d                    # Start PG, Milvus, ES, Kafka, Neo4j
go run ./cmd/server                     # Starts on :8090

# Docker build (linux/arm64)
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
  go build -ldflags="-s -w" -o alchemist-assistant ./cmd/server
docker compose up -d --build

# Run tests (currently only promptctx/ package has tests)
go test ./...
```

**Config**: Set `DEEPSEEK_API_KEY` in `config/config.yaml` or a local `.env` file. All infrastructure services (Milvus, ES, Kafka, Neo4j) are optional — the system degrades gracefully on connection failure.

## Architecture: 6-Layer Dependency Stack

```
cmd/server/main.go          (wires everything via application.Deps)
  config/                    (YAML config, 100+ flat fields)
  internal/interfaces/http/  (12 REST endpoints)
  internal/application/chat/ (UnifiedAgent orchestrator — the hub)
  internal/domain/           (pure business logic, no infra imports)
  internal/infrastructure/   (LLM, platform connectors, persistence, sandbox)
```

**Critical layout note**: The architecture docs in `docs/architecture/` reference an older package structure (agent.go at 1881 lines, runtime/, separate graph/). The current code under `internal/application/chat/` has already been split into ~20 files but the code organization may still be in flux. Refer to actual files, not docs, for current structure.

## Request Flow (6 Stages)

1. **Smart Routing** (`infra_router.go`): Keyword heuristics → chat / RAG / tool / ReAct mode
2. **Context Assembly** (`ctx_builder.go`, domain/promptctx/): Schema-driven system prompt built from pluggable `ContextSource` interfaces — profile, planner state, memory recall, tool state, sandbox constraints
3. **Execution**: Mode-dependent — direct LLM, single-tool+sandbox, hybrid search+generation, or ReAct with DAG graph scheduling
4. **Memory Update** (`mem_writer.go`): STM append, async LTM extraction, consolidation (dedup/merge/decay/expire every 5 items)
5. **Event Publish**: Kafka `agent.chat` event (gracefully degrades)
6. **Response**: JSON (sync) or SSE stream (async)

## Key Design Patterns

- **Constructor DI**: All components receive deps via constructors. Main wires everything in `application.Deps`.
- **Graceful degradation**: Every infra connection fails independently. System continues with reduced capability — check `Ready.Status` indicators per service.
- **Schema-driven prompts** (`domain/promptctx/`): System prompts assembled from declarative slot schemas with token budget management. Three pre-built schemas: Chat, Tool, ReAct.
- **Small-to-big retrieval** (`domain/rag/`): Parent/child chunking — small chunks for vector indexing, large parent chunks for LLM context.
- **Plugin interfaces**: `ContextSource` for prompt assembly, `sandbox.Executor` for execution backends (Docker/Local/Mock), `tool.Tool` for extensible tools.
- **Capacity-based context injection**: Memory recall and RAG results are injected with scores; the assembler filters by relevance thresholds rather than always injecting everything.

## RAG Pipeline

Input → `domain/rag/splitter.go` (parent/child chunking, window overlap) → `domain/rag/hybrid.go` (3-path parallel: Milvus semantic + ES BM25 + Neo4j graph, fused via RRF with configurable weights) → `domain/rag/reranker.go` (LLM listwise rerank) → chunk content loaded from PostgreSQL → LLM synthesis.

`domain/document/parser.go` handles PDF parsing with fallback chain: `pdfplumber` → `pdftotext` → Go PDF library.

## Memory System

- **Short-term** (`domain/memory/shortterm/`): Sliding window, not persisted (dies with process)
- **Long-term** (`domain/memory/longterm/`): Embedding+TF-IDF dual index. Cosine similarity with importance weighting. Dedup at 0.95, merge at 0.80. Decay 0.995/day, expire after 30 days if importance < 0.3. Persisted to PostgreSQL.
- **Preferences** (`domain/memory/preference/`): LLM NER extraction (async) + rule-based extraction (sync, immediate). Persisted, restored on startup.
- **Graph memory** (`domain/memory/graph/`): Neo4j layer on LTM with FOLLOWS, SIMILAR_TO, CAUSES, BELONGS_TO relationships. Protects high-centrality nodes during consolidation.

## API Endpoints (12 routes, unversioned)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/chat` | Sync chat |
| POST | `/api/chat/stream` | SSE streaming |
| POST | `/api/chat/cancel` | Cancel in-flight task |
| POST | `/api/upload` | Upload doc to RAG |
| POST | `/api/docs/delete` | Delete doc by hash |
| GET/POST | `/api/documents` | List/create docs |
| GET/POST | `/api/documents/{id}` | Get doc / ingest to RAG |
| GET | `/api/memory` | Memory state |
| GET | `/api/tools` | List tools |
| POST | `/api/tools/mcp` | Register MCP tool |
| GET | `/api/snapshots` | Task snapshots |
| GET | `/api/status` | System status |

## Sandbox Execution

`domain/sandbox/validator.go` (static security: whitelist, length limits) → executor (Docker/Local/Mock via `infrastructure/sandbox/`) → Kafka audit. Docker backend enforces CPU/memory/PID/network/read-only filesystem limits.

## Known Issues & Active Considerations

- The main orchestrator (1881-line `agent.go` in docs) has been split into `internal/application/chat/` (~20 files mixed concern). Verify actual boundaries before following docs.
- Test coverage is ~0.1% (only `promptctx/` has tests at `internal/domain/promptctx/`)
- Routes are not versioned (no `/api/v1/` — planned P1 refactor)
- Neo4j queries may not be parameterized (injection risk — planned P2)
- No structured logging or trace IDs (planned P3)
- Config is a flat 100+ field struct (planned P2 split)
- OCR for PDFs not yet implemented
