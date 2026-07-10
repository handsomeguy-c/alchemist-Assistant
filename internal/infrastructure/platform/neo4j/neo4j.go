// Package neo4j 提供 Neo4j 驱动连接的薄封装与启动期约束/索引初始化。
// 业务级图操作（实体/关系 / 记忆图节点边）由 graph 与 memory domain 层负责。
package neo4j

import (
	"context"
	"log"
	"time"

	"alchemist-Assistant/config"

	driver "github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Client 持有 Neo4j 驱动连接
type Client struct {
	driver    driver.DriverWithContext
	available bool
}

// Connect 创建连接；连接失败时返回不可用实例，不阻塞启动。
// 由调用方在启动后调用 EnsureConstraints 应用基础约束。
func Connect(cfg config.Neo4jConfig) *Client {
	if !cfg.KGEnabled || cfg.Neo4jURI == "" {
		log.Printf("ℹ️  Neo4j 未启用（KGEnabled=%v, URI=%q）", cfg.KGEnabled, cfg.Neo4jURI)
		return &Client{available: false}
	}

	d, err := driver.NewDriverWithContext(
		cfg.Neo4jURI,
		driver.BasicAuth(cfg.Neo4jUser, cfg.Neo4jPassword, ""),
	)
	if err != nil {
		log.Printf("⚠️  Neo4j 驱动初始化失败: %v（知识图谱将降级跳过）", err)
		return &Client{available: false}
	}

	// 连通性验证（超时 5s）
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := d.VerifyConnectivity(ctx); err != nil {
		log.Printf("⚠️  Neo4j 连通性验证失败: %v（知识图谱将降级跳过）", err)
		_ = d.Close(context.Background())
		return &Client{available: false}
	}

	c := &Client{driver: d, available: true}
	c.EnsureConstraints()
	log.Printf("✅ Neo4j 已连接: %s", cfg.Neo4jURI)
	return c
}

// Available 报告 Neo4j 是否可用
func (c *Client) Available() bool { return c.available }

// Close 关闭驱动
func (c *Client) Close() {
	if c.driver != nil {
		c.driver.Close(context.Background())
	}
}

// Session 返回一个写入 session；调用方需 defer session.Close
func (c *Client) Session() driver.SessionWithContext {
	return c.driver.NewSession(context.Background(), driver.SessionConfig{
		AccessMode: driver.AccessModeWrite,
	})
}

// ReadSession returns a read-routed session for KG queries.
func (c *Client) ReadSession() driver.SessionWithContext {
	return c.driver.NewSession(context.Background(), driver.SessionConfig{AccessMode: driver.AccessModeRead})
}

// KGV2SchemaQueries is kept separate so schema changes are reviewable and testable.
func KGV2SchemaQueries() []string {
	return []string{
		`CREATE CONSTRAINT kg_entity_id IF NOT EXISTS FOR (e:KGEntity) REQUIRE e.id IS UNIQUE`,
		`CREATE CONSTRAINT kg_mention_id IF NOT EXISTS FOR (m:KGEntityMention) REQUIRE m.id IS UNIQUE`,
		`CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (c:KGChunk) REQUIRE c.id IS UNIQUE`,
		`CREATE CONSTRAINT kg_chunk_pg_id IF NOT EXISTS FOR (c:KGChunk) REQUIRE c.pg_id IS UNIQUE`,
		`CREATE INDEX kg_entity_normalized_name IF NOT EXISTS FOR (e:KGEntity) ON (e.normalized_name)`,
		`CREATE INDEX kg_entity_type IF NOT EXISTS FOR (e:KGEntity) ON (e.type)`,
		`CREATE INDEX kg_chunk_doc_hash IF NOT EXISTS FOR (c:KGChunk) ON (c.doc_hash)`,
		`CREATE INDEX kg_mention_doc_hash IF NOT EXISTS FOR (m:KGEntityMention) ON (m.doc_hash)`,
		`CREATE INDEX kg_mention_pg_id IF NOT EXISTS FOR (m:KGEntityMention) ON (m.pg_id)`,
	}
}

// EnsureConstraints 确保 Neo4j 中存在唯一约束/索引（幂等）
func (c *Client) EnsureConstraints() {
	if !c.available {
		return
	}
	ctx := context.Background()
	sess := c.Session()
	defer sess.Close(ctx)

	queries := []string{
		`CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE`,
		`CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)`,
		`CREATE INDEX memory_node_id IF NOT EXISTS FOR (m:Memory) ON (m.mem_id)`,
	}
	queries = append(queries, KGV2SchemaQueries()...)
	for _, q := range queries {
		if _, err := sess.Run(ctx, q, nil); err != nil {
			// 约束已存在或版本不支持时忽略
			log.Printf("ℹ️  Neo4j constraint/index: %v", err)
		}
	}
}
