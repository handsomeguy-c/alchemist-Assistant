package knowledge

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"unicode/utf8"
)

// ─────────────────────────────── 配置 ─────────────────────────────────────

// ExtractorConfig 抽取器配置
type ExtractorConfig struct {
	// EntityTypes 允许的实体类型列表；空列表使用默认类型
	EntityTypes []string
	// RelationTypes 允许的关系类型列表；空列表使用默认关系
	RelationTypes []string
	// MaxGleaning 最大 gleaning 迭代次数（0 = 禁用）
	MaxGleaning int
	// MaxInputTokens 单次 LLM 输入的安全 token 上限（超限跳过 gleaning）
	MaxInputTokens int
	// Language 提示词语言（"zh" | "en"），默认 "zh"
	Language string
}

// DefaultExtractorConfig 返回合理的默认配置
func DefaultExtractorConfig() ExtractorConfig {
	return ExtractorConfig{
		EntityTypes:    defaultEntityTypes(),
		RelationTypes:  defaultRelationTypes(),
		MaxGleaning:    1,
		MaxInputTokens: 20480,
		Language:       "zh",
	}
}

func defaultEntityTypes() []string {
	return []string{"Person", "Organization", "Location", "Concept", "Event", "Product", "Unknown"}
}

func defaultRelationTypes() []string {
	return []string{"RELATES_TO", "PART_OF", "CAUSES", "DESCRIBES", "MENTIONS", "WORKS_FOR", "LOCATED_IN"}
}

// ─────────────────────────────── Extractor ──────────────────────────────────

// Extractor 通过注入的 LLM 回调从文本中抽取实体和关系。
// 支持 LightRAG 风格的结构化抽取 + Gleaning 迭代精炼 + 查询关键词提取。
type Extractor struct {
	llmFn      func(systemPrompt, userMsg string) string
	cfg        ExtractorConfig
	tokenEstFn func(text string) int // 可选的 token 估算函数；nil 时用字符数 / 2 估算
}

// NewExtractor 创建 Extractor，llmFn 为 LLM 调用回调（与 agent 解耦）。
// 使用默认配置（单轮 gleaning，中英文混合提示词）。
func NewExtractor(llmFn func(systemPrompt, userMsg string) string) *Extractor {
	return NewExtractorWithConfig(llmFn, DefaultExtractorConfig())
}

// NewExtractorWithConfig 创建带自定义配置的 Extractor
func NewExtractorWithConfig(llmFn func(systemPrompt, userMsg string) string, cfg ExtractorConfig) *Extractor {
	if len(cfg.EntityTypes) == 0 {
		cfg.EntityTypes = defaultEntityTypes()
	}
	if len(cfg.RelationTypes) == 0 {
		cfg.RelationTypes = defaultRelationTypes()
	}
	if cfg.MaxInputTokens <= 0 {
		cfg.MaxInputTokens = 20480
	}
	return &Extractor{llmFn: llmFn, cfg: cfg}
}

// ─────────────────────── 提示词模板（LightRAG 风格）─────────────────────────

// entityTypesDesc 构建实体类型说明
func (e *Extractor) entityTypesDesc() string {
	types := make([]string, len(e.cfg.EntityTypes))
	for i, t := range e.cfg.EntityTypes {
		types[i] = fmt.Sprintf("- %s", t)
	}
	return strings.Join(types, "\n")
}

// relationTypesDesc 构建关系类型说明
func (e *Extractor) relationTypesDesc() string {
	types := make([]string, len(e.cfg.RelationTypes))
	for i, t := range e.cfg.RelationTypes {
		types[i] = fmt.Sprintf("- %s", t)
	}
	return strings.Join(types, "\n")
}

// buildExtractionPrompt 构建 LightRAG 风格的结构化抽取提示词。
//
// 输出格式使用分隔符（与 LightRAG 论文一致）：
//
//	("entity"<|>实体名<|>类型<|>描述)
//	("relation"<|>源实体<|>目标实体<|>关系描述<|>关键词1,关键词2<|>强度)
//	("content_keywords"<|>主题词1, 主题词2, ...)
//
// 记录分隔符为 ##，完成标记为 <|COMPLETE|>
func (e *Extractor) buildExtractionPrompt() string {
	return fmt.Sprintf(`你是一个信息抽取专家。从给定文本中识别所有实体和它们之间的明确关系。

## 实体类型（type 字段只能从以下选择）：
%s

## 关系类型（rel_type 字段只能从以下选择）：
%s

## 步骤：
1. 识别文本中所有实体。对每个实体提取：
   - entity_name：实体名称（精确、无歧义）
   - entity_type：从上述实体类型中选择
   - entity_description：基于文本对该实体的简洁描述（不要编造文本中没有的信息）

2. 识别文本中所有明确存在关系的 (source_entity, target_entity) 对。对每对关系提取：
   - source_entity：源实体名称
   - target_entity：目标实体名称
   - relationship_description：简洁描述两者关系
   - relationship_keywords：3-8 个描述此关系的高层关键词/主题词（用逗号分隔）
   - relationship_strength：关系强度 1-10 的整数（10 最强）

3. 提取整个文本的 content_keywords：3-8 个总结文本主要主题的高层关键词（用逗号分隔）

## 输出格式（严格遵守，每条一行）：
("entity"<|>entity_name<|>entity_type<|>entity_description)
("relation"<|>source_entity<|>target_entity<|>relationship_description<|>keyword1,keyword2,keyword3<|>strength)
("content_keywords"<|>keyword1, keyword2, keyword3)

每条记录用 ## 分隔。所有记录输出完毕后输出 <|COMPLETE|>。

## 输出示例：
("entity"<|>张三<|>Person<|>张三是一名资深后端工程师，精通 Go 和分布式系统)
##
("entity"<|>阿里巴巴<|>Organization<|>阿里巴巴是中国的大型互联网科技公司)
##
("relation"<|>张三<|>阿里巴巴<|>张三曾在阿里巴巴担任高级工程师<|>职业经历, 工程师, 互联网公司, 就业<|>8)
##
("content_keywords"<|>后端工程师, Go语言, 阿里巴巴, 职业背景)
##
<|COMPLETE|>

如果文本中没有可抽取的实体，只输出：
<|COMPLETE|>

只输出上述格式，不要加任何额外说明。`, e.entityTypesDesc(), e.relationTypesDesc())
}

// buildGleaningPrompt 构建 gleaning（遗漏补全）提示词
func (e *Extractor) buildGleaningPrompt() string {
	return fmt.Sprintf(`在上一次抽取中，可能遗漏了一些实体和关系。请仔细重新阅读文本，找出之前遗漏的实体和关系。

## 实体类型：
%s

## 关系类型：
%s

## 格式（与之前相同）：
("entity"<|>entity_name<|>entity_type<|>entity_description)
("relation"<|>source_entity<|>target_entity<|>relationship_description<|>keyword1,keyword2<|>strength)

每条记录用 ## 分隔。如果没有遗漏，只输出 <|COMPLETE|>。`, e.entityTypesDesc(), e.relationTypesDesc())
}

// buildLoopCheckPrompt 构建 gleaning 循环检查提示词
func (e *Extractor) buildLoopCheckPrompt() string {
	return `在上一次遗漏补全中，我们找到了一些额外的实体和关系。

请回答：文本中是否仍然存在需要添加的遗漏实体或关系？

只回答 YES 或 NO，不要有其他内容。`
}

// buildQueryKeywordPrompt 构建查询关键词提取提示词（双层次：HL + LL）。
//
// 这是 LightRAG 检索的第一步：将用户查询分解为高层（抽象主题）和
// 低层（具体实体）关键词，分别用于匹配关系 keys 和实体 keys。
func (e *Extractor) buildQueryKeywordPrompt() string {
	return `你是一个查询分析专家。给定用户的查询，提取两组关键词：

1. high_level_keywords（高层关键词）：查询涉及哪些抽象概念、主题、领域或宏观话题？
   这些关键词用于匹配文本中关系的主题标签（如"人工智能"、"教育改革"、"气候政策"）。

2. low_level_keywords（低层关键词）：查询涉及哪些具体的实体名称、属性、专有名词？
   这些关键词用于匹配文本中的具体实体（如"GPT-4"、"北京大学"、"巴黎协定"）。

## 输出格式（严格遵守）：
{"high_level_keywords": ["关键词1", "关键词2", ...], "low_level_keywords": ["实体1", "实体2", ...]}

每组至少提供 1 个、最多 8 个关键词。只输出 JSON，不要有任何其他内容。`
}

// ─────────────────────── 核心抽取方法 ──────────────────────────────────────

// Extract 从单段文本中抽取实体和关系（单轮，无 gleaning，向后兼容）。
// 若 LLM 不可用或解析失败，返回空结果（不影响主流程）。
func (e *Extractor) Extract(text string) ExtractResult {
	record := e.ExtractWithGleaning(text)
	return ExtractResult{
		Entities:  record.Entities,
		Relations: record.Relations,
	}
}

// ExtractWithGleaning 执行 LightRAG 风格的带 gleaning 的实体关系抽取。
//
// 流程：
//  1. 首轮 LLM 抽取（结构化分隔符格式）
//  2. Gleaning 迭代（最多 cfg.MaxGleaning 轮）：提示 LLM 找出遗漏的实体/关系
//  3. 循环检查：LLM 判断 YES/NO 是否仍需继续
//  4. 去重 + 清洗
//
// Gleaning 在以下情况自动跳过：
//   - cfg.MaxGleaning == 0
//   - 预估输入 token 超过 cfg.MaxInputTokens
//   - 首轮抽取未返回任何实体
func (e *Extractor) ExtractWithGleaning(text string) ExtractionRecord {
	if e.llmFn == nil || strings.TrimSpace(text) == "" {
		return ExtractionRecord{}
	}

	// 首轮抽取
	record := e.extractSinglePass(text, e.buildExtractionPrompt())
	if len(record.Entities) == 0 || e.cfg.MaxGleaning <= 0 {
		return record
	}

	// 检查 token 预算：超限跳过 gleaning
	if e.estTokens(text) > e.cfg.MaxInputTokens {
		log.Printf("ℹ️  文本过长（est %d tokens），跳过 KG gleaning", e.estTokens(text))
		return record
	}

	// Gleaning 迭代
	for i := 0; i < e.cfg.MaxGleaning; i++ {
		// 遗漏补全
		gleanRecord := e.extractSinglePass(text, e.buildGleaningPrompt())
		if len(gleanRecord.Entities) == 0 {
			break // 没有新发现，终止
		}

		// 合并
		beforeEnt, beforeRel := len(record.Entities), len(record.Relations)
		record = e.mergeRecords(record, gleanRecord)
		addedEnt := len(record.Entities) - beforeEnt
		addedRel := len(record.Relations) - beforeRel
		log.Printf("🔍 KG gleaning 第 %d 轮：发现 +%d 实体, +%d 关系", i+1, addedEnt, addedRel)

		// 循环检查（最后一轮跳过，避免额外 LLM 调用）
		if i == e.cfg.MaxGleaning-1 {
			break
		}
		cont := e.checkContinue(text, record)
		if !cont {
			break
		}
	}

	return record
}

// extractSinglePass 单次 LLM 抽取（解析结构化分隔符格式）
func (e *Extractor) extractSinglePass(text string, systemPrompt string) ExtractionRecord {
	raw := e.llmFn(systemPrompt, "文本：\n"+text)
	return e.parseStructured(raw)
}

// checkContinue 询问 LLM 是否还有遗漏（YES/NO）
func (e *Extractor) checkContinue(text string, _ ExtractionRecord) bool {
	raw := e.llmFn(e.buildLoopCheckPrompt(), "原始文本：\n"+text)
	raw = strings.TrimSpace(strings.ToUpper(raw))
	return strings.HasPrefix(raw, "YES") || strings.Contains(raw, "YES")
}

// ─────────────────────── 结构化格式解析 ────────────────────────────────────

const (
	tupleDelimiter      = "<|>"
	recordDelimiter     = "##"
	completionDelimiter = "<|COMPLETE|>"
)

// parseStructured 解析 LightRAG 风格的分隔符格式输出
func (e *Extractor) parseStructured(raw string) ExtractionRecord {
	raw = strings.TrimSpace(raw)
	// 去掉可能的 markdown 代码块包裹
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	// 移除完成标记
	raw = strings.ReplaceAll(raw, completionDelimiter, "")

	// 按记录分隔符拆分
	records := strings.Split(raw, recordDelimiter)

	var record ExtractionRecord
	seenEntities := make(map[string]bool)

	for _, rec := range records {
		rec = strings.TrimSpace(rec)
		if rec == "" {
			continue
		}

		// 按 <|> 拆分
		parts := strings.Split(rec, tupleDelimiter)
		if len(parts) < 2 {
			continue
		}

		// 去掉开头的括号标记
		typeTag := strings.TrimPrefix(strings.TrimSpace(parts[0]), "(\"")

		switch typeTag {
		case "entity":
			if len(parts) < 4 {
				continue
			}
			name := strings.TrimSpace(parts[1])
			entType := strings.TrimSpace(parts[2])
			// 去掉末尾的 )
			desc := strings.TrimSuffix(strings.TrimSpace(strings.Join(parts[3:], tupleDelimiter)), ")")
			desc = strings.Trim(desc, "\"")

			if name == "" || seenEntities[name] {
				continue
			}
			if !isValidEntityType(EntityType(entType)) {
				entType = string(EntityUnknown)
			}

			record.Entities = append(record.Entities, Entity{
				Name: name,
				Type: EntityType(entType),
			})
			seenEntities[name] = true

		case "relation":
			if len(parts) < 6 {
				continue
			}
			src := strings.TrimSpace(parts[1])
			tgt := strings.TrimSpace(parts[2])
			relDesc := strings.TrimSpace(parts[3])
			keywords := strings.TrimSpace(parts[4])
			// 去掉末尾的 )
			strengthStr := strings.TrimSuffix(strings.TrimSpace(strings.Join(parts[5:], tupleDelimiter)), ")")
			strengthStr = strings.Trim(strengthStr, "\"")
			strength := parseStrength(strengthStr)

			if src == "" || tgt == "" {
				continue
			}

			// 关系类型从 keywords 推导（取第一个匹配的已知类型或 RELATES_TO）
			relType := e.inferRelType(keywords, relDesc)

			record.Relations = append(record.Relations, Relation{
				FromName: src,
				ToName:   tgt,
				RelType:  relType,
				Weight:   strength,
			})

		case "content_keywords":
			// content_keywords 出现在 ( 之后
			kwRaw := strings.TrimSuffix(strings.TrimSpace(strings.Join(parts[1:], tupleDelimiter)), ")")
			kwRaw = strings.Trim(kwRaw, "\"")
			for _, kw := range strings.Split(kwRaw, ",") {
				kw = strings.TrimSpace(kw)
				if kw != "" {
					record.ContentKeywords = append(record.ContentKeywords, kw)
				}
			}
		}
	}

	// 如果分隔符格式解析失败（0 条 entity），回退到 JSON 格式
	if len(record.Entities) == 0 && len(record.Relations) == 0 {
		return e.parseJSONFallback(raw)
	}

	return record
}

// parseJSONFallback 回退到旧 JSON 格式解析（向后兼容旧版 prompt 输出）
func (e *Extractor) parseJSONFallback(raw string) ExtractionRecord {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	type jsonExtract struct {
		Entities  []Entity   `json:"entities"`
		Relations []Relation `json:"relations"`
	}
	var result jsonExtract
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return ExtractionRecord{}
	}

	seen := make(map[string]bool)
	var record ExtractionRecord
	for _, ent := range result.Entities {
		ent.Name = strings.TrimSpace(ent.Name)
		if ent.Name == "" || seen[ent.Name] {
			continue
		}
		if !isValidEntityType(ent.Type) {
			ent.Type = EntityUnknown
		}
		record.Entities = append(record.Entities, ent)
		seen[ent.Name] = true
	}
	for _, rel := range result.Relations {
		rel.FromName = strings.TrimSpace(rel.FromName)
		rel.ToName = strings.TrimSpace(rel.ToName)
		if rel.FromName == "" || rel.ToName == "" || rel.RelType == "" {
			continue
		}
		if !isValidRelType(rel.RelType) {
			rel.RelType = "RELATES_TO"
		}
		record.Relations = append(record.Relations, rel)
	}
	return record
}

// mergeRecords 合并两个抽取记录（去重）
func (e *Extractor) mergeRecords(a, b ExtractionRecord) ExtractionRecord {
	seenEnt := make(map[string]bool)
	for _, ent := range a.Entities {
		seenEnt[ent.Name] = true
	}
	for _, ent := range b.Entities {
		if !seenEnt[ent.Name] {
			a.Entities = append(a.Entities, ent)
			seenEnt[ent.Name] = true
		}
	}

	// 关系去重：按 (src, tgt) 对判断
	type relKey struct{ src, tgt string }
	seenRel := make(map[relKey]bool)
	for _, rel := range a.Relations {
		seenRel[relKey{rel.FromName, rel.ToName}] = true
	}
	for _, rel := range b.Relations {
		if !seenRel[relKey{rel.FromName, rel.ToName}] {
			a.Relations = append(a.Relations, rel)
			seenRel[relKey{rel.FromName, rel.ToName}] = true
		}
	}

	// 合并 content keywords
	seenKW := make(map[string]bool)
	for _, kw := range a.ContentKeywords {
		seenKW[strings.ToLower(kw)] = true
	}
	for _, kw := range b.ContentKeywords {
		if !seenKW[strings.ToLower(kw)] {
			a.ContentKeywords = append(a.ContentKeywords, kw)
			seenKW[strings.ToLower(kw)] = true
		}
	}

	return a
}

// ─────────────────────── 查询关键词提取（双层次：HL + LL）───────────────────

// ExtractQueryKeywords 从用户查询中提取双层次关键词（LightRAG 检索 Step 1）。
//
// 返回：
//   - HighLevel：抽象主题关键词 → 用于匹配关系 keys
//   - LowLevel：具体实体关键词 → 用于匹配实体 keys
//
// 若 LLM 不可用，使用规则兜底（简单分词）。
func (e *Extractor) ExtractQueryKeywords(query string) QueryKeywords {
	if e.llmFn == nil || strings.TrimSpace(query) == "" {
		return e.extractKeywordsRuleBased(query)
	}

	raw := e.llmFn(e.buildQueryKeywordPrompt(), "查询："+query)
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	type kwResult struct {
		HighLevel []string `json:"high_level_keywords"`
		LowLevel  []string `json:"low_level_keywords"`
	}
	var result kwResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		log.Printf("⚠️  查询关键词提取解析失败: %v（原始: %.100s），使用规则兜底", err, raw)
		return e.extractKeywordsRuleBased(query)
	}

	// 确保非空
	if len(result.HighLevel) == 0 && len(result.LowLevel) == 0 {
		return e.extractKeywordsRuleBased(query)
	}

	return QueryKeywords{
		HighLevel: result.HighLevel,
		LowLevel:  result.LowLevel,
	}
}

// extractKeywordsRuleBased 规则兜底：简单提取查询中的关键词
func (e *Extractor) extractKeywordsRuleBased(query string) QueryKeywords {
	// 简单策略：把查询按空格/标点分词，较长的词作为 low_level，
	// 整个查询作为 high_level 兜底
	words := tokenize(query)
	var lowLevel []string
	for _, w := range words {
		if utf8.RuneCountInString(w) >= 2 {
			lowLevel = append(lowLevel, w)
		}
	}
	return QueryKeywords{
		HighLevel: []string{query}, // 整个查询作为高层主题
		LowLevel:  lowLevel,
	}
}

// ─────────────────────── 辅助方法 ──────────────────────────────────────────

// inferRelType 从 relation keywords 和 description 推断关系类型
func (e *Extractor) inferRelType(keywords, desc string) string {
	combined := strings.ToLower(keywords + " " + desc)
	// 按优先级匹配已知关系类型
	typePriority := []struct {
		keyword string
		relType string
	}{
		{"工作", "WORKS_FOR"},
		{"位于", "LOCATED_IN"},
		{"属于", "PART_OF"},
		{"导致", "CAUSES"},
		{"引起", "CAUSES"},
		{"描述", "DESCRIBES"},
		{"介绍", "DESCRIBES"},
		{"提及", "MENTIONS"},
		{"相关", "RELATES_TO"},
	}
	for _, p := range typePriority {
		if strings.Contains(combined, p.keyword) {
			return p.relType
		}
	}
	return "RELATES_TO"
}

// estTokens 粗略估算 token 数（中文按字符数，英文按空格分词）
func (e *Extractor) estTokens(text string) int {
	if e.tokenEstFn != nil {
		return e.tokenEstFn(text)
	}
	// 粗略估算：中文每字符 ~1 token，英文每 4 字符 ~1 token
	cn := 0
	en := 0
	for _, r := range text {
		if r >= 0x4e00 && r <= 0x9fff {
			cn++
		} else if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			en++
		}
	}
	return cn + en/4 + len(strings.Fields(text))
}

func parseStrength(s string) float64 {
	s = strings.TrimSpace(s)
	// 尝试解析整数
	var v int
	if _, err := fmt.Sscanf(s, "%d", &v); err == nil {
		return float64(v) / 10.0 // 归一化到 0-1
	}
	// 尝试解析浮点
	var f float64
	if _, err := fmt.Sscanf(s, "%f", &f); err == nil {
		if f > 1.0 {
			return f / 10.0
		}
		return f
	}
	return 0.5
}

// tokenize 简单分词（按常见分隔符）
func tokenize(text string) []string {
	// 按空格和常见标点切分
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == ',' || r == '.' || r == '!' || r == '?' ||
			r == '，' || r == '。' || r == '！' || r == '？' ||
			r == '；' || r == '：' || r == '、' || r == '\n' || r == '\t'
	})
	// 去重
	seen := make(map[string]bool)
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" && !seen[p] {
			result = append(result, p)
			seen[p] = true
		}
	}
	return result
}

// ─────────────────────── 类型校验（导出供 kgstore 使用）─────────────────────

func isValidEntityType(t EntityType) bool {
	switch t {
	case EntityPerson, EntityOrg, EntityLocation, EntityConcept,
		EntityEvent, EntityProduct, EntityUnknown:
		return true
	}
	return false
}

func isValidRelType(r string) bool {
	switch r {
	case "RELATES_TO", "PART_OF", "CAUSES", "DESCRIBES",
		"MENTIONS", "WORKS_FOR", "LOCATED_IN":
		return true
	}
	return false
}
