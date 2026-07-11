# Alchemist Assistant RAG 知识库评测

使用 [RAGAS](https://github.com/explodinggradients/ragas) 框架对知识库检索增强生成质量进行系统评测。

## 评测指标

| 指标 | 说明 | 优秀阈值 | 考察维度 |
|---|---|---|---|
| **faithfulness** | 回答是否基于检索内容（反幻觉） | > 0.90 | 生成质量 |
| **answer_relevancy** | 回答是否切题、完整 | > 0.85 | 生成质量 |
| **context_precision** | 检索到的 chunk 中有多少真正有用 | > 0.75 | 检索质量 |
| **context_recall** | 是否检索到了所有必要信息 | > 0.80 | 检索质量 |

## 快速开始

### 1. 启动服务

```bash
cd /path/to/alchemist-Assistant
go run ./cmd/server
```

### 2. 安装 Python 依赖

```bash
pip install -r eval/requirements.txt
```

### 3. 配置 RAGAS 评判 LLM

RAGAS 使用 LLM-as-Judge，需要一个 LLM 来评估 RAG 输出。使用 DeepSeek：

```bash
export DEEPSEEK_API_KEY="your-key"
# RAGAS 自动使用 OpenAI 兼容接口连接到 DeepSeek
```

或使用 OpenAI：

```bash
export OPENAI_API_KEY="your-key"
```

### 4. 运行评测

```bash
# 完整评测（上传文档 + 查询 + 指标计算）
./eval/run_eval.sh

# 快速评测（文档已在知识库中）
./eval/run_eval.sh --quick

# 手动运行
python3 eval/evaluate_rag.py --base-url http://localhost:8090
```

## 评测数据集

`eval_dataset.yaml` 包含：
- **2 篇测试文档**：人工智能基础（~800字）、Go 语言高级特性（~600字）
- **10 条测试问题**：覆盖事实查询、概念解释、列表枚举、多跳推理等类型
- 每条问题配有标准答案（ground_truth）和期望检索上下文

## 评测流程

```
1. 上传测试文档 → 2. 逐条查询 → 3. 收集 answers + contexts
       ↓
4. RAGAS 评估 → 5. 生成报告 → 6. 清理文档
```

## 输出示例

```
==========================================
  Alchemist Assistant RAG 知识库评测报告
==========================================

              RAGAS Metrics
┌──────────────┬────────┬────────┬────────┐
│ 指标          │ 均值   │ 中位数  │ 解读   │
├──────────────┼────────┼────────┼────────┤
│ faithfulness │ 0.9234 │ 0.9500 │ ✅     │
│ relevancy    │ 0.8912 │ 0.9100 │ ✅     │
│ precision    │ 0.8234 │ 0.8500 │ ⚠️     │
│ recall       │ 0.7856 │ 0.8000 │ ⚠️     │
└──────────────┴────────┴────────┴────────┘

综合评分: 0.8559
```

## 自定义评测

编辑 `eval_dataset.yaml` 添加自定义文档和问题：

```yaml
test_documents:
  - id: my_doc
    title: "我的文档"
    content: "文档内容..."

questions:
  - id: q1
    question: "你的问题？"
    ground_truth: "标准答案"
    expected_contexts:
      - "期望检索到的上下文"
```

## 持续评测

可将评测集成到 CI/CD：

```bash
# 启动服务
go run ./cmd/server &
SERVER_PID=$!
sleep 5

# 运行评测
python3 eval/evaluate_rag.py --output results.json

# 检查阈值
python3 -c "
import json
with open('eval/results.json') as f:
    d = json.load(f)
metrics = {m['metric']: m['mean'] for m in d['ragas_metrics']}
assert metrics.get('faithfulness', 0) > 0.80, 'Faithfulness too low'
assert metrics.get('context_recall', 0) > 0.70, 'Context Recall too low'
print('✅ All thresholds passed')
"

kill $SERVER_PID
```
