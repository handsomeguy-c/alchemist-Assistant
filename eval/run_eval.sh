#!/bin/bash
# Alchemist Assistant RAG 知识库评测脚本
#
# 用法:
#   ./run_eval.sh              # 完整评测（上传文档 + 查询 + RAGAS 指标）
#   ./run_eval.sh --quick      # 快速评测（跳过上传，使用已有文档）
#   ./run_eval.sh --report     # 仅生成报告（需要有 results.json）
#
# 前置条件:
#   1. Alchemist Assistant 服务已启动: go run ./cmd/server
#   2. Python 3.10+ 已安装
#   3. (可选) 设置 RAGAS LLM 评判器:
#      export OPENAI_API_KEY="your-deepseek-key"
#      export OPENAI_BASE_URL="https://api.deepseek.com/v1"
#      export RAGAS_LLM_MODEL="deepseek-v4-pro"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================"
echo "  Alchemist Assistant RAG 评测"
echo "========================================"

# 检查 Python 环境
if ! command -v python3 &>/dev/null; then
    echo "❌ 需要 Python 3.10+"
    exit 1
fi

# 安装依赖
echo ""
echo "📦 检查 Python 依赖..."
pip3 install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || {
    echo "⚠️  部分依赖安装失败，尝试单独安装核心依赖..."
    pip3 install -q ragas datasets requests pyyaml pandas rich
}

# 配置 RAGAS 使用 DeepSeek 作为评判 LLM（OpenAI 兼容接口）
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
    export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com/v1}"
    export RAGAS_LLM_MODEL="${RAGAS_LLM_MODEL:-deepseek-v4-pro}"
    echo "✅ 使用 DeepSeek 作为 RAGAS 评判 LLM"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "✅ 使用 OpenAI 作为 RAGAS 评判 LLM"
else
    echo "⚠️  未设置 API Key，RAGAS 指标计算可能失败"
    echo "   请设置: export DEEPSEEK_API_KEY=your_key"
fi

# 检查服务
echo ""
echo "🔍 检查 Alchemist Assistant 服务..."
if curl -s --max-time 3 http://localhost:8090/api/status >/dev/null 2>&1; then
    echo "✅ 服务运行中"
else
    echo "❌ 服务未启动，请先运行: go run ./cmd/server"
    exit 1
fi

# 运行评测
echo ""
ARGS="${*:---base-url http://localhost:8090}"

if [ "$1" = "--quick" ]; then
    echo "🚀 快速评测模式（跳过上传）"
    python3 "$SCRIPT_DIR/evaluate_rag.py" \
        --base-url http://localhost:8090 \
        --skip-upload \
        --output "$SCRIPT_DIR/results.json"
elif [ "$1" = "--report" ]; then
    if [ -f "$SCRIPT_DIR/results.json" ]; then
        python3 -c "
import json
with open('$SCRIPT_DIR/results.json') as f:
    data = json.load(f)
# Re-run RAGAS from saved data
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import os

ed = data['eval_data']
dataset = Dataset.from_dict(ed)
score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(score.to_pandas().to_string())
"
    else
        echo "❌ 未找到 results.json"
        exit 1
    fi
else
    echo "🚀 完整评测模式"
    python3 "$SCRIPT_DIR/evaluate_rag.py" \
        --base-url http://localhost:8090 \
        --output "$SCRIPT_DIR/results.json" \
        "$@"
fi

echo ""
echo "✅ 评测完成"
