"""
RAGAS 配置：使用 DeepSeek 作为评判 LLM（OpenAI 兼容接口）

用法：
    from ragas_config import configure_ragas
    configure_ragas()

或在运行 evaluate_rag.py 之前设置环境变量：
    export OPENAI_API_KEY="your-deepseek-key"
    export OPENAI_BASE_URL="https://api.deepseek.com/v1"
    export RAGAS_LLM_MODEL="deepseek-v4-pro"
"""

import os


def configure_ragas(
    api_key: str = None,
    base_url: str = None,
    model: str = None,
):
    """
    配置 RAGAS 使用 OpenAI 兼容的 LLM API 作为评判器。

    RAGAS 底层使用 LangChain 的 ChatOpenAI，设置这些环境变量
    即可将任何 OpenAI 兼容的 API（DeepSeek、vLLM、Together 等）
    用作 RAGAS 的评判 LLM。

    参数:
        api_key: API Key（默认从 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 读取）
        base_url: API 地址（默认 https://api.deepseek.com/v1）
        model: 模型名称（默认 deepseek-v4-pro）
    """
    # API Key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "DEEPSEEK_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    elif "OPENAI_API_KEY" not in os.environ:
        print("⚠️  未设置 OPENAI_API_KEY，RAGAS 评判 LLM 将不可用")

    # Base URL
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    elif "OPENAI_BASE_URL" not in os.environ:
        # 默认使用 DeepSeek
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
        print("ℹ️  默认使用 DeepSeek API (https://api.deepseek.com/v1)")

    # Model
    if model:
        os.environ["RAGAS_LLM_MODEL"] = model
    elif "RAGAS_LLM_MODEL" not in os.environ:
        os.environ["RAGAS_LLM_MODEL"] = "deepseek-v4-pro"


# ─────────────────────────── 指标说明 ────────────────────────────────────

METRICS_GUIDE = {
    "faithfulness": {
        "name_zh": "忠实度 (Faithfulness)",
        "range": "0-1",
        "excellent": "> 0.90",
        "good": "> 0.80",
        "description": (
            "评估生成的回答中，有多少断言（claims）能在检索的上下文中找到支撑。\n"
            "这是最重要的 RAG 质量指标之一——分数低意味着模型在'编造'信息。\n"
            "改进方法：优化系统提示词，要求模型严格基于上下文回答。"
        ),
    },
    "answer_relevancy": {
        "name_zh": "答案相关性 (Answer Relevancy)",
        "range": "0-1",
        "excellent": "> 0.90",
        "good": "> 0.85",
        "description": (
            "评估生成的回答与原始问题的相关程度，包括完整性、聚焦度和是否跑题。\n"
            "分数低可能意味着模型回答了不相关的内容，或没有完全理解问题。\n"
            "改进方法：优化查询改写（Query Rewrite），提升问题表达的清晰度。"
        ),
    },
    "context_precision": {
        "name_zh": "上下文精确度 (Context Precision)",
        "range": "0-1",
        "excellent": "> 0.85",
        "good": "> 0.75",
        "description": (
            "评估检索到的上下文中有多少是真正有用的（信号 vs 噪声）。\n"
            "也评估相关 chunk 是否被排在更高的位置（排序质量）。\n"
            "分数低意味着检索器返回了大量不相关内容。\n"
            "改进方法：调整 RRF 权重、优化 chunk 大小、提升 embedding 质量。"
        ),
    },
    "context_recall": {
        "name_zh": "上下文召回率 (Context Recall)",
        "range": "0-1",
        "excellent": "> 0.90",
        "good": "> 0.80",
        "description": (
            "评估检索是否找到了回答所需的所有必要信息。\n"
            "分数低意味着有重要信息未被检索到，导致回答不完整。\n"
            "改进方法：增加检索 Top-K、使用混合检索、引入查询改写/扩展。"
        ),
    },
}


def print_metrics_guide():
    """打印指标说明"""
    print("\n" + "=" * 60)
    print("  RAGAS 评测指标说明")
    print("=" * 60)
    for key, info in METRICS_GUIDE.items():
        print(f"\n📊 {info['name_zh']}")
        print(f"   范围: {info['range']}")
        print(f"   优秀: {info['excellent']}, 良好: {info['good']}")
        print(f"   {info['description']}")


if __name__ == "__main__":
    print_metrics_guide()
