#!/usr/bin/env python3
"""
Alchemist Assistant RAG 知识库效果评测
========================================
使用 RAGAS 框架评测 RAG 管道的检索与生成质量。

评测指标：
  - Faithfulness（忠实度）：回答中的断言是否被检索上下文支持
  - Answer Relevancy（答案相关性）：回答是否切题、完整
  - Context Precision（上下文精确度）：检索到的 chunk 中有多少真正有用
  - Context Recall（上下文召回率）：是否检索到所有必要信息
  - Context Relevancy（上下文相关性）：检索到的内容与问题的相关程度

用法：
  python evaluate_rag.py [--base-url http://localhost:8090] [--llm deepseek-v4-pro]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# ─────────────────────────── 配置 ────────────────────────────────────────

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 尝试导入 RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    console.print(
        "[red]⚠️  RAGAS 未安装。请运行: pip install ragas datasets[/red]"
    )

# ─────────────────────────── API 客户端 ──────────────────────────────────


class AlchemistClient:
    """Alchemist Assistant API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8090"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        """检查服务是否可用"""
        try:
            resp = self.session.get(f"{self.base_url}/api/status", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def upload_document(self, content: str, title: str) -> Optional[str]:
        """上传文档到知识库，返回 doc_hash"""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/upload",
                json={"content": content, "title": title},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("doc_hash")
            else:
                logger.error(f"上传文档失败 ({resp.status_code}): {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            logger.error(f"上传文档异常: {e}")
            return None

    def chat_with_rag(
        self, question: str
    ) -> Tuple[str, List[str], List[float]]:
        """使用 RAG 模式提问，返回 (答案, 检索上下文列表, 相似度分数)"""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/chat",
                json={
                    "message": question,
                    "use_rag": True,
                    "explicit": True,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")

                # 提取检索到的上下文
                contexts = []
                scores = []
                search_results = data.get("search_results", [])
                for sr in search_results:
                    chunk = sr.get("chunk", {})
                    content = chunk.get("content", "")
                    if content:
                        contexts.append(content)
                    scores.append(sr.get("similarity", 0.0))

                return answer, contexts, scores
            else:
                logger.error(f"查询失败 ({resp.status_code}): {resp.text[:200]}")
                return "", [], []
        except requests.RequestException as e:
            logger.error(f"查询异常: {e}")
            return "", [], []

    def delete_document(self, doc_hash: str) -> bool:
        """删除文档"""
        try:
            resp = self.session.post(
                f"{self.base_url}/api/docs/delete",
                json={"doc_hash": doc_hash},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ─────────────────────────── 数据加载 ────────────────────────────────────


def load_dataset(yaml_path: str) -> Tuple[List[Dict], List[str]]:
    """加载评测数据集，返回 (文档列表, 问题列表)"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = []
    for doc in data.get("test_documents", []):
        documents.append({"id": doc["id"], "title": doc["title"], "content": doc["content"]})

    questions = []
    for q in data.get("questions", []):
        questions.append(
            {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "expected_contexts": q.get("expected_contexts", []),
            }
        )

    return documents, questions


# ─────────────────────────── 评测执行 ────────────────────────────────────


def run_evaluation(
    client: AlchemistClient,
    documents: List[Dict],
    questions: List[Dict],
) -> Dict:
    """执行完整评测流程"""

    # ── Step 1: 上传文档 ──
    console.print("\n[bold cyan]📄 Step 1: 上传评测文档到知识库[/bold cyan]")
    doc_hashes = []
    for doc in documents:
        console.print(f"  上传: {doc['title']} ({len(doc['content'])} 字符)")
        doc_hash = client.upload_document(doc["content"], doc["title"])
        if doc_hash:
            doc_hashes.append(doc_hash)
            console.print(f"    ✅ doc_hash: {doc_hash}")
        else:
            console.print(f"    ❌ 上传失败")
    console.print(f"  共上传 {len(doc_hashes)}/{len(documents)} 篇文档")

    if not doc_hashes:
        console.print("[red]❌ 没有文档上传成功，终止评测[/red]")
        return {}

    # 等待索引完成
    console.print("\n[yellow]⏳ 等待文档索引完成（5 秒）...[/yellow]")
    time.sleep(5)

    # ── Step 2: 逐条查询 ──
    console.print(f"\n[bold cyan]🔍 Step 2: 执行 {len(questions)} 条查询[/bold cyan]")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("查询中...", total=len(questions))

        for q in questions:
            progress.update(task, description=f"查询: {q['question'][:50]}...")
            answer, contexts, scores = client.chat_with_rag(q["question"])

            eval_data["question"].append(q["question"])
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts if contexts else [""])
            eval_data["ground_truth"].append(q["ground_truth"])

            progress.advance(task)

    # ── Step 3: 运行 RAGAS 评测 ──
    console.print("\n[bold cyan]📊 Step 3: 运行 RAGAS 评测[/bold cyan]")

    if not RAGAS_AVAILABLE:
        console.print("[red]❌ RAGAS 不可用，跳过指标计算[/red]")
        return eval_data

    # 构建 HuggingFace Dataset
    dataset = Dataset.from_dict(eval_data)

    # 选择指标
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    console.print(f"  评测指标: faithfulness, answer_relevancy, context_precision, context_recall")
    console.print(f"  数据条数: {len(dataset)}")

    try:
        # 如果配置了 LLM 用于 RAGAS 评判，使用用户指定的模型
        # RAGAS 默认使用 OpenAI，需配置环境变量
        score = evaluate(dataset, metrics=metrics)
        eval_results = score.to_pandas()
        console.print("  ✅ RAGAS 评测完成")
    except Exception as e:
        logger.error(f"RAGAS 评测异常: {e}")
        console.print(f"[red]❌ RAGAS 评测失败: {e}[/red]")
        console.print("[yellow]💡 可能需要配置 LLM API Key（RAGAS 使用 LLM 作为评判器）[/yellow]")
        console.print("[yellow]   export OPENAI_API_KEY=your_key  # 或设置其他兼容的 LLM[/yellow]")
        return eval_data

    # ── Step 4: 清理 ──
    console.print("\n[bold cyan]🧹 Step 4: 清理评测文档[/bold cyan]")
    for doc_hash in doc_hashes:
        client.delete_document(doc_hash)
    console.print(f"  已删除 {len(doc_hashes)} 篇文档")

    return {
        "eval_data": eval_data,
        "ragas_results": eval_results,
    }


# ─────────────────────────── 报告生成 ────────────────────────────────────


def print_report(results: Dict):
    """打印评测报告"""
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]  Alchemist Assistant RAG 知识库评测报告[/bold cyan]")
    console.print("=" * 70)

    eval_data = results.get("eval_data", {})
    if eval_data:
        console.print(f"\n[bold]评测数据概览:[/bold]")
        console.print(f"  问题数量: {len(eval_data.get('question', []))}")
        contexts_list = eval_data.get("contexts", [])
        avg_contexts = (
            sum(len(c) for c in contexts_list) / len(contexts_list)
            if contexts_list
            else 0
        )
        console.print(f"  平均检索 chunk 数: {avg_contexts:.1f}")

    # RAGAS 指标表
    ragas_df = results.get("ragas_results")
    if ragas_df is not None and not ragas_df.empty:
        console.print(f"\n[bold]RAGAS 评测指标:[/bold]")

        table = Table(title="RAGAS Metrics")
        table.add_column("指标", style="cyan")
        table.add_column("均值", style="green")
        table.add_column("中位数", style="yellow")
        table.add_column("标准差", style="dim")
        table.add_column("解读", style="white")

        metrics_info = {
            "faithfulness": ("忠实度", "回答是否基于检索内容，>0.90 优秀"),
            "answer_relevancy": ("答案相关性", "回答是否切题，>0.85 良好"),
            "context_precision": ("上下文精确度", "检索信号是否精准，>0.75 可用"),
            "context_recall": ("上下文召回率", "是否检索到全部必要信息，>0.80 良好"),
        }

        for col in ragas_df.columns:
            if col in metrics_info:
                name, desc = metrics_info[col]
                vals = ragas_df[col].dropna()
                if len(vals) > 0:
                    table.add_row(
                        f"{name}\n({col})",
                        f"{vals.mean():.4f}",
                        f"{vals.median():.4f}",
                        f"{vals.std():.4f}",
                        desc,
                    )

        console.print(table)

        # 综合评分
        score_cols = [c for c in ragas_df.columns if c in metrics_info]
        if score_cols:
            overall = ragas_df[score_cols].mean().mean()
            console.print(f"\n[bold]综合评分: {overall:.4f}[/bold]")
            if overall >= 0.85:
                console.print("[green]✅ 知识库检索质量优秀[/green]")
            elif overall >= 0.70:
                console.print("[yellow]⚠️  知识库检索质量一般，有优化空间[/yellow]")
            else:
                console.print("[red]❌ 知识库检索质量较差，需要重点优化[/red]")

    # 逐条详情
    console.print(f"\n[bold]逐条查询详情:[/bold]")
    questions = eval_data.get("question", [])
    answers = eval_data.get("answer", [])
    contexts_list = eval_data.get("contexts", [])

    for i in range(len(questions)):
        console.print(f"\n[bold]Q{i+1}:[/bold] {questions[i][:80]}...")
        answer = answers[i] if i < len(answers) else ""
        console.print(f"  [dim]A: {answer[:150]}...[/dim]" if len(answer) > 150 else f"  [dim]A: {answer}[/dim]")
        ctxs = contexts_list[i] if i < len(contexts_list) else []
        console.print(f"  [dim]检索到 {len(ctxs)} 个 chunk[/dim]")

    console.print("\n" + "=" * 70)


# ─────────────────────────── 主入口 ──────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Alchemist Assistant RAG 知识库评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate_rag.py
  python evaluate_rag.py --base-url http://localhost:8090
  python evaluate_rag.py --skip-upload  # 文档已在知识库中
  python evaluate_rag.py --output results.json
        """,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8090",
        help="Alchemist Assistant 服务地址 (默认: http://localhost:8090)",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "eval_dataset.yaml"),
        help="评测数据集路径",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="跳过文档上传（知识库中已有评测文档）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出结果 JSON 文件路径",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="评测后不删除上传的文档",
    )
    args = parser.parse_args()

    # 检查数据集
    if not os.path.exists(args.dataset):
        console.print(f"[red]❌ 数据集文件不存在: {args.dataset}[/red]")
        sys.exit(1)

    # 检查 RAGAS
    if not RAGAS_AVAILABLE:
        console.print(
            "[yellow]💡 RAGAS 未安装。将仅收集原始数据，不计算指标。[/yellow]"
        )
        console.print(
            "[yellow]   安装: pip install ragas datasets[/yellow]"
        )

    # 检查服务
    client = AlchemistClient(args.base_url)
    console.print(f"\n🔍 检查服务连接: {args.base_url}")
    if not client.health_check():
        console.print(
            f"[red]❌ 无法连接到 {args.base_url}[/red]\n"
            f"   请确保服务已启动: go run ./cmd/server"
        )
        sys.exit(1)
    console.print("[green]✅ 服务可用[/green]")

    # 加载数据集
    documents, questions = load_dataset(args.dataset)
    console.print(
        f"📋 数据集: {len(documents)} 篇文档, {len(questions)} 条问题"
    )

    # 执行评测
    if args.skip_upload:
        # 跳过上传，直接查询
        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("查询中...", total=len(questions))
            for q in questions:
                progress.update(task, description=f"查询: {q['question'][:50]}...")
                answer, contexts, scores = client.chat_with_rag(q["question"])
                eval_data["question"].append(q["question"])
                eval_data["answer"].append(answer)
                eval_data["contexts"].append(contexts if contexts else [""])
                eval_data["ground_truth"].append(q["ground_truth"])
                progress.advance(task)

        if RAGAS_AVAILABLE:
            dataset = Dataset.from_dict(eval_data)
            score = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            results = {"eval_data": eval_data, "ragas_results": score.to_pandas()}
        else:
            results = {"eval_data": eval_data}
    else:
        results = run_evaluation(client, documents, questions)

    # 生成报告
    print_report(results)

    # 保存结果
    if args.output:
        output_path = args.output
        save_data = {"eval_data": results.get("eval_data", {})}
        ragas_df = results.get("ragas_results")
        if ragas_df is not None and not ragas_df.empty:
            save_data["ragas_metrics"] = ragas_df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        console.print(f"\n📁 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
