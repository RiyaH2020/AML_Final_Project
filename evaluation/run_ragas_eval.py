import os
import json
import warnings
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Suppress deprecation warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ── 1. LLM ─────────────────────────────────────────────────────────────────
# 70b has 6000 TPM too but handles large contexts better than 8b
# gemma2-9b-it has 15000 TPM — best free tier option for large contexts
base_llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    n=1,
    max_tokens=2048,    # ← prevents LLMDidNotFinishException
    max_retries=10,
)
judge_llm = LangchainLLMWrapper(base_llm)

# ── 2. METRICS ─────────────────────────────────────────────────────────────
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

metrics = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, strictness=1),
    ContextRecall(llm=judge_llm),
    ContextPrecision(llm=judge_llm),
]

# ── 3. RUN CONFIG ──────────────────────────────────────────────────────────
run_config = RunConfig(
    max_workers=1,
    timeout=240,
    max_wait=120,
)


def main():
    input_path = "evaluation/eval_results_filtered.json"
    if not os.path.exists(input_path):
        print(" File not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data = data[:5]
    print(f" Testing with {len(test_data)} samples...")

    dataset = Dataset.from_dict({
        "question":     [item["question"]    for item in test_data],
        "answer":       [item["answer"]       for item in test_data],
        "contexts":     [item["contexts"]     for item in test_data],
        "ground_truth": [item["ground_truth"] for item in test_data],
    })

    print(f" Starting throttled evaluation (Total tasks: {len(test_data) * len(metrics)})...")

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            run_config=run_config,
        )

        df_results = result.to_pandas()
        report_path = "evaluation/test_metrics_5_samples.csv"
        df_results.to_csv(report_path, index=False)

        print("\n" + "=" * 30)
        print(" TEST RESULTS SUMMARY")
        print("=" * 30)
        print(result)
        print(f"\n Detailed CSV saved to: {report_path}")

    except Exception as e:
        print(f" Still hitting limits: {e}")


if __name__ == "__main__":
    main()