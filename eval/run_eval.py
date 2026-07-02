"""
Simple evaluation harness: hits the running /query endpoint with a set of
golden questions and checks whether expected keywords show up in the answer.

Usage:
    1. Make sure the API is running locally (uvicorn app.main:app --reload)
       and you've ingested at least one document via /ingest.
    2. Edit eval_questions.json to match the documents you ingested.
    3. Run: python eval/run_eval.py
"""

import json
import time
from pathlib import Path

import requests

API_URL = "http://localhost:8000/query"
QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def run() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    results = []
    correct = 0
    total_latency = 0.0

    for item in questions:
        start = time.time()
        resp = requests.post(API_URL, json={"question": item["question"]})
        elapsed = time.time() - start
        total_latency += elapsed

        data = resp.json()
        answer = data.get("answer", "")
        passed = any(kw.lower() in answer.lower() for kw in item["expected_keywords"])
        correct += int(passed)

        results.append(
            {
                "question": item["question"],
                "answer": answer,
                "passed": passed,
                "latency_s": round(elapsed, 2),
            }
        )

    accuracy = correct / len(questions) if questions else 0
    avg_latency = total_latency / len(questions) if questions else 0

    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(questions)})")
    print(f"Avg latency: {avg_latency:.2f}s")

    RESULTS_PATH.write_text(
        json.dumps({"accuracy": accuracy, "avg_latency_s": avg_latency, "results": results}, indent=2)
    )
    print(f"Full results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
