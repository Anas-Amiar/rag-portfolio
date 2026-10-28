"""Evaluation: score the search with a fixed set of test questions.

    python rag/evaluate.py

For each question in eval/questions.json we run the search and count how many of
the top papers mention at least one of the question's keywords. This is a rough,
cheap proxy for "is the paper relevant?". It does not replace reading the results.

Two numbers come out:
  precision@k : of the k papers returned, what share were relevant (average)
  hit@k       : in what share of questions was at least one paper relevant
"""

import argparse
import json
from pathlib import Path

from search import find_papers

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def is_relevant(paper, keywords):
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return any(keyword.lower() in text for keyword in keywords)


def score_question(papers, keywords):
    """Return (number of relevant papers, number of papers)."""
    relevant = sum(1 for paper in papers if is_relevant(paper, keywords))
    return relevant, len(papers)


def summarize(per_question):
    """per_question: list of (relevant, total). Returns (precision, hit_rate)."""
    precision = sum(r / t for r, t in per_question if t) / len(per_question)
    hit_rate = sum(1 for r, _ in per_question if r > 0) / len(per_question)
    return precision, hit_rate


def run(questions, top_k, search):
    results = []
    for item in questions:
        papers = search(item["question"], top_k)
        results.append(score_question(papers, item["keywords"]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from common import get_embedder, get_vectors_client

    model, client = get_embedder(), get_vectors_client()
    questions = json.loads(QUESTIONS_FILE.read_text())

    def search(question, k):
        return find_papers(question, k, model=model, client=client)

    results = run(questions, args.top_k, search)
    for item, (relevant, total) in zip(questions, results):
        print(f"{relevant}/{total}  {item['question']}")

    precision, hit_rate = summarize(results)
    print(f"\nprecision@{args.top_k}: {precision:.2f}")
    print(f"hit@{args.top_k}:       {hit_rate:.2f}")


if __name__ == "__main__":
    main()
