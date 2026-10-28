"""Tests for the scoring logic in rag/evaluate.py. They need no AWS access."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))

import evaluate  # noqa: E402

PAPER = {"title": "Mitigating Hallucination", "abstract": "We study LLMs."}
OTHER = {"title": "x", "abstract": "y"}


def test_questions_file_is_well_formed():
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    assert len(questions) == 20
    assert all(q["question"] and q["keywords"] for q in questions)


def test_is_relevant_ignores_case():
    assert evaluate.is_relevant(PAPER, ["HALLUCINAT"])
    assert not evaluate.is_relevant(PAPER, ["diffusion"])


def test_score_question_counts_relevant_papers():
    assert evaluate.score_question([PAPER, PAPER, OTHER], ["hallucinat"]) == (2, 3)


def test_summarize():
    precision, hit_rate = evaluate.summarize([(5, 5), (0, 5), (2, 4)])
    assert abs(precision - 0.5) < 1e-9
    assert abs(hit_rate - 2 / 3) < 1e-9


def test_run_uses_the_search_function():
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())[:2]
    results = evaluate.run(questions, 5, lambda q, k: [PAPER] * k)
    assert results == [(5, 5)] * len(results)
