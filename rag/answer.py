"""Answer: find the closest papers, then let an AI model write a cited answer.

    python rag/answer.py "What methods reduce hallucinations in large language models?"
"""

import argparse

from common import REGION
from search import find_papers

# In this region the model is called through an "inference profile" (the eu. prefix).
LLM_MODEL_ID = "eu.amazon.nova-micro-v1:0"
MAX_ABSTRACT_CHARS = 1200  # keeps each question small and cheap

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered "
    "sources provided. Cite sources like [1] or [2] after the claims they support. "
    "If the sources do not contain enough information, say so plainly instead of "
    "guessing. Keep the answer under 150 words."
)


def build_prompt(question, papers):
    """Put the question and the retrieved papers into one prompt."""
    sources = []
    for number, paper in enumerate(papers, start=1):
        abstract = paper["abstract"][:MAX_ABSTRACT_CHARS]
        sources.append(f"[{number}] {paper['title']}\n{abstract}")
    return "Sources:\n\n" + "\n\n".join(sources) + f"\n\nQuestion: {question}"


def ask_llm(prompt, client=None):
    """Send the prompt to the AI model on Amazon Bedrock and return its text."""
    if client is None:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.converse(
        modelId=LLM_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]


def answer_question(question, top_k=5, model=None, vectors_client=None, llm_client=None):
    """Full RAG loop: retrieve, then generate. Returns (answer, papers)."""
    papers = find_papers(question, top_k, model=model, client=vectors_client)
    return ask_llm(build_prompt(question, papers), llm_client), papers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    text, papers = answer_question(args.question, args.top_k)
    print(text)
    print("\nSources:")
    for number, paper in enumerate(papers, start=1):
        print(f"  [{number}] {paper['title']}\n      {paper['url']}")


if __name__ == "__main__":
    main()
