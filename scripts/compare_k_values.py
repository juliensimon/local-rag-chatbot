#!/usr/bin/env python3
"""Compare RAG retrieval quality with different k values."""

import sys
sys.path.insert(0, '.')

from models import create_embeddings
from vectorstore import load_or_create_vectorstore
from config import MMR_LAMBDA

# Test questions
TEST_QUESTIONS = [
    "What are the CO2 emissions reduction targets in the Net Zero Roadmap?",
    "What is the projected global renewable energy capacity for 2028?",
    "What role does energy efficiency play in reducing global energy demand?",
]

K_VALUES = [3, 5, 7]


def test_retrieval_with_k(vectorstore, k_value: int, question: str) -> dict:
    """Run retrieval with a specific k value."""
    # Create retriever with specific k
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_value,
            "fetch_k": max(k_value * 3, 10),  # Fetch more candidates for MMR
            "lambda_mult": MMR_LAMBDA,
        },
    )

    # Retrieve documents
    docs = retriever.invoke(question)

    return {
        "k": k_value,
        "num_sources": len(docs),
        "sources": [
            {
                "file": d.metadata.get("source", "unknown").split("/")[-1],
                "page": d.metadata.get("page", 0),
                "content_preview": d.page_content[:100].replace("\n", " ")
            }
            for d in docs
        ]
    }


def main():
    print("Loading vectorstore...")
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)

    print("\n" + "=" * 80)
    print("RAG RETRIEVAL COMPARISON - Testing different k values")
    print("=" * 80)

    for question in TEST_QUESTIONS:
        print(f"\n{'─' * 80}")
        print(f"Q: {question}")
        print("─" * 80)

        results = {}
        for k in K_VALUES:
            result = test_retrieval_with_k(vectorstore, k, question)
            results[k] = result
            print(f"\nk={k}: Retrieved {result['num_sources']} chunks")
            for i, src in enumerate(result['sources'], 1):
                print(f"  [{i}] {src['file']} p.{src['page']}: {src['content_preview'][:60]}...")

        # Analysis: What did higher k values add?
        print(f"\n📊 Analysis:")
        base_sources = set((s['file'], s['page']) for s in results[3]['sources'])
        for k in [5, 7]:
            new_sources = set((s['file'], s['page']) for s in results[k]['sources'])
            added = new_sources - base_sources
            if added:
                print(f"   k={k} added: {[f'{f} p.{p}' for f,p in added]}")
            else:
                print(f"   k={k}: Same top-{K_VALUES[0]} sources (good consistency)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If sources are consistent across k values:
  ✓ Your retrieval is stable - k=3 is likely sufficient

If k=5 or k=7 adds relevant new sources:
  → Consider increasing RETRIEVER_K in config.py

If k=5 or k=7 adds mostly irrelevant sources:
  ✓ k=3 is optimal - additional chunks add noise
""")


if __name__ == "__main__":
    main()
