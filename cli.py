"""CLI entry point for testing RAG functionality."""

from models import create_embeddings, create_llm
from qa_chain import create_qa_chain
from vectorstore import load_or_create_vectorstore


def main():
    """Main execution function for CLI testing."""
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    qa_chain = create_qa_chain(vectorstore)

    query = "Tell me about Arcee Fusion."
    chat_history = []

    # Test vanilla response
    print("\n=== Vanilla Response (No RAG) ===")
    streaming_llm = create_llm(streaming=True)
    print("Answer: ", end="", flush=True)
    try:
        for chunk in streaming_llm.stream(query):
            print(chunk.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nError: {e}")

    # Test RAG response with MMR (diverse results)
    print("\n=== RAG Response (MMR for diversity) ===")
    print("Answer: ", end="", flush=True)
    chunk_data = None
    try:
        for chunk_data in qa_chain.stream(
            {"question": query, "chat_history": chat_history}
        ):
            print(chunk_data["chunk"], end="", flush=True)
        print()

        # Print sources
        if chunk_data and chunk_data.get("source_documents"):
            print("\nSources:")
            seen_sources = set()
            for doc in chunk_data["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "unknown")
                source_key = f"{source}:{page}"
                if source_key not in seen_sources:
                    print(f"- {source}, page {page}")
                    seen_sources.add(source_key)
    except Exception as e:
        print(f"\nError: {e}")

    # Example: RAG with metadata filter (filter by page number)
    print("\n=== RAG Response with Metadata Filter (page >= 5) ===")
    print("Answer: ", end="", flush=True)
    chunk_data = None
    try:
        for chunk_data in qa_chain.stream(
            {
                "question": query,
                "chat_history": chat_history,
                "filter": {"page": {"$gte": 5}},  # Only pages 5+
            }
        ):
            print(chunk_data["chunk"], end="", flush=True)
        print()

        if chunk_data and chunk_data.get("source_documents"):
            print("\nSources (filtered):")
            seen_sources = set()
            for doc in chunk_data["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "unknown")
                source_key = f"{source}:{page}"
                if source_key not in seen_sources:
                    print(f"- {source}, page {page}")
                    seen_sources.add(source_key)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

