"""Tests for the FastAPI routes, SSE streaming helpers and app factory."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import routes
from api.main import create_api_app, initialize_qa_chain
from api.streaming import (
    build_context_response,
    format_sse_event,
    stream_rag_response,
    stream_vanilla_response,
)
from config import PDF_PATH


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split an SSE body into (event, data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


@pytest.fixture
def docs():
    return [
        Document(page_content="First chunk", metadata={"source": "pdf/a.pdf", "page": 1}),
        Document(page_content="Second chunk", metadata={"source": "pdf/b.pdf", "page": 2}),
    ]


@pytest.fixture
def qa_chain(docs):
    chain = MagicMock()
    chain.stream.return_value = [
        {"chunk": "Hello ", "source_documents": docs},
        {
            "chunk": "world",
            "source_documents": docs,
            "docs_with_scores": [(docs[0], 0.9), (docs[1], 0.5)],
            "rewritten_query": "rewritten",
        },
    ]
    return chain


@pytest.fixture
def client(qa_chain):
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    routes.init_routes(qa_chain, ["a.pdf", "b.pdf"])
    yield TestClient(app)
    routes.init_routes(None, [])


@pytest.fixture
def uninitialized_client():
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    routes.init_routes(None, [])
    return TestClient(app)


class TestStreamingHelpers:
    def test_format_sse_event(self):
        assert format_sse_event("token", {"content": "x"}) == (
            'event: token\ndata: {"content": "x"}\n\n'
        )

    def test_build_context_response_empty(self):
        context = build_context_response([], rewritten_query="q")
        assert context.sources == []
        assert context.rewritten_query == "q"

    def test_build_context_response_with_scores(self, docs):
        context = build_context_response(docs, [(docs[0], 0.2), (docs[1], 0.8)])
        assert [s.score for s in context.sources] == [0.2, 0.8]
        assert sum(s.is_top for s in context.sources) == 1

    def test_build_context_response_with_hybrid_scores(self, docs):
        hybrid = [(docs[0], 0.7, 0.6, None), (docs[1], None, 0.1, 0.3)]
        context = build_context_response(docs, hybrid_scores=hybrid)
        first, second = context.sources
        assert (first.score, first.semantic_score, first.keyword_score) == (0.7, 0.6, None)
        assert (second.score, second.semantic_score, second.keyword_score) == (None, 0.1, 0.3)

    def test_stream_rag_response_passes_filter(self, qa_chain):
        events = parse_sse(
            "".join(
                stream_rag_response(
                    qa_chain, "q", [], "mmr", {"source": {"$eq": "x"}}, False, False, 0.7
                )
            )
        )
        assert [e for e, _ in events] == ["token", "token", "context", "done"]
        assert qa_chain.stream.call_args[0][0]["filter"] == {"source": {"$eq": "x"}}

    def test_stream_vanilla_response(self):
        llm = MagicMock()
        llm.stream.return_value = [MagicMock(content="Hi"), MagicMock(content="")]
        events = parse_sse("".join(stream_vanilla_response(llm, "q")))
        assert events == [("token", {"content": "Hi"}), ("done", {})]

    def test_stream_vanilla_response_error(self):
        llm = MagicMock()
        llm.stream.side_effect = RuntimeError("server down")
        events = parse_sse("".join(stream_vanilla_response(llm, "q")))
        assert events == [("error", {"message": "server down"})]


class TestValidateDocFilter:
    def test_all_documents_means_no_filter(self, client):
        assert routes.validate_doc_filter(None) is None
        assert routes.validate_doc_filter("All Documents") is None

    def test_unknown_source_is_ignored(self, client):
        assert routes.validate_doc_filter("../etc/passwd") is None

    def test_known_source(self, client):
        assert routes.validate_doc_filter("a.pdf") == {"source": {"$eq": os.path.join(PDF_PATH, "a.pdf")}}


class TestRoutes:
    def test_health_ready(self, client):
        assert client.get("/api/health").json() == {
            "status": "healthy",
            "vectorstore_ready": True,
            "llm_ready": True,
        }

    def test_health_degraded(self, uninitialized_client):
        assert uninitialized_client.get("/api/health").json()["status"] == "degraded"

    def test_sources(self, client):
        assert client.get("/api/sources").json() == {"sources": ["a.pdf", "b.pdf"]}

    @pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
    def test_chat_before_init_returns_503(self, uninitialized_client, path):
        response = uninitialized_client.post(path, json={"message": "hi", "rag_enabled": True})
        assert response.status_code == 503

    def test_chat_rag(self, client, qa_chain):
        response = client.post(
            "/api/chat",
            json={
                "message": "What is CCUS?",
                "rag_enabled": True,
                "doc_filter": "a.pdf",
                "hybrid_alpha": 40,
                "history": [
                    {"role": "user", "content": "Earlier question"},
                    {"role": "assistant", "content": "Earlier answer"},
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "Hello world"
        assert body["context"]["rewritten_query"] == "rewritten"
        assert len(body["context"]["sources"]) == 2

        stream_input = qa_chain.stream.call_args[0][0]
        assert stream_input["hybrid_alpha"] == 0.4
        assert stream_input["filter"] == {"source": {"$eq": os.path.join(PDF_PATH, "a.pdf")}}
        assert stream_input["chat_history"] == [("Earlier question", "Earlier answer")]

    @patch("models.create_llm")
    def test_chat_vanilla(self, mock_create_llm, client):
        mock_create_llm.return_value.invoke.return_value = MagicMock(content="Plain answer")
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.json() == {"response": "Plain answer", "context": None}
        mock_create_llm.assert_called_once_with(streaming=False)

    def test_chat_stream_rag(self, client):
        response = client.post("/api/chat/stream", json={"message": "hi", "rag_enabled": True})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert [e for e, _ in events] == ["token", "token", "context", "done"]

    @patch("models.create_llm")
    def test_chat_stream_vanilla(self, mock_create_llm, client):
        mock_create_llm.return_value.stream.return_value = [MagicMock(content="Hi")]
        response = client.post("/api/chat/stream", json={"message": "hi"})
        assert parse_sse(response.text) == [("token", {"content": "Hi"}), ("done", {})]
        mock_create_llm.assert_called_once_with(streaming=True)


class TestAppFactory:
    def test_create_api_app_mounts_routes(self):
        paths = set(create_api_app().openapi()["paths"])
        assert {"/api/health", "/api/sources", "/api/chat", "/api/chat/stream"} <= paths

    @patch("api.main.create_qa_chain")
    @patch("api.main.load_or_create_vectorstore")
    @patch("api.main.create_embeddings")
    def test_initialize_qa_chain_lists_sources(self, _embeddings, mock_load, mock_chain):
        mock_load.return_value.get.return_value = {
            "metadatas": [{"source": "pdf/b.pdf"}, {"source": "pdf/a.pdf"}, {"source": "pdf/a.pdf"}, {}]
        }
        chain, sources = initialize_qa_chain()
        assert chain is mock_chain.return_value
        assert sources == ["a.pdf", "b.pdf"]

    @patch("api.main.create_qa_chain")
    @patch("api.main.load_or_create_vectorstore")
    @patch("api.main.create_embeddings")
    def test_initialize_qa_chain_empty_store(self, _embeddings, mock_load, _chain):
        mock_load.return_value.get.return_value = {"metadatas": []}
        _, sources = initialize_qa_chain()
        assert sources == []
