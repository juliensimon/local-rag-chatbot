# RAG - Document Question-Answering System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)

🚀 A Retrieval-Augmented Generation (RAG) powered chat interface for document Q&A using local LLM

## Overview
This application provides an interactive chat interface that allows users to ask questions about their documents. It combines the power of Large Language Models with document retrieval to provide accurate, source-backed answers.

## Features
- **RAG-Powered Responses**: Leverages document context to provide accurate, factual answers
- **Flexible Query Modes**: Switch between RAG and vanilla LLM responses
- **Multiple Search Types**: MMR (diversity), Similarity, and Hybrid (semantic + keyword)
- **Advanced Options**: Query rewriting and cross-encoder re-ranking
- **Source Citations**: Automatically includes relevant document sources and page numbers
- **Modern React UI**: Clean, responsive interface with real-time streaming
- **Context Visibility**: View the retrieved document chunks used to generate responses
- **Dark/Light Mode**: Theme support with system preference detection

## Technical Details
- **Backend**: FastAPI with LangChain for RAG pipeline
- **Frontend**: React 18 + TypeScript + Vite with Tailwind CSS
- **LLM**: Local llama-server (OpenAI-compatible API)
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Vector Store**: ChromaDB for document storage
- **Streaming**: Server-Sent Events (SSE) for real-time responses
- **Documents**: PDF processing with automatic chunking

## Local LLM Setup with llama.cpp

This application uses a local llama-server with an OpenAI-compatible API. To run it with the Trinity-Mini Q8 model:

```bash
llama-server -hf arcee-ai/Trinity-Mini-GGUF:Q8_0
```

This will automatically download the Q8 quantized model (~27.8 GB) from [Hugging Face](https://huggingface.co/arcee-ai/Trinity-Mini-GGUF) and start the server.

## Running the Application

```bash
# Terminal 1: Start the API server
uvicorn api.main:app --reload

# Terminal 2: Start the React frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## Included Documents
The `pdf` directory contains IEA (International Energy Agency) reports and publications covering topics such as:

- Clean energy transitions and net zero pathways
- Renewable energy capacity and projections
- Electric vehicles and battery technologies
- Carbon capture, utilisation and storage (CCUS)
- Energy efficiency and critical minerals
- Regional energy profiles and policy recommendations

## Deployment

### Local Development
The application runs locally with:
- FastAPI backend on port 8000
- React frontend on port 5173 (with Vite proxy to backend)

### Production Build
```bash
cd frontend
npm run build
```
The built files in `frontend/dist/` can be served by any static file server. Configure your server to proxy `/api/*` requests to the FastAPI backend.

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

---
Built with 💖 using local LLM
