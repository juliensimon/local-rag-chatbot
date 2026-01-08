---
title: RAG
emoji: 🚀
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: "5.23.1"
app_file: app.py
pinned: false
---

# RAG - Document Question-Answering System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23.1-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

🚀 A Retrieval-Augmented Generation (RAG) powered chat interface for document Q&A using local LLM

## Overview
This application provides an interactive chat interface that allows users to ask questions about their documents. It combines the power of Large Language Models with document retrieval to provide accurate, source-backed answers.

## Features
- **RAG-Powered Responses**: Leverages document context to provide accurate, factual answers
- **Flexible Query Modes**: Switch between RAG and vanilla LLM responses
- **Source Citations**: Automatically includes relevant document sources and page numbers
- **Interactive Interface**: Clean, user-friendly Gradio-based chat interface
- **Context Visibility**: View the retrieved document chunks used to generate responses

## Technical Details
- Built with Langchain and Gradio
- Uses local llama-server (OpenAI-compatible API) for LLM capabilities
- Document embedding via BAAI/bge-small-en-v1.5
- ChromaDB for vector storage
- Supports PDF document processing

## Local LLM Setup with llama.cpp

This application uses a local llama-server with an OpenAI-compatible API. To run it with the Trinity-Mini Q8 model:

```bash
llama-server -hf arcee-ai/Trinity-Mini-GGUF:Q8_0
```

This will automatically download the Q8 quantized model (~27.8 GB) from [Hugging Face](https://huggingface.co/arcee-ai/Trinity-Mini-GGUF) and start the server.

## Included Documents
The `pdf` directory contains IEA (International Energy Agency) reports and publications covering topics such as:

- Clean energy transitions and net zero pathways
- Renewable energy capacity and projections
- Electric vehicles and battery technologies
- Carbon capture, utilisation and storage (CCUS)
- Energy efficiency and critical minerals
- Regional energy profiles and policy recommendations

## Deployment
This application is hosted as a Hugging Face Space. Configuration details can be found in the [spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference).

## Creating Your Own Hugging Face Space Using CLI

You can easily deploy this application as your own Hugging Face Space using the Hugging Face CLI. Follow these steps:

1. **Install the Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   You'll be prompted to enter your Hugging Face token, which you can find in your account settings.

3. **Clone this Repository**:
   ```bash
   git clone https://github.com/username/conductor-rag.git
   cd conductor-rag
   ```

4. **Create a New Space**:
   ```bash
   huggingface-cli repo create conductor-rag-your-name --type space --space-sdk gradio
   ```

5. **Add Your Environment Variables**:
   The application uses the following environment variables, which you need to set in the Space settings:
   - `OPENAI_API_KEY`: Optional - API key for the LLM server (not required for local llama-server)

7. **Push Your Code to the Space**:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/conductor-rag
   git push space main
   ```

8. **Add Your PDF Documents**:
   You can either add PDFs directly to the repository before pushing, or upload them later through git.

9. **Monitor Deployment**:
   Visit `https://huggingface.co/spaces/your-username/conductor-rag-your-name` to see your Space being built and deployed.

Your Space will automatically build and deploy the application. Once complete, you can access it via the provided URL and share it with others.

## Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces-overview)
- [Gradio Documentation](https://gradio.app/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

---
Built with 💖 using local LLM
