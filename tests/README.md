# Test Suite

This directory contains the test suite for the RAG application.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=. --cov-report=term-missing
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run specific test
```bash
pytest tests/test_utils.py::test_format_chat_history_empty
```

## Coverage

The test suite aims for 80% code coverage. Current coverage: **84%**

### Coverage by Module

- `config.py`: 100%
- `models.py`: 100%
- `utils.py`: 95%
- `vectorstore.py`: 91%
- `retrievers.py`: 89%
- `qa_chain.py`: 83%
- `ui/handlers.py`: 100%

### Excluded from Coverage

- `app.py`: Entry point (minimal logic)
- `cli.py`: CLI entry point (integration testing)
- `ui/app.py`: UI setup (requires full Gradio integration)
- `ui/components.py`: UI component definitions

## Test Structure

- `conftest.py`: Shared fixtures and test configuration
- `test_config.py`: Configuration constants tests
- `test_models.py`: Model creation tests
- `test_utils.py`: Utility function tests
- `test_vectorstore.py`: Vectorstore management tests
- `test_retrievers.py`: Retrieval strategy tests
- `test_qa_chain.py`: QA chain wrapper tests
- `test_handlers.py`: UI handler function tests

## Fixtures

Common fixtures available in `conftest.py`:

- `mock_embeddings`: Mock embedding model
- `mock_vectorstore`: Mock vectorstore
- `sample_documents`: Sample document objects
- `mock_llm`: Mock language model
- `mock_reranker`: Mock reranker
- `sample_chat_history`: Sample chat history
- `temp_pdf_dir`: Temporary PDF directory
- `temp_vectorstore_dir`: Temporary vectorstore directory

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_<function_name>`
2. Use descriptive docstrings
3. Mock external dependencies (LLMs, vectorstores, etc.)
4. Test both success and error cases
5. Aim for high coverage of business logic

## Continuous Integration

Tests should pass before committing. Run:

```bash
pytest --cov=. --cov-fail-under=80
```

