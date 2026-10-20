# Web search tests

From the repository root, install the lightweight test dependencies in a clean
virtual environment and run:

```bash
python -m pip install -r tests/requirements.txt
GRADIO_ANALYTICS_ENABLED=False python -m unittest discover -s tests -v
```

These tests run the application's search callback, MCP HTTP encoding and the real
LangChain retrieval/prompt/result path. Model and GPU imports, model inference,
embeddings and the local vector store are replaced with deterministic fixtures;
no model weights, API keys or external search requests are needed.

Coverage includes provider selection, disabled web access, conversation identity,
source URLs, request headers, JSON/SSE responses, errors, timeouts and cancellation
propagation. It does not test GPU inference or add a UI stop button.
