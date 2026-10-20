"""Run with python -m unittest discover -s tests -v.

The model weights, embedding model and GPU are replaced at their boundaries.
Application callbacks, LangChain prompt/retrieval chains and MCP encoding run normally.
"""

import asyncio
import copy
from email.message import Message
import io
import json
import runpy
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from uuid import UUID


# These optional heavyweight imports do not participate in the search callback.
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
chatllm = types.ModuleType("chatllm")
chatllm.ChatLLM = object
model_modules = {"torch": torch, "chatllm": chatllm,
                 "sentence_transformers": types.ModuleType("sentence_transformers")}
saved_modules = {name: sys.modules.get(name) for name in model_modules}
try:
    sys.modules.update(model_modules)
    import app
finally:
    for name, saved in saved_modules.items():
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved

from langchain.llms.base import LLM
from langchain.schema import BaseRetriever, Document
import parallel_search


class ScriptedLLM(LLM):
    temperature: float = 0.01
    top_p: float = 0.1
    history: list = []
    prompts: list = []

    @property
    def _llm_type(self):
        return "scripted-test"

    def _call(self, prompt, stop=None, **kwargs):
        self.prompts.append(prompt)
        return "A scripted answer using the supplied sources."


class LocalRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Local knowledge fixture.")]


class LocalVectorStore:
    def as_retriever(self, **kwargs):
        return LocalRetriever()


class Reply(io.BytesIO):
    def __init__(self, payload, sse=False, session=None):
        if sse:
            body = ": heartbeat\r\n\r\ndata: {}\r\n\r\n".format(json.dumps(payload))
        else:
            body = json.dumps(payload)
        super().__init__(body.encode())
        self.headers = Message()
        self.headers["Content-Type"] = "text/event-stream" if sse else "application/json"
        if session:
            self.headers["Mcp-Session-Id"] = session


class MCPServer:
    """Observe real urllib Requests rather than discarding transport arguments."""

    def __init__(self, sse=False, failure=None, tool_error=False, text_only=False):
        self.requests = []
        self.sse = sse
        self.failure = failure
        self.tool_error = tool_error
        self.text_only = text_only
        self.url = "https://example.org/source?q=climate&lang=zh"
        self.excerpt = "Source excerpt with literal {braces}."

    def open(self, request, timeout):
        message = json.loads(request.data)
        self.requests.append((request, timeout, message))
        method = message["method"]
        if method == "initialize":
            result = {"protocolVersion": parallel_search.PROTOCOL_VERSION,
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "notifications/initialized":
            return Reply({})
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}]}
        elif method == "tools/call":
            if self.failure:
                raise self.failure
            payload = {"results": [{"title": "Example source", "url": self.url,
                                     "excerpts": [self.excerpt]}]}
            result = {"isError": self.tool_error,
                      "content": [{"type": "text", "text": json.dumps(payload)}]}
            if not self.text_only:
                result["structuredContent"] = payload
        else:
            raise AssertionError(method)
        return Reply({"jsonrpc": "2.0", "id": message["id"], "result": result},
                     sse=self.sse, session="transport-only")

    @property
    def calls(self):
        return [m["params"]["arguments"] for _, _, m in self.requests
                if m["method"] == "tools/call"]


class CallerTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = app.KnowledgeBasedChatLLM()
        self.knowledge.llm = ScriptedLLM()
        self.knowledge.embeddings = object()
        self.patches = [patch.object(app, "knowladge_based_chat_llm", self.knowledge),
                        patch.object(app.FAISS, "load_local", return_value=LocalVectorStore())]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def predict(self, context, query="Current climate research", provider="Parallel", enabled="True",
                history=None):
        return app.predict(query, enabled, 6, 3, 0.01, 0.1, history,
                           provider, context)

    def test_parallel_reaches_real_retrieval_prompt_and_answer(self):
        server = MCPServer()
        context = {}
        with patch("parallel_search.build_opener", return_value=server), \
                patch.dict("os.environ", {"PARALLEL_API_KEY": "must-not-be-used"}), \
                patch.object(app, "ddg") as ddg:
            out = self.predict(context)
            ddg.assert_not_called()
        self.assertEqual(out[1][-1][1], "A scripted answer using the supplied sources.")
        prompt = self.knowledge.llm.prompts[-1]
        self.assertIn(server.excerpt, prompt)
        self.assertIn(server.url, prompt)
        self.assertIn("Local knowledge fixture.", prompt)
        self.assertIn("Current climate research", prompt)
        self.assertEqual(str(UUID(context["session_id"])), context["session_id"])
        for req, timeout, message in server.requests:
            self.assertEqual(req.full_url, "https://search.parallel.ai/mcp")
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.get_header("User-agent"), parallel_search.USER_AGENT)
            self.assertIsNone(req.get_header("Authorization"))
            self.assertEqual(timeout, parallel_search.TIMEOUT)
            if message["method"] != "initialize":
                self.assertEqual(req.get_header("Mcp-session-id"), "transport-only")
        self.assertEqual(server.calls[0]["objective"], "Current climate research")
        self.assertEqual(server.calls[0]["search_queries"], ["Current climate research"])
        self.assertEqual(server.calls[0]["session_id"], context["session_id"])

    def test_default_duckduckgo_and_disabled_permission(self):
        with patch.object(app, "ddg", return_value=[{"body": "Original DDG result"}]) as ddg, \
                patch("parallel_search.build_opener") as opener:
            # Old callers still choose DuckDuckGo with no extra arguments.
            self.assertEqual(app.search_web("saved choice"), "Original DDG result")
            self.predict({}, provider="DuckDuckGo")
            self.assertEqual(ddg.call_count, 2)
            self.assertEqual(app.SESSION.proxies["https"], "socks5h://localhost:7890")
            for provider in ("DuckDuckGo", "Parallel", "invalid"):
                context = {}
                self.predict(context, provider=provider, enabled="False")
                self.assertEqual(context, {})
            self.assertEqual(ddg.call_count, 2)
            opener.assert_not_called()

    def test_conversation_followup_interleaving_reset_and_retry(self):
        server = MCPServer()
        first, second = {}, {}
        with patch("parallel_search.build_opener", return_value=server):
            out = self.predict(first, "First question")
            self.predict(second, "Independent question")
            self.predict(first, "Follow-up", history=out[2])
            server.tool_error = True
            failed = self.predict(first, "Quota or tool failure", history=out[2])
            self.assertEqual(failed[0], "Quota or tool failure")
            server.tool_error = False
            self.predict(first, "Retry", history=out[2])
            cleared = app.clear_session()
            self.assertEqual(cleared, ("", None, {}))
            self.predict(cleared[2], "New chat")
        ids = [call["session_id"] for call in server.calls]
        self.assertEqual(ids[0], ids[2])
        self.assertEqual(ids[0], ids[3])
        self.assertEqual(ids[0], ids[4])
        self.assertNotEqual(ids[0], ids[1])
        self.assertNotIn(ids[5], ids[:5])

    def test_concurrent_chats_keep_separate_context(self):
        server = MCPServer()
        contexts = [{}, {}]
        with patch("parallel_search.build_opener", return_value=server):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda c: self.predict(c), contexts))
        self.assertNotEqual(contexts[0]["session_id"], contexts[1]["session_id"])
        self.assertEqual({c["session_id"] for c in server.calls},
                         {c["session_id"] for c in contexts})

    def test_protocol_errors_timeouts_and_no_fallback(self):
        for failure in (TimeoutError(), ConnectionResetError(), URLError("offline"),
                        HTTPError(parallel_search.MCP_URL, 429, "quota", {}, None),
                        HTTPError(parallel_search.MCP_URL, 401, "unauthorized", {}, None)):
            with self.subTest(failure=failure):
                server = MCPServer(failure=failure)
                context = {}
                history = [("old", "answer")]
                before = copy.deepcopy(history)
                with patch("parallel_search.build_opener", return_value=server), \
                        patch.object(app, "ddg") as ddg:
                    out = self.predict(context, history=history)
                self.assertEqual(out[0], "Current climate research")
                self.assertIn("Parallel", out[1][-1][1])
                self.assertEqual(out[2], before)
                self.assertEqual(history, before)
                ddg.assert_not_called()
                self.assertEqual(server.calls[0]["session_id"], context["session_id"])

    def test_cancellation_propagates_without_fallback(self):
        server = MCPServer(failure=asyncio.CancelledError())
        context = {}
        with patch("parallel_search.build_opener", return_value=server), \
                patch.object(app, "ddg") as ddg:
            with self.assertRaises(asyncio.CancelledError):
                self.predict(context)
            ddg.assert_not_called()
        self.assertIn("session_id", context)

    def test_sse_and_text_content(self):
        server = MCPServer(sse=True, text_only=True)
        with patch("parallel_search.build_opener", return_value=server):
            self.predict({})
        self.assertIn(server.url, self.knowledge.llm.prompts[-1])

    def test_invalid_urls_and_malformed_rpc_are_visible(self):
        server = MCPServer()
        server.url = "javascript:alert(1)"
        with patch("parallel_search.build_opener", return_value=server):
            out = self.predict({})
            self.assertIn("invalid source", out[1][-1][1])
        client = parallel_search._Client()
        for payload in ({"jsonrpc": "2.0", "id": 999, "result": {}},
                        {"jsonrpc": "2.0", "id": 1, "error": {"code": -1}}, []):
            with self.subTest(payload=payload), \
                    patch.object(client.opener, "open", return_value=Reply(payload)):
                client.request_id = 0
                with self.assertRaises(parallel_search.ParallelSearchError):
                    client.request("initialize")

    def test_real_gradio_callbacks_and_per_session_state(self):
        from gradio.state_holder import SessionState

        # Execute the actual UI wiring, stopping only model initialization and
        # the launch network boundary. A missing embedding fixture is caught by
        # the existing init_model error handler; no model download is attempted.
        with patch.dict(sys.modules, model_modules), \
                patch.object(app.gr.Blocks, "launch", return_value=None):
            namespace = runpy.run_path(app.__file__, run_name="__main__")
        demo = namespace["demo"]
        self.addCleanup(demo.close)
        namespace["predict"].__globals__["knowladge_based_chat_llm"] = self.knowledge
        self.assertEqual(namespace["use_web"].value, "False")
        self.assertEqual(namespace["search_provider"].value, "DuckDuckGo")
        callbacks = [i for i, fn in enumerate(demo.fns) if fn.fn.__name__ == "predict"]
        self.assertEqual(len(callbacks), 2)  # Send button and Enter.
        clear = next(i for i, fn in enumerate(demo.fns) if fn.fn.__name__ == "clear_session")
        context_id = namespace["search_context"]._id
        first, second = SessionState(demo), SessionState(demo)
        server = MCPServer()

        async def send(index, state):
            return await demo.process_api(index, ["UI question", "True", 6, 3, 0.01,
                                                  0.1, None, "Parallel", None], state=state)

        with patch("parallel_search.build_opener", return_value=server):
            response = asyncio.run(send(callbacks[0], first))
            self.assertIn("scripted answer", response["data"][1][-1][1])
            first_id = first[context_id]["session_id"]
            asyncio.run(send(callbacks[1], second))
            self.assertNotEqual(first_id, second[context_id]["session_id"])
            asyncio.run(send(callbacks[1], first))
            self.assertEqual(first_id, first[context_id]["session_id"])
            asyncio.run(demo.process_api(clear, [], state=first))
            self.assertEqual(first[context_id], {})
            asyncio.run(send(callbacks[0], first))
            self.assertNotEqual(first_id, first[context_id]["session_id"])


if __name__ == "__main__":
    unittest.main()
