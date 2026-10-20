"""Anonymous Parallel Search MCP over Streamable HTTP (no extra dependencies)."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "LangChain-ChatGLM-Webui/0.1 "
    "(+https://github.com/X-D-Lab/LangChain-ChatGLM-Webui)"
)
TIMEOUT = 20
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PROTOCOL_VERSION = "2025-03-26"


class ParallelSearchError(RuntimeError):
    """A search failed; the caller must not silently switch providers."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ParallelSearchError("Parallel Search MCP redirected unexpectedly.")


class _Client:
    """One short-lived MCP connection. Conversation identity is supplied separately."""

    def __init__(self):
        self.opener = build_opener(_NoRedirect())
        self.request_id = 0
        self.transport_session = None

    def request(self, method, params=None, notification=False):
        self.request_id += 1
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            message["id"] = self.request_id
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if method != "initialize":
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        if self.transport_session:
            headers["Mcp-Session-Id"] = self.transport_session
        request = Request(MCP_URL, data=json.dumps(message).encode(), headers=headers)
        try:
            with self.opener.open(request, timeout=TIMEOUT) as response:
                self.transport_session = response.headers.get(
                    "Mcp-Session-Id", self.transport_session
                )
                if notification:
                    return None
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ParallelSearchError("Parallel Search MCP response is too large.")
                text = raw.decode("utf-8")
                content_type = response.headers.get_content_type()
        except HTTPError as exc:
            exc.close()
            raise ParallelSearchError(
                "Parallel search failed (HTTP {}). Please try again later.".format(exc.code)
            ) from exc
        except (OSError, URLError) as exc:
            raise ParallelSearchError(
                "Parallel search could not connect or timed out. Please try again later."
            ) from exc
        except UnicodeError as exc:
            raise ParallelSearchError("Parallel Search MCP returned invalid text.") from exc

        try:
            if content_type == "text/event-stream":
                # Streamable HTTP may return SSE instead of a single JSON object.
                messages = []
                data = []
                for line in text.splitlines() + [""]:
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        messages.append(json.loads("\n".join(data)))
                        data = []
            elif content_type == "application/json":
                messages = [json.loads(text)]
            else:
                raise ValueError("Unexpected content type")
            reply = next(
                m for m in messages
                if isinstance(m, dict) and m.get("id") == message["id"]
            )
            if reply.get("jsonrpc") != "2.0" or "error" in reply:
                raise ValueError("RPC failure")
            result = reply["result"]
            if not isinstance(result, dict):
                raise ValueError("Invalid result")
            return result
        except (ValueError, KeyError, StopIteration) as exc:
            raise ParallelSearchError("Parallel Search MCP returned an invalid response.") from exc

    def search(self, query, session_id):
        initialized = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "LangChain-ChatGLM-Webui", "version": "0.1"},
        })
        if initialized.get("protocolVersion") != PROTOCOL_VERSION:
            raise ParallelSearchError("Parallel Search MCP protocol version is unsupported.")
        self.request("notifications/initialized", notification=True)
        tools = self.request("tools/list")
        advertised = tools.get("tools")
        if not isinstance(advertised, list) or not any(
                   t.get("name") == "web_search" for t in advertised
                   if isinstance(t, dict)):
            raise ParallelSearchError("Parallel Search MCP does not advertise web_search.")
        result = self.request("tools/call", {
            "name": "web_search",
            "arguments": {
                "objective": query,
                "search_queries": [query],
                "session_id": session_id,
            },
        })
        if result.get("isError"):
            raise ParallelSearchError("Parallel search failed. Please try again later.")
        try:
            payload = result.get("structuredContent")
            if payload is None:
                payload = json.loads("\n".join(
                    part["text"] for part in result.get("content", [])
                    if part.get("type") == "text"
                ))
            results = payload["results"]
            if not isinstance(results, list):
                raise ValueError("Invalid results")
            sources = []
            for item in results:
                url = item["url"]
                parsed = urlsplit(url)
                excerpts = item["excerpts"]
                if (parsed.scheme not in ("http", "https") or not parsed.hostname
                        or parsed.username or parsed.password
                        or not isinstance(excerpts, list)
                        or any(not isinstance(e, str) for e in excerpts)):
                    raise ValueError("Invalid source")
                sources.append("{}\n{}\n{}".format(
                    item.get("title") or url, url, "\n".join(excerpts)
                ))
            return "\n\n".join(sources)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ParallelSearchError("Parallel search returned invalid source data.") from exc


def search_parallel(query, session_id):
    """Return source excerpts and URLs for this conversation, without credentials."""
    return _Client().search(query, session_id)
