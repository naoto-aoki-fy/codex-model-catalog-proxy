"""Serve a Codex model catalog from LiteLLM's /model/info endpoint."""

import json
import os
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


DEFAULT_UPSTREAM_URL = "http://127.0.0.1:4000/model/info"
CATALOG_ROUTES = ("/models", "/models.json")
CHAT_MODES = ("chat", "responses")
CHAT_ENDPOINTS = ("/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses")
REASONING_DESCRIPTIONS = {
    "none": "Thinking disabled",
    "minimal": "Minimal reasoning",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra-high reasoning depth",
    "max": "Maximum reasoning depth",
}
MODALITY_FLAGS = {"image": "supports_vision", "audio": "supports_audio_input"}
INPUT_MODALITIES = ("text", *MODALITY_FLAGS)


def supports_chat(info: dict[str, Any]) -> bool:
    mode = info.get("mode")
    if mode is not None and mode not in CHAT_MODES:
        return False
    endpoints = info.get("supported_endpoints")
    if endpoints is None:
        return mode in CHAT_MODES
    return isinstance(endpoints, list) and any(path in endpoints for path in CHAT_ENDPOINTS)


def reasoning_levels(info: dict[str, Any]) -> list[str]:
    if info.get("supports_reasoning") is False:
        return []
    declared = info.get("reasoning_effort_levels")
    if isinstance(declared, list):
        return [level for level in REASONING_DESCRIPTIONS if level in declared]
    return [
        level for level in REASONING_DESCRIPTIONS
        if info.get(f"supports_{level}_reasoning_effort") is True
    ]


def input_modalities(info: dict[str, Any]) -> list[str]:
    declared = info.get("input_modalities")
    if isinstance(declared, list):
        return [modality for modality in INPUT_MODALITIES if modality in declared]
    return ["text", *(kind for kind, flag in MODALITY_FLAGS.items() if info.get(flag) is True)]


def deployment_groups(payload: Any) -> dict[str, list[tuple[dict[str, Any], Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Expected a LiteLLM data array")

    groups = {}
    for deployment in payload["data"]:
        if not isinstance(deployment, dict):
            raise ValueError("Expected a LiteLLM deployment object")
        name = deployment.get("model_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Expected a public model_name")
        info = deployment.get("model_info")
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise ValueError("Expected a model_info object")
        groups.setdefault(name, []).append((info, deployment.get("description")))
    return groups


def convert_models(payload: Any) -> dict[str, Any]:
    """Convert deployment metadata into a catalog keyed by public model names."""
    models = []
    for name, group in deployment_groups(payload).items():
        infos = [info for info, _ in group]
        if not all(supports_chat(info) for info in infos):
            continue

        common_levels = set.intersection(*(set(reasoning_levels(info)) for info in infos))
        common_modalities = set.intersection(*(set(input_modalities(info)) for info in infos))
        default = infos[0].get("default_reasoning_effort")
        limits = [info.get("max_input_tokens") for info in infos]
        description = next(
            (value for info, public_description in group
             for value in (info.get("description"), public_description)
             if isinstance(value, str) and value.strip()),
            None,
        )

        model = {
            "slug": name,
            "display_name": name,
            "description": description,
            "supported_reasoning_levels": [
                {"effort": level, "description": explanation}
                for level, explanation in REASONING_DESCRIPTIONS.items()
                if level in common_levels
            ],
            "input_modalities": [kind for kind in INPUT_MODALITIES if kind in common_modalities],
            "shell_type": "unified_exec",
            "visibility": "list",
            "supported_in_api": True,
            "priority": len(models) + 1,
            "base_instructions": "You are a coding assistant.",
            "supports_reasoning_summary_parameter": False,
            "default_reasoning_summary": "none",
            "support_verbosity": False,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "experimental_supported_tools": [],
        }
        if isinstance(default, str) and default in common_levels and all(
            info.get("default_reasoning_effort") == default for info in infos
        ):
            model["default_reasoning_level"] = default
        if all(type(limit) is int and 0 < limit < 2**63 for limit in limits):
            model["context_window"] = min(limits)
        models.append(model)
    return {"models": models}


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


class CatalogServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        upstream_url: str = DEFAULT_UPSTREAM_URL,
        api_key: str = "",
        upstream_timeout: float = 5,
    ) -> None:
        self.upstream_url = upstream_url
        self.api_key = api_key
        self.upstream_timeout = upstream_timeout
        super().__init__(address, CatalogHandler)

    def fetch_catalog(self) -> dict[str, Any]:
        url = urlsplit(self.upstream_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("Expected an HTTP or HTTPS upstream URL")
        connection_type = HTTPSConnection if url.scheme == "https" else HTTPConnection
        connection = connection_type(url.hostname, url.port, timeout=self.upstream_timeout)
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        target = url.path or "/"
        if url.query:
            target += f"?{url.query}"
        try:
            connection.request("GET", target, headers=headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError("Upstream returned an unsuccessful status")
                payload = json.loads(response.read(), parse_constant=reject_json_constant)
        finally:
            connection.close()
        return convert_models(payload)


class CatalogHandler(BaseHTTPRequestHandler):
    server: CatalogServer

    def do_GET(self) -> None:
        if self.path.partition("?")[0] not in CATALOG_ROUTES:
            self.send_json(404, b'{"error": "Not found"}')
            return
        try:
            catalog = self.server.fetch_catalog()
            body = json.dumps(catalog, ensure_ascii=False).encode("utf-8")
        except (HTTPException, OSError, ValueError, RecursionError):
            self.send_json(502, b'{"error": "Upstream model catalog unavailable"}')
            return
        self.send_json(200, body)

    def send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    with CatalogServer(
        ("127.0.0.1", 8000),
        upstream_url=os.getenv("UPSTREAM_MODEL_INFO_URL", DEFAULT_UPSTREAM_URL),
        api_key=os.getenv("UPSTREAM_API_KEY", ""),
    ) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
