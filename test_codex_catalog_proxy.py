import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from codex_catalog_proxy import CatalogServer, convert_models


def deployment(name="public-model", **info):
    return {
        "model_name": name,
        "model_info": {"mode": "chat", "id": "internal-id", **info},
        "litellm_params": {"model": "provider/internal-name", "api_key": "internal-key"},
    }


def models(*deployments):
    return convert_models({"data": list(deployments)})["models"]


def efforts(model):
    return [level["effort"] for level in model["supported_reasoning_levels"]]


class ConversionTests(unittest.TestCase):
    def test_explicit_reasoning_list_takes_priority(self):
        model, = models(deployment(
            reasoning_effort_levels=["high", "low", "low", "invalid", {"effort": "max"}],
            supports_low_reasoning_effort=False,
            supports_xhigh_reasoning_effort=True,
            default_reasoning_effort="high",
        ))
        self.assertEqual(efforts(model), ["low", "high"])
        self.assertTrue(all(level["description"] for level in model["supported_reasoning_levels"]))
        self.assertEqual(model["default_reasoning_level"], "high")

    def test_empty_list_and_disabled_reasoning(self):
        for metadata in (
            {"reasoning_effort_levels": []},
            {"reasoning_effort_levels": ["low"], "supports_reasoning": False},
            {"supports_reasoning": False},
        ):
            with self.subTest(metadata=metadata):
                model, = models(deployment(
                    supports_low_reasoning_effort=True, default_reasoning_effort="low", **metadata
                ))
                self.assertEqual(efforts(model), [])
                self.assertNotIn("default_reasoning_level", model)

    def test_only_true_flags_supply_fallback_efforts(self):
        for metadata in ({}, {"reasoning_effort_levels": None}):
            with self.subTest(metadata=metadata):
                model, = models(deployment(
                    supports_none_reasoning_effort=False,
                    supports_low_reasoning_effort=True,
                    supports_medium_reasoning_effort="true",
                    supports_high_reasoning_effort=1,
                    supports_xhigh_reasoning_effort=True,
                    default_reasoning_effort="high",
                    **metadata,
                ))
                self.assertEqual(efforts(model), ["low", "xhigh"])
                self.assertNotIn("default_reasoning_level", model)

    def test_missing_reasoning_effort_information(self):
        model, = models(deployment(
            supports_reasoning=True,
            supported_openai_params=["reasoning_effort"],
            default_reasoning_effort="medium",
        ))
        self.assertEqual(efforts(model), [])
        self.assertNotIn("default_reasoning_level", model)

    def test_context_window_uses_only_positive_integer_input_limit(self):
        model, = models(deployment(max_input_tokens=8192, max_output_tokens=1024, max_tokens=2048))
        self.assertEqual(model["context_window"], 8192)
        for invalid_limit in (None, 0, -1, True, 8192.0, "8192", 2**63):
            with self.subTest(invalid_limit=invalid_limit):
                model, = models(deployment(
                    max_input_tokens=invalid_limit, max_output_tokens=1024, max_tokens=2048
                ))
                self.assertNotIn("context_window", model)

    def test_input_modalities_prefer_an_explicit_list(self):
        for metadata, expected in (
            ({"supports_vision": True, "supports_audio_input": True}, ["text", "image", "audio"]),
            ({"supports_audio_input": True}, ["text", "audio"]),
            ({"supports_vision": 1, "supports_audio_input": "true"}, ["text"]),
            ({"input_modalities": ["audio", "image", "audio", "video"]}, ["image", "audio"]),
            ({"input_modalities": [], "supports_vision": True}, []),
        ):
            with self.subTest(metadata=metadata):
                model, = models(deployment(**metadata))
                self.assertEqual(model["input_modalities"], expected)

    def test_same_name_deployments_advertise_only_common_capabilities(self):
        first = deployment(
            reasoning_effort_levels=["low", "high"], default_reasoning_effort="low",
            supports_vision=True, supports_audio_input=True, max_input_tokens=8192,
        )
        second = deployment(
            reasoning_effort_levels=["low", "medium"], default_reasoning_effort="low",
            input_modalities=["text", "image"], max_input_tokens=4096, description="Public model",
        )
        model, = models(first, second)
        self.assertEqual(efforts(model), ["low"])
        self.assertEqual(model["default_reasoning_level"], "low")
        self.assertEqual(model["input_modalities"], ["text", "image"])
        self.assertEqual(model["context_window"], 4096)
        self.assertEqual(model["description"], "Public model")

        second["model_info"].update(default_reasoning_effort="medium", max_input_tokens=None)
        model, = models(first, second)
        self.assertEqual(efforts(model), ["low"])
        self.assertNotIn("default_reasoning_level", model)
        self.assertNotIn("context_window", model)

        model, = models(first, deployment())
        self.assertEqual(efforts(model), [])
        self.assertEqual(model["input_modalities"], ["text"])

    def test_requires_conversational_metadata(self):
        catalog = models(
            deployment("chat"),
            deployment("responses", mode="responses"),
            deployment("via-endpoint", mode=None, supported_endpoints=["/v1/responses"]),
            deployment("embedding", mode="embedding"),
            deployment("embedding-conflict", mode="embedding", supported_endpoints=["/responses"]),
            deployment("conflicting-endpoint", supported_endpoints=["/v1/embeddings"]),
            deployment("unknown", mode=None, supports_vision=True),
            deployment("mixed"),
            deployment("mixed", mode="embedding"),
        )
        self.assertEqual([model["slug"] for model in catalog], ["chat", "responses", "via-endpoint"])


class MockLiteLLMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append((self.path, self.headers.get("Authorization")))
        if self.server.stall.is_set():
            self.server.release.wait()
            return
        self.send_response(self.server.response_status)
        self.send_header("Content-Length", str(len(self.server.response_body)))
        self.end_headers()
        self.wfile.write(self.server.response_body)

    def log_message(self, format, *args):
        pass


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockLiteLLMHandler)
        self.upstream.requests = []
        self.upstream.stall = threading.Event()
        self.upstream.release = threading.Event()
        self.run_server(self.upstream)
        self.addCleanup(self.upstream.release.set)
        self.proxy = CatalogServer(
            ("127.0.0.1", 0),
            upstream_url=f"http://127.0.0.1:{self.upstream.server_port}/model/info",
            api_key="upstream-secret",
            upstream_timeout=0.2,
        )
        self.run_server(self.proxy)

    def run_server(self, server):
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)

    def request(self, path="/models", *, body=b"", status=200, expected_status=200):
        self.upstream.response_body = body
        self.upstream.response_status = status
        connection = HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=3)
        try:
            connection.request("GET", path, headers={"Authorization": "Bearer client-secret"})
            with connection.getresponse() as response:
                self.assertEqual(response.status, expected_status)
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                return json.load(response)
        finally:
            connection.close()

    def test_routes_fetch_fresh_catalog_and_send_upstream_bearer_auth(self):
        for route, name in (("/models", "first"), ("/models.json?client_version=0.162.1", "second")):
            with self.subTest(route=route):
                result = self.request(route, body=json.dumps({"data": [deployment(name)]}).encode())
                model, = result["models"]
                self.assertEqual(model["slug"], name)
                self.assertEqual(model["display_name"], name)
                for private_value in (
                    "litellm_params", "provider/internal-name", "internal-id", "internal-key",
                    "upstream-secret", "client-secret",
                ):
                    self.assertNotIn(private_value, json.dumps(result))
        self.assertEqual(self.upstream.requests, [("/model/info", "Bearer upstream-secret")] * 2)

    def test_empty_catalog_without_upstream_auth(self):
        self.proxy.api_key = ""
        self.assertEqual(self.request("/models.json", body=b'{"data": []}'), {"models": []})
        self.assertEqual(self.upstream.requests, [("/model/info", None)])

    def test_unknown_route_does_not_contact_upstream(self):
        self.request("/model/info", expected_status=404)
        self.assertEqual(self.upstream.requests, [])

    def test_upstream_failures_are_generic_bad_gateway_responses(self):
        for status, body in (
            (401, b'{"error": "upstream-secret"}'),
            (403, b'{"error": "upstream-secret"}'),
            (200, b'not JSON upstream-secret'),
            (200, b'{"data": [], "invalid_json": NaN}'),
            (200, b'["upstream-secret"]'),
            (200, b'{"error": "upstream-secret"}'),
            (200, b'{"data": {"error": "upstream-secret"}}'),
            (200, b'{"data": ["upstream-secret"]}'),
            (200, b'{"data": [{"model_info": {"mode": "chat"}}]}'),
            (200, b'{"data": [{"model_name": "public-model", "model_info": "upstream-secret"}]}'),
        ):
            with self.subTest(status=status, body=body):
                self.assertEqual(
                    self.request(body=body, status=status, expected_status=502),
                    {"error": "Upstream model catalog unavailable"},
                )

    def test_upstream_timeout(self):
        self.upstream.stall.set()
        self.assertEqual(
            self.request(expected_status=502),
            {"error": "Upstream model catalog unavailable"},
        )
        self.assertEqual(self.upstream.requests, [("/model/info", "Bearer upstream-secret")])


if __name__ == "__main__":
    unittest.main()
