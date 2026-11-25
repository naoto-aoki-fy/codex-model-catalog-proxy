# Codex model catalog proxy

This local HTTP server converts LiteLLM's `/model/info` response to a Codex model
catalog. It uses the Python 3.9 standard library.

## Start the server

```sh
python3 codex_catalog_proxy.py
```

Both `http://127.0.0.1:8000/models` and `http://127.0.0.1:8000/models.json` fetch a fresh
catalog on each GET request.

| Environment variable | Purpose |
| --- | --- |
| `UPSTREAM_MODEL_INFO_URL` | Full LiteLLM model info URL. Defaults to `http://127.0.0.1:4000/model/info`. |
| `UPSTREAM_API_KEY` | Optional Bearer token for the upstream model info request. |

## Configure Codex

In `~/.codex/config.toml`, set `base_url` to your Responses-compatible LiteLLM
gateway and `model_catalog_url` to this server. Set `model` to the public
`model_name` in LiteLLM and provide `LITELLM_API_KEY` to Codex.

```toml
model = "your-public-model-name"
model_provider = "litellm"

[model_providers.litellm]
name = "LiteLLM"
base_url = "http://127.0.0.1:4000/v1"
wire_api = "responses"
env_key = "LITELLM_API_KEY"
model_catalog_url = "http://127.0.0.1:8000/models"

[features]
api_key_model_discovery = true
```

An effective `model_catalog_json` setting takes precedence over remote discovery.
See the [Codex gateway documentation](https://learn.chatgpt.com/docs/enterprise/roll-out-a-gateway#implement-the-catalog-endpoint).

## Run the tests

```sh
python3 -B -m unittest -v
```
