# LLM Chatbot with Key Failover and Response Caching

React chat UI → FastAPI backend → LiteLLM (key pool + built-in cache) → LLM provider, with session history in SQLite.

```
frontend (React, :3000) ──HTTP──▶ backend (FastAPI, :8000)
                                   ├─ ChatService: history from SQLite + system prompt + new message
                                   ├─ LLMGateway: LiteLLM cache lookup → key_1 → key_2 → … → 503
                                   └─ SQLite (/data/app.db on a Docker volume)
```

## 1. Live URLs

| Tier | URL |
| --- | --- |
| Frontend | `https://<your-frontend>.vercel.app` |
| Backend | `https://<your-backend>.onrender.com` (try `/health`) |
| Repository | `https://github.com/<you>/<repo>` |
| Demo video | `<link>` |

## 2. Run locally

### With Docker

```bash
cp .env.example .env      # then put two real keys in LLM_API_KEYS
make up                   # frontend http://localhost:3000, backend http://localhost:8000
make down                 # stop (history stays in the app-data volume)
```

`make test-docker` runs the backend tests inside the build image. `make clean` also deletes the volume.

### Without Docker

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" DATABASE_URL="sqlite:///./data/app.db" \
       ALLOWED_ORIGINS="http://localhost:3000" ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:create_app --factory --reload --port 8000
python -m pytest -q        # tests need no network and no real keys

# frontend (second terminal)
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

### Configuration

All settings are environment variables; `.env.example` lists every one.

| Variable | Meaning |
| --- | --- |
| `LLM_API_KEYS` | Comma-separated key pool (≥ 2). Ids are `key_1`, `key_2`, … in this order |
| `LLM_MODEL` | LiteLLM model name, default `gemini/gemini-3.5-flash`. Comma-separate to pair a model with each key |
| `DATABASE_URL` | SQLite location, e.g. `sqlite:////data/app.db` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (the deployed frontend URL) |
| `ENABLE_FAILURE_SIMULATION` | `true` enables the `X-Simulate-Failure` hook |
| `CACHE_ENABLED`, `CACHE_TYPE` (`local`/`disk`), `CACHE_DIR`, `CACHE_TTL_SECONDS` | LiteLLM cache settings |
| `DEFAULT_SYSTEM_PROMPT`, `LLM_TIMEOUT_SECONDS`, `MAX_HISTORY_MESSAGES`, `LLM_MAX_OUTPUT_TOKENS` | Optional tuning |
| `FRONTEND_API_BASE_URL` (Docker) / `VITE_API_BASE_URL` (Vercel, dev) | Backend URL the browser calls |

### Deploying

- **Backend on Render:** New → Blueprint → this repo (`render.yaml`). Set `LLM_API_KEYS` and `ALLOWED_ORIGINS=https://<your-frontend>.vercel.app` in the dashboard. Render's free plan has no persistent disk, so history survives restarts only with a paid disk mounted at `/data`; any Docker host with a volume (Railway, Fly.io) works the same way.
- **Frontend on Vercel:** import the repo, set Root Directory to `frontend` and the env var `VITE_API_BASE_URL=https://<your-backend>.onrender.com`, deploy.

## 3. Design decisions

### Key fallback

`backend/app/llm.py` holds the key pool. Every provider call goes through `litellm.acompletion`, with the key passed per call and LiteLLM's own retries set to 0. The gateway tries `key_1`, then `key_2`, and so on:

- Any failure of a key (a 429, a quota error, a 5xx, a timeout, an auth error or an empty reply) is logged with its type and status only, never the message, which can echo credentials. Then the next key is tried. The user never sees the individual failure.
- The first key that answers produces the reply. `key_id` names it and `fallback_attempts` counts the keys that failed before it.
- If every key fails, the API returns `503 {"detail": "all providers unavailable"}` and nothing is written to SQLite.

I wrote the failover loop myself instead of using LiteLLM's `Router`. The contract needs the exact id of the answering key, an exact count of failed keys and per-key failure injection for a single request. A short explicit loop gives all three in a deterministic order and is easy to test. LiteLLM still does the provider calls, maps provider errors and handles caching.

**`X-Simulate-Failure`** (`backend/app/simulation.py`) is parsed only when `ENABLE_FAILURE_SIMULATION=true`. A listed key raises a simulated provider error before any network call, so it goes through exactly the same failover path as a real error. The plan exists only for that one request. A simulated request skips the cache *lookup*, so it always exercises the key pool and returns a real answer. Its answer is still written to the cache. A malformed header returns 422.

### Caching

`litellm.cache` is LiteLLM's built-in `Cache` (in-memory by default, or disk with `CACHE_TYPE=disk`), with one change: `get_cache_key` is replaced by a SHA-256 of **model + the full message list** (system prompt + history + new message). The API key is left out of the cache key on purpose, so an answer produced by `key_2` during a failover is still a hit later.

- A hit is detected from `response._hidden_params["cache_hit"]`. Then `resolution_source` is `LLM_CACHE`, the reply text and token usage are exactly the stored ones, `key_id` is `null` and `fallback_attempts` is `0`.
- The same first message in a new session gives an identical message list, so it is a hit. Any different message (or different history or system prompt) changes the key and goes live.
- A hit makes no network call, so its `latency_ms` is a few ms instead of hundreds.

### SQLite session storage

`backend/app/db.py` creates the schema on startup: the required `messages` table, an index on `(session_id, id)` and a small `sessions` table that stores each session's system prompt. A turn is stored only after the LLM succeeds. The user message and the assistant message (with metadata JSON) are written together in **one transaction**, so a 422 or 503 never leaves anything in the history. On every request the backend rebuilds the context from SQLite (client sends only the new message), keeping the last `MAX_HISTORY_MESSAGES`. A per-session lock keeps two fast messages in the same chat in order. The database file lives on the `app-data` named volume at `/data`, so history survives container restarts.

### Errors

Validation errors use FastAPI's default 422 shape (empty or whitespace-only message, more than 4000 characters after trimming, empty session id or one longer than 128 characters, malformed JSON). Unknown sessions return 404. A catch-all handler turns any unexpected exception into `500 {"detail": "internal server error"}`, never a stack trace. The OpenAPI and docs routes are off, so the API has exactly the three contract endpoints.

## 4. Verification

```bash
API=http://localhost:8000          # or the deployed backend URL

curl -s $API/health
# {"status":"ok","keys_configured":2,"cache_enabled":true}

# Live call, then the same first message in a NEW session -> cache hit
curl -s $API/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"What is the capital of France?"}'
curl -s $API/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-2","message":"What is the capital of France?"}'
# second: "resolution_source":"LLM_CACHE", "key_id":null, same reply, much lower latency_ms

# Multi-turn memory + history
curl -s $API/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-1","message":"And what is its population?"}'
curl -s $API/sessions/demo-1/messages

# Failover
curl -s $API/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"demo-3","message":"Say hi"}'           # 200, key_id=key_2, fallback_attempts=1
curl -s $API/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=500' \
  -d '{"session_id":"demo-3","message":"Say hi again"}'     # 200, key_id=key_2
curl -s -i $API/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"demo-4","message":"Say hi"}'           # 503 all providers unavailable
curl -s -i $API/sessions/demo-4/messages                    # 404: nothing was stored

# Validation
curl -s -i $API/chat -H 'Content-Type: application/json' -d '{"session_id":"x","message":"   "}'  # 422
```

**In the UI:** every assistant reply shows `source`, `model`, `latency`, `in tokens` and `out tokens` for that request only, plus the answering key. Open *Session settings* to set a system prompt or pick a failover preset. The preset adds `X-Simulate-Failure` to the next messages, so failover and the 503 message can be shown from the deployed frontend. Reloading the page starts a new session, so sending the same first message again shows `LLM_CACHE`.

**Tests:** `cd backend && python -m pytest -q`. They cover the API contract, 422 and 503 cases, failover on real and simulated 429/500/502/503 errors, cache hits and misses, per-request tokens, multi-turn memory, session isolation, history surviving a reopen and the no-history-on-failure rule. A fake LLM is used, so no keys are needed. `test_litellm_integration.py` runs the real LiteLLM cache with `mock_response`.

## 5. Setup Used

- Editor: VS Code
- AI tools: Claude Code (Anthropic) in VS Code to scaffold the code, tests and docs, and to debug a LiteLLM version issue (1.110.x broke cache hits, so `litellm` is pinned to 1.100.0). Every part was reviewed and run locally.

## 6. Bonus features

- **Failover demo in the UI:** a preset selector sends `X-Simulate-Failure` from the browser, so failover can be shown on the live site without curl.
- **Per-session system prompt** (`system-prompt-input`). It is stored in SQLite, so later turns in the same session keep it.
- **Extra metadata shown**: the answering `key_id` and a failover count badge when `fallback_attempts > 0`.
- **Model per key**: `LLM_MODEL` can list one model per key, so `key_2` can be a different model or provider (e.g. `gemini/gemini-3.5-flash,groq/openai/gpt-oss-120b`).
- **Disk cache option** (`CACHE_TYPE=disk`) so cached answers survive restarts. The cache lives on the same `/data` volume.
- **Robustness:** atomic turn writes, per-session ordering lock, non-root runtime image, Docker `HEALTHCHECK`, and a test build stage (`make test-docker`).
