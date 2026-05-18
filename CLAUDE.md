# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Paper Tutor — a Flask web app that pairs a PDF.js viewer with an LLM tutor backed by **host-side Ollama**. Korean-Mac PoC running on a single user's machine; not multi-tenant, not internet-exposed. Single `app.py` (Flask), single `templates/index.html` (vanilla JS, no build step).

## Run / iterate

Container is the normal path. Ollama runs on the **host** (macOS) and is reached via `host.docker.internal:11434` from the container.

```bash
docker compose up -d --build       # first time
docker compose up -d               # subsequent
docker compose restart paper-tutor # after app.py change (gunicorn does not auto-reload)
docker compose logs -f paper-tutor # streaming logs
```

Bind mounts in `docker-compose.yml` are load-bearing for iteration speed:
- `./templates:/app/templates:ro` — HTML/JS edits show on **browser refresh** alone
- `./app.py:/app/app.py:ro` — Python edits need only `docker compose restart paper-tutor`
- Rebuild (`--build`) is only required when `requirements.txt` or `Dockerfile` change

Access via Caddy (`http://localhost` or `http://paper.test` if `/etc/hosts` entry exists) or direct Flask (`http://localhost:8181`). Caddy is only there for SSE-friendly reverse proxy + clean URL; both paths work.

### Smoke-testing endpoints

There's no test suite. Validate with curl + the `/debug` endpoint:

```bash
# Quick health
curl -s http://localhost/api/models                 # via Caddy
curl -s http://localhost/api/library/tree
# Per-paper debug: shows exact system prompt sent, token counts, loaded Ollama model + its context
curl -s http://localhost/api/papers/<paper_id>/debug | python3 -m json.tool
# Python syntax check before restart
python3 -c "import ast; ast.parse(open('app.py').read())"
```

The debug panel UI (top-right `디버그` button) wraps the same `/debug` endpoint — use it when answering "is the prompt actually being sent" or "does the paper fit in context" questions.

### Running without Docker

```bash
pip install -r requirements.txt
python3 app.py        # uses OLLAMA_BASE_URL=http://localhost:11434
```

## Architecture

### Storage = filesystem (no DB)

The `data/library/` tree IS the library. Each paper is a directory of fixed-named files; folders in the UI map 1:1 to real subdirectories.

```
data/
  library/<folder...>/<paper_id>/
    original.pdf      # served raw to PDF.js
    content.md        # pymupdf4llm markdown — what the LLM sees
    notes.md          # user notes + AI summary destination
    meta.json         # title, original_filename, uploaded_at, char_count, approx_tokens
    chat.json         # {messages: [...], checkpoints: [...]}
  vocabulary.json     # global dictionary for English-study word list
```

Implications: `paper_id` (12-char sha1 prefix) is unique; `find_paper_dir(id)` does an `rglob`. Folders, papers, chats are all just files — there is no migration story, no schema, just JSON files. Anything saved to `./data` survives `docker compose down`.

### Three LLM routing roles

Different work uses different models — this is intentional, not legacy.

| Role | Env var | Default | Why |
|---|---|---|---|
| Tutor chat | `DEFAULT_MODEL` | `gemma4:e4b` | Selected by user in dropdown |
| Auto-overview (after upload) | `OVERVIEW_MODEL` | `gemma4:e4b` | Routed by backend when `fast_overview=true` in chat request — avoids waiting on a heavy model for the first response after upload |
| Drag-to-translate | `EXPLAIN_WORD_MODEL` / `EXPLAIN_SENTENCE_MODEL` | `gemma3:4b` | Must feel instant; uses small `num_ctx=4096` regardless of `NUM_CTX` |

The frontend `state.selectedModel` is **deliberately not sent** to `/api/papers/<id>/explain` so the small model wins.

### Prompt assembly

Defined as module-level strings in `app.py` (`TUTOR_SYSTEM_PROMPT`, `OVERVIEW_PROMPT`, `EXPLAIN_WORD_PROMPT`, `EXPLAIN_SENTENCE_PROMPT`, `SUMMARY_NOTES_PROMPT`). Structure is deliberate:

- **Rules first, paper last.** Huge papers in the middle of a prompt cause "lost in the middle" attention dilution; both anchors (top + bottom of the system message) carry the rules.
- The full `content.md` is re-sent in the system message **every single chat turn** (`api_chat_stream`). The chat-history grows but the system header is identical, so Ollama's prefix cache kicks in from turn 2 onward.
- `OVERVIEW_PROMPT` forces Korean output regardless of paper language; `TUTOR_SYSTEM_PROMPT` matches the user's language. The `EXPLAIN_*` prompts have a strict 3-section format (`📖 일반 뜻 / 📄 이 논문에서 / ✍️ 영어 예문`) — the frontend parses this format and the vocabulary feature depends on it.

### Ollama context cap gotcha

Per-request `num_ctx` in `OLLAMA_OPTIONS` is the **requested** context, but the model also has a native max (qwen3:32b = 40,960 without YaRN; gemma4 supports more). If you change `NUM_CTX`, verify with `/api/ps`:

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool
# context_length must match — if it's e.g. 2048, the server is capping
```

If the loaded `context_length` is below what we send, the system prompt gets silently truncated and the model appears to ignore its rules. Symptoms map to "LLM not respecting prompt" reports. Fix is either to restart Ollama with `OLLAMA_CONTEXT_LENGTH` set higher, or to drop our `NUM_CTX` to match the cap. The `/debug` endpoint surfaces both numbers side by side.

### Thinking mode

`THINK_MODE` env var controls Ollama's top-level `think` field per request. Default `false` because qwen3/r1/gpt-oss add 10–40s of pre-content latency when thinking is on, killing the streaming feel. Re-enable with `THINK_MODE=true|low|medium|high` only for tasks where reasoning quality matters more than first-token latency.

### Streaming + Caddy

Chat uses SSE (`/api/papers/<id>/chat/stream`). Caddy's `flush_interval -1` in `Caddyfile` is required — without it, the proxy buffers SSE and the UI shows nothing until generation completes. If you add new streaming routes, run them through Caddy in the smoke test, not just Flask direct.

### Why Ollama can't be containerized here

**Apple Silicon Metal GPU is not exposed inside Docker Desktop containers.** Putting `ollama/ollama` in `docker-compose` works but falls back to CPU — qwen3:32b moves from ~30 tok/s to ~1 tok/s, effectively unusable. The host-runs-Ollama, container-runs-Flask split is deliberate. Don't suggest moving Ollama into compose without flagging this.

## Frontend conventions

- No build, no framework — `templates/index.html` is the only frontend file, vanilla JS + `marked` + `katex` + `pdfjs-dist` via CDN.
- Three resizable panes: explorer / PDF viewer / chat+notes. Widths persist in `localStorage` (`paper-tutor:left-w`, `paper-tutor:right-w`); double-click a resizer to reset.
- PDF zoom: `Cmd/Ctrl + wheel` and `Cmd/Ctrl + drag` (capture-phase mousedown so it doesn't fight text selection).
- Selection popup over the PDF uses `backdrop-filter: blur` (frosted glass) — when adding new floating UI over the viewer, follow that pattern, not solid backgrounds.

## When asked about "registered prompts" or "why isn't the LLM following X"

1. Open the `디버그` modal in the UI (or hit `/api/papers/<id>/debug`).
2. Check `loaded_models[].context_length` vs `configured_num_ctx`. If the loaded value is smaller, the prompt is being silently cut. This is the most common root cause.
3. Check `think_mode`. If `true` on a thinking model, output starts slow — looks like "no response" to users.
4. Inspect the `system_prompt` field directly. Confirm rules are at both top and bottom, paper in the middle.
