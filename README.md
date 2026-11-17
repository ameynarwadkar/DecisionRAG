# DecisionRAG

A standalone FastAPI + async SQLAlchemy backend for **retriever-free document navigation** using non-generative decision models — **zero passage embeddings** required.

> **Inspiration:** Inspired by *"Retriever-Free Retrieval: Navigating Documents with Decision Models"* (Wan et al., `arXiv:2610.35868`). This is an independent implementation exploring hierarchical document navigation with decision-model backends and is not affiliated with the paper's authors.

## How It Works

1. **Structure-Only Ingestion:** Parses `.pdf`, `.html`, `.md`, and `.txt` documents into a lightweight relational hierarchy (`Category / TopicCluster -> Page -> SectionHeading -> Passage`) in strict reading order without computing or storing passage vectors.
2. **Hierarchical Decision Navigation:** Navigates top-down across categories/clusters, pages, section headings (for long pages), and shared-context passages using parallel binary relevance judgments, followed by optional reading-order multiple-choice calibration.
3. **Conversational Notepad:** Tracks recently visited pages across multi-turn chat sessions so follow-up questions keep context across turns.
4. **Grounded Synthesis:** Generates answers greedily (`temperature=0.0`) from the top-ranked passages with full per-stage latency, token, and request telemetry.

## Supported Decision Model Backends

| Judge ID | Backend | Endpoint |
| :--- | :--- | :--- |
| `clef` | Colab Cloudflare Clef-Flash 9B | `POST {CLEF_API_URL}/v1/systemone` |
| `openrouter_jev` | OpenRouter (`typesafe/jev-1.13`) | `POST https://openrouter.ai/api/alpha/decisions` |
| `openrouter_luna` | OpenRouter (`openai/gpt-6-luna-decisions`) | `POST https://openrouter.ai/api/alpha/decisions` |
| `openai_luna` | OpenAI Direct (`gpt-6-luna`) | `POST https://api.openai.com/v1/decisions` |

## Quickstart

1. **Install backend dependencies:**
   ```powershell
   uv sync
   ```

2. **Configure environment (optional — you can also enter & test keys directly in the Web UI!):**
   ```powershell
   Copy-Item .env.example .env
   ```

3. **Start Dec-RAG (API + Web Playground):**
   ```powershell
   uv run dec-rag
   ```
   - **Web Playground UI:** Open `http://localhost:8000/` in your browser to configure/test decision judges, create namespaces, drag-and-drop `.pdf`/`.md`/`.html`/`.txt` files, chat with live Session Notepad tracking, and inspect stage-by-stage decision probabilities.
   - **Interactive OpenAPI Docs:** `http://localhost:8000/docs`

### Frontend Development (`frontend/` — React + TypeScript + Vite)

Pre-built static assets are already bundled in `src/dec_rag/static/` so `uv run dec-rag` works out of the box without Node.js. If you want to customize the TypeScript UI:

```powershell
cd frontend
npm install
npm run dev      # HMR dev server on http://localhost:5173 (proxies /api -> :8000)
npm run build    # Type-checks (tsc) and rebuilds into ../src/dec_rag/static
```

## Core API Workflow (`/api/v1`)

1. **Check Health & Configured Judges:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. **Create a Namespace:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces \
     -H "Content-Type: application/json" \
     -d '{"name": "Product Help Center", "default_profile": "auto"}'
   ```

3. **Upload Structured Documents (`.pdf`, `.html`, `.md`, `.txt`):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces/<NS_ID>/documents \
     -F "file=@guide.md" \
     -F "category=Billing"
   ```

4. **Retriever-Free Passage Navigation (`/retrieve`):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces/<NS_ID>/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the 30-day refund policy?", "judge": "openrouter_jev", "profile": "auto"}'
   ```

5. **Single-Turn Grounded QA (`/query`):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces/<NS_ID>/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the 30-day refund policy?", "judge": "openrouter_jev"}'
   ```

6. **Multi-Turn Chat with Session Notepad (`/chat`):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces/<NS_ID>/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Can I also download PDF invoices?", "session_id": "<OPTIONAL_SESSION_UUID>"}'
   ```

7. **Side-by-Side Judge Comparison (`/compare-judges`):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/namespaces/<NS_ID>/compare-judges \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the 30-day refund policy?", "judges": ["clef", "openrouter_jev", "openrouter_luna", "openai_luna"]}'
   ```

## Navigation Profiles (`profile`)

- `auto` — Automatically adapts routing (categories, clusters, or direct page pass), token batching, Top-50 joint rescoring, `>60`-passage heading screening, and `mc_mode="order"`.
- `cqa` — Single-request flat page screening, `k_pages=3`, `mc_mode="order"`.
- `md2d` — Category routing (`p<0.8` runner-up + Notepad retention), `k_pages=10`, `mc_mode="order"`.
- `mtrag` — Parallel `<=10k`-token page batches + pooled Top-50 joint rescore + `>60`-passage Top-8 heading screening + `mc_mode="top1"`.
- `nq` — Top-15% cluster pruning + `k_pages=30` tiered passage pool (`all` of 1–3, `10` of 4–10, `3` of 11–30) + `p_para * p_page` + `m_shortlist=5`.

## Running Tests

```powershell
uv run pytest -v
```