# RAG Knowledge Assistant

A production-style Retrieval-Augmented Generation API: upload documents, ask
questions, get cited answers. Built with FastAPI, hybrid search (vector +
BM25), Qdrant, and OpenAI — plus a small eval harness so you can quote a real
accuracy number instead of "it works on my machine."

## Architecture

```
Client
  │
  ├─ POST /ingest  → extract text → chunk → embed → upsert to Qdrant
  │                                              └─ append to chunks.json (BM25 corpus)
  │
  └─ POST /query   → hybrid_search()
                         ├─ vector_search()  (Qdrant cosine similarity)
                         ├─ bm25_search()    (keyword search over chunks.json)
                         └─ reciprocal_rank_fusion()  → top-k chunks
                     → generate_answer()  (OpenAI chat completion, cited)
                     → answer + sources + latency + token count
```

## 0. Accounts and tools you need (one-time setup)

If you've never published a project before, do these in order. Each is free
to start.

1. **GitHub account** — [github.com/join](https://github.com/join). This is
   where your code lives and what recruiters/interviewers will look at.
2. **Git installed locally** — check with `git --version` in a terminal. If
   missing: [git-scm.com/downloads](https://git-scm.com/downloads).
3. **Python 3.11+** — check with `python3 --version`.
   [python.org/downloads](https://www.python.org/downloads/).
4. **Docker Desktop** — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
   Lets you run Qdrant locally and test the container before deploying.
5. **A code editor** — VS Code ([code.visualstudio.com](https://code.visualstudio.com/))
   or Cursor ([cursor.com](https://cursor.com)) both work well for this.
6. **OpenAI API key** — [platform.openai.com](https://platform.openai.com/).
   Sign up, add a small amount of billing credit ($5-10 is plenty for this
   project), then create a key under API Keys. This powers both embeddings
   and answer generation.
7. **Qdrant Cloud account** (only needed when you deploy, not for local dev)
   — [cloud.qdrant.io](https://cloud.qdrant.io/). Free tier gives you a
   hosted vector database cluster with a URL + API key.
8. **Railway account** (for deployment) — [railway.app](https://railway.app/).
   Sign in with GitHub. Free tier is enough to get this live.

## 1. Local setup

```bash
cd rag-knowledge-assistant
cp .env.example .env
# open .env and paste your OPENAI_API_KEY

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run it locally

Start Qdrant + the API together:

```bash
docker compose up --build
```

Or run Qdrant in Docker and the API directly (faster iteration while coding):

```bash
docker run -p 6333:6333 qdrant/qdrant
uvicorn app.main:app --reload
```

API docs (auto-generated) are at `http://localhost:8000/docs` — use this to
test `/ingest` and `/query` without writing curl commands.

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/some-document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the remote work policy?"}'
```

## 3. Run the eval

Edit `eval/eval_questions.json` so the questions and `expected_keywords`
actually match whatever document(s) you ingested, then:

```bash
python eval/run_eval.py
```

This prints an accuracy score and average latency, and writes full results
to `eval/eval_results.json`. Put this number in your README and resume bullet
— "achieved X% retrieval accuracy on a Y-question eval set" is exactly the
kind of specific claim that makes interviewers lean in.

## 4. Publish to GitHub (first time)

```bash
git init
git add .
git commit -m "Initial commit: RAG knowledge assistant"
```

Then on github.com: click **New repository**, name it
`rag-knowledge-assistant`, leave it empty (no README/license — you already
have one), and create it. GitHub will show you commands like these — run
them:

```bash
git remote add origin https://github.com/<your-username>/rag-knowledge-assistant.git
git branch -M main
git push -u origin main
```

Your `.env` file is excluded by `.gitignore`, so your API key won't leak.
Double-check before pushing: `git status` should never show `.env`.

## 5. Deploy (Railway)

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io/) —
   copy its URL and API key.
2. On [railway.app](https://railway.app/): **New Project → Deploy from
   GitHub repo** → select `rag-knowledge-assistant`.
3. Railway detects the `Dockerfile` automatically. Under the service's
   **Variables** tab, add: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `COLLECTION_NAME`.
4. Deploy. Railway gives you a public URL — that's your live demo link for
   your resume/LinkedIn/GitHub README.
5. Re-run the ingest + query curl commands against the public URL to confirm
   it works end to end, then take a screenshot or short screen recording for
   your README.

## What to highlight in interviews

- Why hybrid search (BM25 + vector) instead of vector-only, and what
  Reciprocal Rank Fusion does (`app/search.py`).
- Your actual eval accuracy number and what it means when you drop hybrid
  down to vector-only or BM25-only (try it — comment one out and re-run
  `eval/run_eval.py`, then explain the change).
- Where cost/latency shows up (`tokens_used`, `latency_ms` in every
  response) and how you'd reduce it (smaller embedding model, caching
  repeated queries, cheaper chat model for simple questions).
- What happens when the answer isn't in the context (the prompt explicitly
  tells the model to say "I don't know" — this is your hallucination
  guardrail, and interviewers will ask about it).
