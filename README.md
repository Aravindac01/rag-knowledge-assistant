## Results

On a 5-question evaluation set covering resume-specific facts (employer name, a metric, a technology, education, tooling), the pipeline achieved:

- **Accuracy: 100% (5/5)**
- **Average latency: 3.81s**

Run `python eval/run_eval.py` yourself against `eval/eval_questions.json` to reproduce.

## Web UI

A single-page animated frontend is served at the root URL (`/`) — drag-and-drop document upload, chat-style Q&A, and clickable source citations. No separate frontend build step; FastAPI serves it directly from `static/index.html`.