# Mercy Protocol

> A local-first, multimodal personal AI assistant with a holographic HUD, an interactive Live2D avatar, a LangGraph agent engine, and real-time backend telemetry.

![Main Console](docs/screenshots/1_main_dashboard.png)

## Features

- **LangGraph agent** — multi-turn reasoning, autonomous tool calls, exponential-backoff retries
- **Hybrid memory** — SQLite checkpointer (short-term) + ChromaDB vector store (long-term RAG)
- **Multimodal vision** — attach images; a dedicated vision model analyzes them
- **Live2D avatar** — WebGL rendering, hit-area interactions, audio-driven lip-sync
- **Voice I/O** — browser speech recognition + `edge-tts` streaming TTS
- **Live telemetry** — every backend step streamed to the UI over SSE (`/api/logs`)
- **Desktop or web** — run in a browser via Vite, or as a transparent always-on-top Tauri v2 window

See [docs/feature-overview.md](docs/feature-overview.md) for the full walkthrough, architecture diagram, and assessment.

## Project Structure

```
.
├── backend/                 # FastAPI + LangGraph service (:8000)
│   ├── main.py              # API: /api/chat, /api/logs (SSE), /api/tts
│   ├── core/
│   │   ├── agent/graph.py   # LangGraph state graph, tools, LLM setup
│   │   ├── memory/          # ChromaDB long-term memory
│   │   └── log_bus.py       # Real-time log pub/sub
│   ├── requirements.txt
│   └── .env.example
├── frontend/                # React 19 + Vite + Tailwind v4 + Tauri v2 (:1420)
│   ├── src/                 # App.tsx (HUD), Live2DAvatar.tsx
│   ├── src-tauri/           # Rust desktop shell
│   └── public/live2d/       # Live2D model (not committed, see below)
└── docs/
    ├── feature-overview.md
    ├── plans/               # Design & implementation plans
    └── screenshots/
```

## Getting Started

### Prerequisites

- Python 3.12
- Node.js 20+
- Rust toolchain (only for the Tauri desktop build)

### 1. Backend

```bash
cd backend
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys / model names
./venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Any OpenAI-compatible endpoint works (OpenAI, Volcengine Ark, etc.) via `OPENAI_BASE_URL`.
Runtime data (`aura_checkpoints.db`, `data/chroma_db/`) is created inside `backend/`.

### 2. Live2D model

The Live2D sample model (Shizuku) is subject to the [Live2D Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html) and is not included in this repository. Download it into `frontend/public/live2d/`:

```bash
npx degit guansss/pixi-live2d-display/test/assets/shizuku frontend/public/live2d
```

The frontend loads `/live2d/shizuku.model.json`; you can swap in any Cubism 2 model with the same layout.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                 # web: http://localhost:1420
npm run tauri dev           # or: native desktop window
```

## Tech Stack

| Layer | Stack |
| :--- | :--- |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4, Framer Motion, Pixi.js v6 + pixi-live2d-display |
| Desktop | Tauri v2 (Rust) |
| Backend | FastAPI, LangGraph, LangChain OpenAI, edge-tts, Tavily |
| Storage | SQLite (LangGraph checkpointer), ChromaDB |
