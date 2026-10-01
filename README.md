# Model-Host

A lightweight, unified local AI server providing OpenAI-compatible **Chat Completions** (via `llama.cpp` / GGUF) and **Audio Transcriptions** (via `faster-whisper`), equipped with dynamic VRAM management to optimize memory on consumer GPUs (e.g., 4GB–6GB VRAM).

---

## Features

- **LLM Text Generation**: Powered by `llama-cpp-python` with hardware acceleration (Vulkan/CUDA).
- **Speech-to-Text**: Powered by `faster-whisper` (CTranslate2) with fast INT8 inference on CPU/CUDA.
- **Dynamic VRAM Manager**: Gracefully evicts and loads model instances between speech and text tasks to prevent VRAM overflow.
- **Auto-Discovery**: Automatically discovers and loads any `.gguf` file placed in the `./models/` directory.
- **FastAPI & Uvicorn**: High-performance asynchronous REST endpoints with interactive Swagger UI.

---

## Project Structure

```text
Model-Host/
├── main.py              # FastAPI server application & dynamic VRAM manager
├── pyproject.toml       # Project metadata, dependencies & pre-built wheel sources
├── uv.lock              # Reproducible dependency lockfile
├── .python-version      # Target Python version (3.11)
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore file (excludes heavy model binaries)
└── models/              # Local model weights directory (ignored by git)
    ├── .gitkeep
    ├── *.gguf           # Place your GGUF model here (e.g. Llama-3.2-1B-Instruct)
    └── whisper-base/    # Optional local Whisper model folder
```

---

## Prerequisites

- **Python**: `>= 3.11`
- **uv Package Manager**: [Install uv](https://docs.astral.sh/uv/)

---

## Quickstart

### 1. Install Dependencies

Using `uv`:

```powershell
uv sync
```

> **Note for Windows GPU users**: The `llama-cpp-python` wheel configured in `pyproject.toml` comes pre-compiled with Vulkan GPU acceleration to run natively without requiring Visual Studio C++ build tools.

### 2. Add Model Weights

Place your downloaded `.gguf` file in the `./models/` directory:

```text
./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

The server automatically scans `./models/` for any `.gguf` file on startup.

### 3. Run the Server

```powershell
uv run main.py
```

The server will start at:
- **Base URL**: `http://127.0.0.1:8434`
- **Interactive Swagger Docs**: `http://127.0.0.1:8434/docs`
- **OpenAPI Specification**: `http://127.0.0.1:8434/openapi.json`

---

## Configuration

Customize runtime settings via environment variables (or create a `.env` file):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Network interface to bind |
| `PORT` | `8434` | Port to listen on |
| `LLM_MODEL_PATH` | `./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Specific path to GGUF model file |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`) or local path |
| `WHISPER_DEVICE` | `cpu` | Device for Whisper inference (`cpu` or `cuda`) |

---

## API Endpoints

### 1. Chat Completions
`POST /v1/chat/completions`

```bash
curl -X POST http://127.0.0.1:8434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain machine learning in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### 2. Audio Transcriptions
`POST /v1/audio/transcriptions`

```bash
curl -X POST http://127.0.0.1:8434/v1/audio/transcriptions \
  -F "file=@sample.wav"
```
