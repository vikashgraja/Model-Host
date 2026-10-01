import os
import gc
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import model engines
from faster_whisper import WhisperModel
from llama_cpp import Llama

# Set up FastAPI
app = FastAPI(title="Unified Local AI Server")

# --- MODEL PATHS ---
# Update this to point to your downloaded GGUF file or directory
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # or "tiny", "small", "medium"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" (recommended) or "cuda"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


def resolve_llm_model_path():
    if os.path.exists(LLM_MODEL_PATH):
        return LLM_MODEL_PATH
    models_dir = "./models"
    if os.path.exists(models_dir):
        ggufs = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".gguf")]
        if ggufs:
            return ggufs[0]
    return LLM_MODEL_PATH


class ModelManager:
    def __init__(self):
        self.active_model_type = None
        self.llm = None
        self.whisper = None
        self.lock = asyncio.Lock()

    def clear_vram(self):
        """Destroys active model instances and flushes GPU VRAM."""
        if self.llm is not None:
            print("Unloading Text LLM from VRAM...")
            # Llama-cpp-python requires calling close() to free C++ memory
            self.llm.close()
            del self.llm
            self.llm = None

        if self.whisper is not None:
            print("Unloading Whisper from VRAM...")
            del self.whisper
            self.whisper = None

        # Force Python Garbage Collector
        gc.collect()

        # Try to empty CUDA cache if PyTorch is present in the environment
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def get_llm(self):
        """Loads the text model, evicting Whisper if necessary."""
        async with self.lock:
            if self.active_model_type != "llm":
                self.clear_vram()
                model_file = resolve_llm_model_path()
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"No GGUF model found in './models'. Expected: {LLM_MODEL_PATH}")
                print(f"Loading Text LLM ({model_file}) into VRAM...")
                # n_gpu_layers=-1 offloads the entire model to the GPU
                self.llm = Llama(model_path=model_file, n_gpu_layers=-1, n_ctx=2048, verbose=False)
                self.active_model_type = "llm"
            return self.llm

    async def get_whisper(self):
        """Loads Whisper, evicting the text model if necessary."""
        async with self.lock:
            if self.active_model_type != "whisper":
                if WHISPER_DEVICE == "cuda":
                    self.clear_vram()
                model_target = "./models/whisper-base" if os.path.exists("./models/whisper-base") else WHISPER_MODEL_SIZE
                print(f"Loading Whisper from '{model_target}' on {WHISPER_DEVICE}...")
                try:
                    self.whisper = WhisperModel(model_target, device=WHISPER_DEVICE, compute_type="int8")
                except Exception as e:
                    print(f"Whisper device '{WHISPER_DEVICE}' failed ({e}), falling back to CPU...")
                    self.whisper = WhisperModel(model_target, device="cpu", compute_type="int8")
                self.active_model_type = "whisper"
            return self.whisper


# Initialize our dynamic VRAM manager
manager = ModelManager()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Get the LLM (this will trigger a VRAM swap if Whisper is currently loaded)
        llm = await manager.get_llm()

        # Format messages for llama_cpp
        formatted_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = llm.create_chat_completion(
            messages=formatted_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the incoming audio chunk to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    try:
        # Get Whisper (this will trigger a VRAM swap if the LLM is currently loaded)
        whisper = await manager.get_whisper()

        segments, info = whisper.transcribe(temp_audio_path, beam_size=5)
        full_text = "".join([segment.text for segment in segments])

        return {
            "text": full_text.strip(),
            "language": info.language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8434"))
    uvicorn.run(app, host=host, port=port)