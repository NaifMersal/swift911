import json
import librosa
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from TTS.api import TTS
from agents import Operator, Observer
import io
from scipy.io import wavfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# Constants
SAMPLE_RATE = 16000

STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DCA"
SPAEKER_SPEED = 2
# Load models and initialize pipelines
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model with performance tweaks
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    STT_MODEL,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    # attn_implementation="flash_attention_2",  # You may switch to 'sdpa' or remove if already using compile below
).to(device)

# Enable static cache and Torch compile for faster forward passes
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

# Processor
processor = AutoProcessor.from_pretrained(STT_MODEL)

# Create pipeline
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# warmup steps
for length_sec in (10, 30,10):
    L = length_sec * SAMPLE_RATE
    dummy = torch.randn(L, dtype=torch.float32, device=device)
    with sdpa_kernel(SDPBackend.MATH):
        _ = transcriber(
            {"sampling_rate": SAMPLE_RATE, "raw": dummy.cpu().numpy()},
            generate_kwargs={"language":"english","task":"transcribe"},
        )


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Load models and initialize pipelines
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32





tts = TTS(model_name=TTS_MODEL,gpu=torch.cuda.is_available(), progress_bar=False)

tts_sample_rate = tts.synthesizer.tts_config.audio["sample_rate"]

def convert_int16_float(chunk: bytes):
    # Convert bytes to numpy array
    audio_array = np.frombuffer(chunk, dtype=np.int16)
    # Convert to float32 and normalize
    audio_float = audio_array.astype(np.float32) / (32768.0)
    
    return audio_float
def float32_to_pcm16(float_array):
    if isinstance(float_array, list):
        float_array = np.array(float_array)
    if tts_sample_rate != SAMPLE_RATE:
        float_array = librosa.resample(float_array, orig_sr=tts_sample_rate, target_sr=SAMPLE_RATE)

    # Scale the float values
    scaled = np.clip(float_array * 32768, -32768, 32767)
    # Convert to 16-bit PCM
    pcm16 = scaled.astype(np.int16)
    return pcm16

async def replay_with_audio(audio,operator:Operator):
    audio_float = convert_int16_float(audio)

    with sdpa_kernel(SDPBackend.MATH):
        result = transcriber({"sampling_rate": SAMPLE_RATE, "raw": audio_float}, 
                                    generate_kwargs={"language": "english", "task": "transcribe"})

    audio_PCM =float32_to_pcm16(tts.tts(text = operator.get_message(result['text']), split_sentences=False, speed=SPAEKER_SPEED))
    return audio_PCM


@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    observer = Observer()
    operator =Operator()
    audio_signal =float32_to_pcm16(tts.tts(text=operator.opening_message, split_sentences=False, speed=SPAEKER_SPEED))
    buffer = io.BytesIO()
    wavfile.write(buffer, SAMPLE_RATE, audio_signal)
    await websocket.send_bytes(buffer.getvalue())
    buffer.flush()
    try:

        while True:
            data = await websocket.receive()
            if "bytes" in data:
                # Process audio chunk
                audio_chunk = data["bytes"]
                audio = await replay_with_audio(audio_chunk, operator)
                wavfile.write(buffer, SAMPLE_RATE, audio)
                
                extracted_features,is_dispatch_ready =  observer.extract_features(operator.conversation_history)
                message =json.dumps({
                    'extracted_features': extracted_features.dict(),
                    'is_dispatch_ready': is_dispatch_ready,
                })
                await websocket.send_text(message)

                    
                await websocket.send_bytes(buffer.getvalue())
                buffer.flush()
                if operator.is_finished:
                    # await websocket.send_text(chatLLM.response)
                    await websocket.close(code=1000, reason="Normal closure")

                
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up if necessary
        pass

