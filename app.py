import os

# AI TOGGLE: 1 = Cerebras, 0 = Ollama, 3 = Google AI Studio
AI_MODE = 3

os.environ["HF_HUB_OFFLINE"] = "0"
import sys
import torch
import scipy.io.wavfile
import subprocess
import platform
import time
import logging
import json
import requests
import threading
import uuid
import random
import struct
import socket
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template_string, Response
from dotenv import load_dotenv
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AUDIOCRAFT_DIR = _HERE / "audiocraft"
_ACESTEP_DIR = _HERE / "ACE-Step-1.5"

for _p in [str(_AUDIOCRAFT_DIR), str(_ACESTEP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv()

from audiocraft.models import MusicGen

try:
    from audiocraft.models import MusicGenMelody
    HAS_MELODY = True
except ImportError:
    try:
        from audiocraft.models.musicgen import MusicGen as _MG
        HAS_MELODY = hasattr(_MG, 'get_pretrained')
    except:
        HAS_MELODY = False

try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music as acestep_generate_music
    HAS_ACESTEP = True
    print("[ACE-Step] Library found and imported successfully.")
except ImportError as _e:
    HAS_ACESTEP = False
    print(f"[ACE-Step] Not available: {_e}.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

app = Flask(__name__, template_folder=str(_HERE / "templates"))
logging.basicConfig(level=logging.INFO)

EXPORT_FOLDER = str(_HERE / "exports")
METADATA_FILE = os.path.join(EXPORT_FOLDER, "metadata.json")
IMAGES_FOLDER = os.path.join(EXPORT_FOLDER, "images")
STEMS_FOLDER = os.path.join(EXPORT_FOLDER, "stems")
UPLOADS_FOLDER = os.path.join(EXPORT_FOLDER, "vocal_uploads")
IMAGES_METADATA_FILE = os.path.join(IMAGES_FOLDER, "img_metadata.json")
CHAT_HISTORY_FILE = os.path.join(EXPORT_FOLDER, "chat_histories.json")
CONFIG_FILE = os.path.join(EXPORT_FOLDER, "config.json")

ACESTEP_CHECKPOINT_DIR = str(_ACESTEP_DIR / "checkpoints")
ACESTEP_PROJECT_ROOT = str(_ACESTEP_DIR)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL = "llama3.1-8b"

# Google AI Studio (Gemini) — AI_MODE 3
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID", "1381348800000000000")

for folder in [EXPORT_FOLDER, IMAGES_FOLDER, STEMS_FOLDER, UPLOADS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

model = None
current_model_size = None
melody_model = None
sd_pipe = None

_acestep_dit_handler = None
_acestep_llm_handler = None
_acestep_lock = threading.Lock()

queue_lock = threading.Lock()
generation_queue = []
queue_running = False

# Stems generation job tracking
stems_jobs = {}   # filename -> {"status": "running"|"done"|"failed", "stems": {name: url}, "progress": str}
stems_lock = threading.Lock()

# =================== MODEL MANAGEMENT ===================

def get_model(size="small"):
    global model, current_model_size
    if model is None or current_model_size != size:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MusicGen.get_pretrained(f'facebook/musicgen-{size}')
        current_model_size = size
    return model

def get_melody_model():
    global melody_model
    if melody_model is None:
        melody_model = MusicGen.get_pretrained('facebook/musicgen-melody')
    return melody_model

def get_sd_pipe():
    global sd_pipe
    if sd_pipe is None and HAS_DIFFUSERS:
        model_id = "segmind/tiny-sd"
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        sd_pipe.to("cpu")
        sd_pipe.set_progress_bar_config(disable=True)
    return sd_pipe

def get_acestep_handlers():
    global _acestep_dit_handler, _acestep_llm_handler
    if not HAS_ACESTEP:
        raise RuntimeError("ACE-Step 1.5 is not installed.")
    with _acestep_lock:
        if _acestep_dit_handler is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _acestep_dit_handler = AceStepHandler()
            _acestep_dit_handler.initialize_service(
                project_root=ACESTEP_PROJECT_ROOT,
                config_path="acestep-v15-turbo",
                device=device,
            )
        if _acestep_llm_handler is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _acestep_llm_handler = LLMHandler()
            _acestep_llm_handler.initialize(
                checkpoint_dir=ACESTEP_CHECKPOINT_DIR,
                lm_model_path="acestep-5Hz-lm-0.6B",
                backend="torch",
                device=device,
            )
    return _acestep_dit_handler, _acestep_llm_handler

# =================== CONFIG ===================

DEFAULT_CONFIG = {
    "theme": "dark",
    "accentIndex": 0,
    "chatDefaultDuration": 20,
    "skipDurationPrompt": False,
    "defaultModel": "acestep",
    "defaultManualDuration": 30,
    "notifyOnGen": True,
    "generateCovers": True,
    "discordRpc": False,
    "audioBufferToRam": False,
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return {**DEFAULT_CONFIG, **json.load(f)}
            except json.JSONDecodeError:
                return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({**DEFAULT_CONFIG, **cfg}, f, indent=4)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(load_config())

@app.route('/api/config', methods=['POST'])
def set_config():
    save_config(request.json or {})
    return jsonify({"ok": True})

# =================== METADATA ===================

_meta_lock = threading.Lock()

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_metadata(md):
    with open(METADATA_FILE, 'w') as f:
        json.dump(md, f, indent=4)

def load_images_metadata():
    if os.path.exists(IMAGES_METADATA_FILE):
        with open(IMAGES_METADATA_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_images_metadata(md):
    with open(IMAGES_METADATA_FILE, 'w') as f:
        json.dump(md, f, indent=4)

def load_chat_histories():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_chat_histories(data):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# =================== RENAME ===================

@app.route('/api/rename', methods=['POST'])
def rename_track():
    data = request.json or {}
    filename = data.get('filename', '')
    new_name = (data.get('name') or '').strip()
    if not filename or not new_name:
        return jsonify({"ok": False, "error": "filename and name required"}), 400
    with _meta_lock:
        md = load_metadata()
        if filename not in md:
            return jsonify({"ok": False, "error": "Track not found"}), 404
        md[filename]["name"] = new_name
        save_metadata(md)
    return jsonify({"ok": True, "filename": filename, "name": new_name})

# =================== AUDIO STREAMING ===================

CHUNK_SIZE = 256 * 1024

def _wav_duration_hint(filepath):
    try:
        import soundfile as sf
        info = sf.info(filepath)
        return f"{info.duration:.3f}"
    except Exception:
        return None

def stream_audio_file(filepath):
    file_size = os.path.getsize(filepath)
    etag = f'"{file_size:x}-{int(os.path.getmtime(filepath)):x}"'
    duration_hint = _wav_duration_hint(filepath)

    base_headers = {
        'Content-Type': 'audio/wav',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
        'ETag': etag,
        'X-Content-Type-Options': 'nosniff',
    }
    if duration_hint:
        base_headers['X-Content-Duration'] = duration_hint

    if_none_match = request.headers.get('If-None-Match')
    if if_none_match and if_none_match.strip('"') == etag.strip('"'):
        return Response(status=304, headers={'ETag': etag, 'Cache-Control': 'no-store'})

    range_header = request.headers.get('Range')

    if not range_header:
        def generate_full():
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
        headers = {**base_headers, 'Content-Length': str(file_size)}
        return Response(generate_full(), status=200, headers=headers)

    try:
        raw = range_header.strip()
        if not raw.lower().startswith('bytes='):
            raise ValueError("unsupported unit")
        first_range = raw[6:].split(',')[0].strip()
        parts = first_range.split('-')
        start = int(parts[0]) if parts[0] else 0
        end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    except Exception:
        return Response("Invalid Range header", status=416, headers={
            'Content-Range': f'bytes */{file_size}'
        })

    start = max(0, min(start, file_size - 1))
    end   = max(start, min(end, file_size - 1))
    length = end - start + 1

    def generate_range():
        with open(filepath, 'rb') as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        **base_headers,
        'Content-Range':  f'bytes {start}-{end}/{file_size}',
        'Content-Length': str(length),
    }
    return Response(generate_range(), status=206, headers=headers)


@app.route('/api/buffer/<path:filename>')
def buffer_audio(filename):
    filepath = os.path.join(EXPORT_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    with open(filepath, 'rb') as f:
        data = f.read()

    duration_hint = _wav_duration_hint(filepath)
    headers = {
        'Content-Type': 'audio/wav',
        'Content-Length': str(len(data)),
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
        'Accept-Ranges': 'none',
    }
    if duration_hint:
        headers['X-Content-Duration'] = duration_hint
    return Response(data, status=200, headers=headers)


# =================== DISCORD RPC ===================

_DISCORD_IPC_LOCK = threading.Lock()
_discord_ipc_sock = None
_discord_ipc_connected = False

def _discord_ipc_path():
    if platform.system() == "Windows":
        return r'\\.\pipe\discord-ipc-0'
    for env_var in ("XDG_RUNTIME_DIR", "TMPDIR", "TMP", "TEMP"):
        val = os.environ.get(env_var)
        if val:
            p = os.path.join(val, "discord-ipc-0")
            if os.path.exists(p): return p
    for p in ("/tmp/discord-ipc-0", "/var/run/user/1000/discord-ipc-0"):
        if os.path.exists(p): return p
    return None

def _discord_send_raw(opcode, payload):
    global _discord_ipc_sock, _discord_ipc_connected
    data = json.dumps(payload).encode("utf-8")
    frame = struct.pack("<II", opcode, len(data)) + data
    if platform.system() == "Windows":
        pipe_path = _discord_ipc_path()
        if not pipe_path: raise ConnectionError("Discord pipe not found")
        with open(pipe_path, "r+b", buffering=0) as pipe:
            pipe.write(frame)
            h = pipe.read(8)
            if len(h) == 8:
                _, l = struct.unpack("<II", h)
                if l: pipe.read(l)
    else:
        if not _discord_ipc_connected or _discord_ipc_sock is None:
            ipc_path = _discord_ipc_path()
            if not ipc_path: raise ConnectionError("Discord IPC socket not found")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(ipc_path)
            _discord_ipc_sock = sock
            _discord_ipc_connected = True
            hs = json.dumps({"v": 1, "client_id": DISCORD_CLIENT_ID}).encode()
            _discord_ipc_sock.sendall(struct.pack("<II", 0, len(hs)) + hs)
            _discord_read_resp(_discord_ipc_sock)
        try:
            _discord_ipc_sock.sendall(frame)
            _discord_read_resp(_discord_ipc_sock)
        except (BrokenPipeError, ConnectionResetError, OSError):
            _discord_ipc_connected = False
            try: _discord_ipc_sock.close()
            except: pass
            _discord_ipc_sock = None
            _discord_send_raw(opcode, payload)

def _discord_read_resp(sock):
    try:
        hdr = b""
        while len(hdr) < 8:
            c = sock.recv(8 - len(hdr))
            if not c: break
            hdr += c
        if len(hdr) == 8:
            _, length = struct.unpack("<II", hdr)
            buf = b""
            while len(buf) < length:
                c = sock.recv(length - len(buf))
                if not c: break
                buf += c
    except: pass

def _discord_set_activity(state, details, large_image_key=None, start_timestamp=None):
    activity = {"state": state, "details": details, "instance": True}
    if start_timestamp: activity["timestamps"] = {"start": int(start_timestamp)}
    activity["assets"] = {"large_image": large_image_key or "plamiumai_logo", "large_text": "PlamiumAI Studio"}
    _discord_send_raw(1, {"cmd": "SET_ACTIVITY", "args": {"pid": os.getpid(), "activity": activity}, "nonce": str(uuid.uuid4())})

def _discord_clear_activity():
    _discord_send_raw(1, {"cmd": "SET_ACTIVITY", "args": {"pid": os.getpid(), "activity": None}, "nonce": str(uuid.uuid4())})

@app.route('/api/discord/rpc', methods=['POST'])
def discord_rpc_set():
    cfg = load_config()
    if not cfg.get("discordRpc", False):
        return jsonify({"ok": True, "skipped": True})
    data = request.json or {}
    with _DISCORD_IPC_LOCK:
        try:
            _discord_set_activity(data.get("state",""), data.get("details","PlamiumAI Studio"),
                                   data.get("large_image_key"), data.get("start_timestamp"))
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

@app.route('/api/discord/rpc/clear', methods=['POST'])
def discord_rpc_clear():
    cfg = load_config()
    if not cfg.get("discordRpc", False):
        return jsonify({"ok": True, "skipped": True})
    with _DISCORD_IPC_LOCK:
        try: _discord_clear_activity(); return jsonify({"ok": True})
        except Exception as e: return jsonify({"ok": False, "error": str(e)})

@app.route('/api/discord/rpc/test', methods=['POST'])
def discord_rpc_test():
    global _discord_ipc_sock, _discord_ipc_connected
    with _DISCORD_IPC_LOCK:
        _discord_ipc_connected = False
        if _discord_ipc_sock:
            try: _discord_ipc_sock.close()
            except: pass
            _discord_ipc_sock = None
        try:
            _discord_set_activity("Testing connection", "PlamiumAI Studio", start_timestamp=int(time.time()))
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

# =================== ACE-STEP (Python) ===================

# Vocal/mode presets for the Manual Studio "ACE Mode" dropdown.
ACE_MODES = {
    "default": {"caption_prefix": "", "task_type": "text2music", "instruction": None},
    "rap": {
        # Rap rhythm mode: bias LM planning toward tight rap cadence & flow.
        "caption_prefix": "rap, tight rhythmic rap flow, punchy rap cadence, strong beat emphasis, ",
        "task_type": "text2music",
        "instruction": None,
    },
    "custom_vocals": {
        # Uses ACE-Step 1.5 'complete' task: user-supplied vocal track is the
        # src_audio and the model builds the full mix around it.
        "caption_prefix": "",
        "task_type": "complete",
        "instruction": "Complete the input track with full accompaniment (drums, bass, harmony, instruments):",
    },
}

def generate_acestep_track(prompt, name, duration=30, lyrics="[Instrumental]",
                            inference_steps=27, guidance_scale=15.0, seed=-1, generate_cover=True,
                            ace_mode="default", src_audio_path=None):
    if not HAS_ACESTEP:
        return None, None
    try:
        dit_handler, llm_handler = get_acestep_handlers()
        timestamp = int(time.time())
        filename = f"ace_{timestamp}.wav"
        filepath = os.path.join(EXPORT_FOLDER, filename)
        final_name = name if name else f"ACE-Step Track {timestamp}"

        mode = ACE_MODES.get(ace_mode, ACE_MODES["default"])
        caption = (mode["caption_prefix"] or "") + prompt
        task_type = mode["task_type"]
        instruction = mode["instruction"]

        # custom_vocals requires a source audio; fall back to text2music if missing
        if ace_mode == "custom_vocals" and (not src_audio_path or not os.path.exists(src_audio_path)):
            task_type = "text2music"
            instruction = None

        param_kwargs = dict(
            task_type=task_type, caption=caption, lyrics=lyrics,
            duration=float(duration), inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else random.randint(0, 2**31 - 1),
            infer_method="ode", use_cot_caption=True,
        )
        if task_type != "text2music" and src_audio_path:
            param_kwargs["src_audio"] = src_audio_path
        if instruction:
            param_kwargs["instruction"] = instruction

        params = GenerationParams(**param_kwargs)
        config = GenerationConfig(batch_size=1, audio_format="wav")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = acestep_generate_music(dit_handler, llm_handler, params, config, save_dir=tmpdir)
            if not result.success: return None, None
            generated = list(Path(tmpdir).glob("*.wav")) or list(Path(tmpdir).glob("*.flac"))
            if not generated: return None, None
            src = str(generated[0])
            if src.endswith(".flac"):
                import soundfile as sf
                audio_data, sr = sf.read(src)
                scipy.io.wavfile.write(filepath, rate=sr, data=(audio_data*32767).clip(-32768,32767).astype(np.int16))
            else:
                import shutil; shutil.copy2(src, filepath)
        actual_duration = duration
        try:
            import soundfile as sf
            actual_duration = round(sf.info(filepath).duration, 1)
        except: pass
        with _meta_lock:
            md = load_metadata()
            md[filename] = {"name": final_name, "created": timestamp, "prompt": prompt,
                            "model": "acestep-0.6B", "duration": actual_duration,
                            "cover": None, "lyrics": lyrics if isinstance(lyrics, str) else "",
                            "ace_mode": ace_mode, "stems": {}}
            save_metadata(md)
        if generate_cover:
            threading.Thread(target=generate_cover_fastsd, args=(filename, prompt, final_name), daemon=True).start()
        return filename, final_name
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None

# =================== STEMS (ACE-Step 'extract' task) ===================

STEM_NAMES = ["vocals", "drums", "bass", "other"]
STEM_INSTRUCTIONS = {
    "vocals": "Extract the vocals track from the audio:",
    "drums":  "Extract the drums track from the audio:",
    "bass":   "Extract the bass track from the audio:",
    "other":  "Extract the instruments track without vocals, drums and bass from the audio:",
}

def _extract_single_stem(src_filepath, stem):
    """Run ACE-Step extract task to isolate one stem. Returns stem filename or None."""
    dit_handler, llm_handler = get_acestep_handlers()
    base = os.path.splitext(os.path.basename(src_filepath))[0]
    stem_filename = f"{base}__{stem}.wav"
    stem_path = os.path.join(STEMS_FOLDER, stem_filename)
    if os.path.exists(stem_path):
        return stem_filename
    params = GenerationParams(
        task_type="extract",
        src_audio=src_filepath,
        instruction=STEM_INSTRUCTIONS[stem],
        caption="",
        lyrics="",
        infer_method="ode",
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        result = acestep_generate_music(dit_handler, llm_handler, params, config, save_dir=tmpdir)
        if not result.success:
            return None
        generated = list(Path(tmpdir).glob("*.wav")) or list(Path(tmpdir).glob("*.flac"))
        if not generated:
            return None
        src = str(generated[0])
        if src.endswith(".flac"):
            import soundfile as sf
            audio_data, sr = sf.read(src)
            scipy.io.wavfile.write(stem_path, rate=sr, data=(audio_data*32767).clip(-32768,32767).astype(np.int16))
        else:
            shutil.copy2(src, stem_path)
    return stem_filename

def _stems_worker(filename):
    src_filepath = os.path.join(EXPORT_FOLDER, filename)
    stems_out = {}
    try:
        for i, stem in enumerate(STEM_NAMES):
            with stems_lock:
                stems_jobs[filename]["progress"] = f"Extracting {stem} ({i+1}/{len(STEM_NAMES)})..."
            stem_file = _extract_single_stem(src_filepath, stem)
            if stem_file:
                stems_out[stem] = f"/api/stems/serve/{stem_file}"
            with stems_lock:
                stems_jobs[filename]["stems"] = dict(stems_out)
        with _meta_lock:
            md = load_metadata()
            if filename in md:
                md[filename]["stems"] = stems_out
                save_metadata(md)
        with stems_lock:
            stems_jobs[filename]["status"] = "done" if stems_out else "failed"
            stems_jobs[filename]["progress"] = "Done" if stems_out else "Failed"
    except Exception as e:
        import traceback; traceback.print_exc()
        with stems_lock:
            stems_jobs[filename]["status"] = "failed"
            stems_jobs[filename]["progress"] = str(e)

@app.route('/api/stems/generate/<path:filename>', methods=['POST'])
def stems_generate(filename):
    if not HAS_ACESTEP:
        return jsonify({"ok": False, "error": "ACE-Step not installed — stems unavailable."}), 503
    src_filepath = os.path.join(EXPORT_FOLDER, filename)
    if not os.path.exists(src_filepath):
        return jsonify({"ok": False, "error": "Track not found"}), 404
    md = load_metadata()
    existing = md.get(filename, {}).get("stems") or {}
    if len(existing) >= len(STEM_NAMES):
        return jsonify({"ok": True, "status": "done", "stems": existing})
    with stems_lock:
        job = stems_jobs.get(filename)
        if job and job["status"] == "running":
            return jsonify({"ok": True, "status": "running", "progress": job.get("progress","")})
        stems_jobs[filename] = {"status": "running", "stems": dict(existing), "progress": "Starting..."}
    threading.Thread(target=_stems_worker, args=(filename,), daemon=True).start()
    return jsonify({"ok": True, "status": "running"})

@app.route('/api/stems/status/<path:filename>')
def stems_status(filename):
    md = load_metadata()
    existing = md.get(filename, {}).get("stems") or {}
    with stems_lock:
        job = stems_jobs.get(filename)
    if job:
        return jsonify({"status": job["status"], "stems": job.get("stems", existing), "progress": job.get("progress","")})
    if len(existing) >= len(STEM_NAMES):
        return jsonify({"status": "done", "stems": existing, "progress": "Done"})
    return jsonify({"status": "none", "stems": existing, "progress": ""})

@app.route('/api/stems/serve/<path:filename>')
def stems_serve(filename):
    filepath = os.path.join(STEMS_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Stem not found"}), 404
    return stream_audio_file(filepath)

# =================== COVER / IMAGE ===================

def generate_cover_fastsd(filename, cover_prompt, song_name):
    if not HAS_DIFFUSERS: return
    try:
        pipe = get_sd_pipe()
        image = pipe(f"album cover art, {cover_prompt}, high quality, digital art",
                     num_inference_steps=10, guidance_scale=7.0).images[0]
        cover_filename = filename.replace(".wav", ".jpg")
        image.save(os.path.join(EXPORT_FOLDER, cover_filename))
        with _meta_lock:
            md = load_metadata()
            if filename in md:
                md[filename]["cover"] = f"/api/cover/{cover_filename}"
                save_metadata(md)
    except Exception as e:
        print(f"[CoverGen] failed: {e}")

def generate_standalone_image(img_filename, prompt, title):
    if not HAS_DIFFUSERS: return False
    try:
        pipe = get_sd_pipe()
        image = pipe(f"{prompt}, high quality, digital art, detailed",
                     num_inference_steps=12, guidance_scale=7.5).images[0]
        image.save(os.path.join(IMAGES_FOLDER, img_filename))
        return True
    except: return False

# =================== CHAT HELPERS ===================

def ollama_chat(messages):
    try:
        r = requests.post("http://localhost:11434/api/chat",
                          json={"model": "phi4-mini:latest", "messages": messages, "stream": False}, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return {"error": str(e)}, "Ollama connection failed."

def cerebras_chat_with_fallback(messages):
    if not CEREBRAS_API_KEY:
        return {"error": "Cerebras API Key missing"}, "API Key not configured."
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}
    for i in range(5):
        try:
            r = requests.post("https://api.cerebras.ai/v1/chat/completions",
                              json={"model": CEREBRAS_MODEL, "messages": messages, "stream": False},
                              headers=headers, timeout=30)
            r.raise_for_status()
            return {"message": {"role": "assistant", "content": r.json()['choices'][0]['message']['content']}}, None
        except Exception as e:
            if i == 4: return {"error": str(e)}, "Cerebras API failed."
            time.sleep(2**i)

def gemini_chat(messages):
    """Google AI Studio (Gemini) chat — AI_MODE 3.
    Converts OpenAI-style messages to the Gemini generateContent format."""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API Key missing"}, "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env"
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": m.get("content", "")}],
        })
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    payload = {"contents": contents,
               "generationConfig": {"temperature": 0.9, "maxOutputTokens": 2048}}
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
    for i in range(4):
        try:
            r = requests.post(url, json=payload, timeout=45,
                              headers={"Content-Type": "application/json"})
            r.raise_for_status()
            data = r.json()
            text = ""
            for cand in data.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    text += part.get("text", "")
                break
            if not text:
                raise ValueError("Empty Gemini response")
            return {"message": {"role": "assistant", "content": text}}, None
        except Exception as e:
            if i == 3:
                return {"error": str(e)}, "Gemini API failed."
            time.sleep(1.5 ** i)

def ai_chat(messages):
    """Routes to the configured AI backend."""
    if AI_MODE == 1:
        return cerebras_chat_with_fallback(messages)
    if AI_MODE == 3:
        return gemini_chat(messages)
    return ollama_chat(messages)

def generate_music_track(prompt, name, model_size, duration, cover_prompt,
                          melody_audio_path=None, generate_cover=True):
    try:
        generator = get_melody_model() if model_size == "melody" else get_model(model_size)
        generator.set_generation_params(duration=duration, top_k=250, top_p=0.95, temperature=1.0)
        timestamp = int(time.time())
        filename = f"gen_{timestamp}.wav"
        filepath = os.path.join(EXPORT_FOLDER, filename)
        with torch.no_grad():
            if model_size == "melody" and melody_audio_path and os.path.exists(melody_audio_path):
                import torchaudio
                waveform, sr = torchaudio.load(melody_audio_path)
                wav = generator.generate_with_chroma([prompt], waveform.unsqueeze(0), sr, progress=True)
            else:
                wav = generator.generate([prompt], progress=True)
        scipy.io.wavfile.write(filepath, rate=generator.sample_rate, data=wav[0, 0].cpu().numpy())
        final_name = name if name else filename
        with _meta_lock:
            md = load_metadata()
            md[filename] = {"name": final_name, "created": timestamp, "prompt": prompt,
                            "model": model_size, "duration": duration, "cover": None, "lyrics": "", "stems": {}}
            save_metadata(md)
        if generate_cover:
            threading.Thread(target=generate_cover_fastsd, args=(filename, cover_prompt, final_name), daemon=True).start()
        return filename, final_name
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, None

# =================== QUEUE ===================

def process_queue():
    global queue_running, generation_queue
    queue_running = True
    while True:
        with queue_lock:
            if not generation_queue:
                queue_running = False
                break
            job = generation_queue.pop(0)
        cfg = load_config()
        gen_cover = cfg.get("generateCovers", True)
        model_size = job.get('model', 'small')

        if model_size == 'acestep':
            filename, final_name = generate_acestep_track(
                prompt=job['prompt'], name=job['name'],
                duration=job.get('duration', 30), lyrics=job.get('lyrics', '[Instrumental]'),
                generate_cover=gen_cover, ace_mode=job.get('ace_mode', 'default'))
        else:
            filename, final_name = generate_music_track(
                prompt=job['prompt'], name=job['name'], model_size=model_size,
                duration=job.get('duration', 15), cover_prompt=job.get('cover_prompt', job['prompt']),
                generate_cover=gen_cover)
        qmeta_path = os.path.join(EXPORT_FOLDER, "queue_results.json")
        try:
            qmeta = {}
            if os.path.exists(qmeta_path):
                with open(qmeta_path) as f: qmeta = json.load(f)
            qmeta[job.get('id')] = {"filename": filename, "name": final_name, "done": True}
            with open(qmeta_path, 'w') as f: json.dump(qmeta, f)
        except: pass

# =================== FLASK ROUTES ===================

@app.route('/')
def index():
    try:
        with open(str(_HERE / "templates" / "index.html"), 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        return f"Error loading index.html: {e}", 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    agent = data.get('agent', 'music')
    if agent == 'image':
        system_prompt = """You are a creative Image AI Assistant. Help the user design an image.
When ready, output ONLY:
<name>title</name>
<generate>detailed visual prompt</generate>"""
    else:
        system_prompt = """You are a highly creative Music AI Assistant. Help the user design a song.
When ready output ONLY:
<name>title</name>
<generate>extremely long and detailed style description</generate>
<cover>album cover visual description</cover>
<lyrics>[Verse]\n...\n[Chorus]\n...</lyrics>"""
    messages = [{"role": "system", "content": system_prompt},
                *data.get('history', []),
                {"role": "user", "content": data.get('message', '')}]
    try:
        resp, warn = ai_chat(messages)
        if warn: resp['_warning'] = warn
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat-histories', methods=['GET'])
def get_chat_histories():
    return jsonify(load_chat_histories())

@app.route('/api/chat-histories', methods=['POST'])
def save_chat_history():
    data = request.json
    histories = load_chat_histories()
    chat_id = data.get('id') or str(uuid.uuid4())
    histories[chat_id] = {"id": chat_id, "title": data.get('title','New Chat'),
                          "agent": data.get('agent','music'), "messages": data.get('messages',[]),
                          "updated": int(time.time())}
    save_chat_histories(histories)
    return jsonify({"id": chat_id})

@app.route('/api/chat-histories/<chat_id>', methods=['DELETE'])
def delete_chat_history(chat_id):
    h = load_chat_histories()
    if chat_id in h: del h[chat_id]; save_chat_histories(h)
    return jsonify({"ok": True})

@app.route('/api/library', methods=['GET'])
def library():
    try:
        metadata = load_metadata()
        tracks = []
        for f in os.listdir(EXPORT_FOLDER):
            if not f.endswith(".wav"): continue
            path = os.path.join(EXPORT_FOLDER, f)
            meta = metadata.get(f, {})
            tracks.append({"filename": f, "name": meta.get("name") or f,
                           "url": f"/api/play/{f}",
                           "size": f"{os.path.getsize(path)/(1024*1024):.1f} MB",
                           "model": meta.get("model","unknown"),
                           "created": meta.get("created", os.path.getmtime(path)),
                           "cover": meta.get("cover"), "prompt": meta.get("prompt",""),
                           "duration": meta.get("duration",0), "lyrics": meta.get("lyrics",""),
                           "stems": meta.get("stems", {}),
                           "ace_mode": meta.get("ace_mode", "default")})
        tracks.sort(key=lambda x: x['created'], reverse=True)
        return jsonify(tracks)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        # Supports JSON, or multipart/form-data when a vocal file is attached
        # (ACE Mode: custom_vocals).
        is_multipart = request.content_type and 'multipart/form-data' in request.content_type
        if is_multipart:
            data = {k: v for k, v in request.form.items()}
        else:
            data = request.json or {}

        prompt = data.get('prompt','')
        if not prompt: return jsonify({"status":"error","message":"No prompt"}), 400
        duration = int(data.get('duration', 30))
        custom_name = (data.get('name') or '').strip()
        model_size = data.get('model','acestep')
        lyrics = data.get('lyrics','[Instrumental]')
        ace_mode = data.get('ace_mode', 'default')
        cfg = load_config()
        gen_cover_raw = data.get('generate_cover', cfg.get('generateCovers', True))
        generate_cover = gen_cover_raw in (True, 'true', 'True', '1', 1)

        src_audio_path = None
        if is_multipart and 'vocal_audio' in request.files:
            f = request.files['vocal_audio']
            if f.filename:
                ext = os.path.splitext(f.filename)[1] or '.wav'
                src_audio_path = os.path.join(UPLOADS_FOLDER, f"vocals_{int(time.time())}{ext}")
                f.save(src_audio_path)

        if model_size == 'acestep':
            filename, final_name = generate_acestep_track(
                prompt=prompt, name=custom_name, duration=duration, lyrics=lyrics,
                inference_steps=int(data.get('inference_steps',27)),
                guidance_scale=float(data.get('guidance_scale',15.0)),
                seed=int(data.get('seed',-1)), generate_cover=generate_cover,
                ace_mode=ace_mode, src_audio_path=src_audio_path)
        else:
            filename, final_name = generate_music_track(
                prompt=prompt, name=custom_name, model_size=model_size,
                duration=duration, cover_prompt=(data.get('cover_prompt') or '').strip() or prompt,
                generate_cover=generate_cover)

        if not filename: return jsonify({"status":"error","message":"Generation failed"}), 500
        original_cover = data.get('original_cover')
        if original_cover:
            with _meta_lock:
                md = load_metadata()
                if filename in md:
                    md[filename]["cover"] = original_cover; save_metadata(md)
            return jsonify({"status":"success","filename":filename,"name":final_name,"cover_pending":False})
        return jsonify({"status":"success","filename":filename,"name":final_name,"cover_pending":generate_cover})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route('/api/generate-melody', methods=['POST'])
def generate_melody():
    try:
        prompt = request.form.get('prompt','')
        if not prompt: return jsonify({"status":"error","message":"No prompt"}), 400
        duration = int(request.form.get('duration',15))
        custom_name = request.form.get('name','').strip()
        cover_prompt = request.form.get('cover_prompt','').strip() or prompt
        cfg = load_config()
        melody_audio_path = None
        if 'melody_audio' in request.files:
            f = request.files['melody_audio']
            if f.filename:
                tmp = os.path.join(EXPORT_FOLDER, f"melody_tmp_{int(time.time())}{os.path.splitext(f.filename)[1]}")
                f.save(tmp); melody_audio_path = tmp
        filename, final_name = generate_music_track(
            prompt=prompt, name=custom_name, model_size="melody", duration=duration,
            cover_prompt=cover_prompt, melody_audio_path=melody_audio_path,
            generate_cover=cfg.get('generateCovers',True))
        if melody_audio_path and os.path.exists(melody_audio_path):
            try: os.remove(melody_audio_path)
            except: pass
        if not filename: return jsonify({"status":"error","message":"Melody generation failed"}), 500
        return jsonify({"status":"success","filename":filename,"name":final_name})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route('/api/acestep/status', methods=['GET'])
def acestep_status():
    return jsonify({"available": HAS_ACESTEP, "checkpoint_dir": ACESTEP_CHECKPOINT_DIR,
                    "loaded": _acestep_dit_handler is not None})

@app.route('/api/queue', methods=['POST'])
def add_to_queue():
    global queue_running
    data = request.json
    added = []
    with queue_lock:
        for job in data.get('jobs', []):
            job['id'] = str(uuid.uuid4())
            generation_queue.append(job)
            added.append(job['id'])
    if not queue_running:
        threading.Thread(target=process_queue, daemon=True).start()
    return jsonify({"status":"queued","ids":added,"queue_length":len(generation_queue)})

@app.route('/api/queue/status', methods=['GET'])
def queue_status():
    with queue_lock:
        return jsonify({"queue_length":len(generation_queue),"running":queue_running})

@app.route('/api/cover-status/<filename>')
def cover_status(filename):
    md = load_metadata()
    wav = filename if filename.endswith('.wav') else filename.replace('.jpg','.wav')
    cover = md.get(wav, {}).get("cover")
    return jsonify({"ready": bool(cover), "cover": cover})

@app.route('/api/cover/<path:filename>')
def serve_cover(filename):
    return send_from_directory(EXPORT_FOLDER, filename)

@app.route('/api/play/<path:filename>')
def play(filename):
    filepath = os.path.join(EXPORT_FOLDER, filename)
    if not os.path.exists(filepath): return jsonify({"error":"File not found"}), 404
    return stream_audio_file(filepath)

@app.route('/api/track/<filename>', methods=['GET'])
def get_track_metadata(filename):
    md = load_metadata()
    track = md.get(filename)
    if not track: return jsonify({"error":"Not found"}), 404
    return jsonify(track)

@app.route('/api/images/library', methods=['GET'])
def images_library():
    try:
        md = load_images_metadata()
        images = []
        for f in os.listdir(IMAGES_FOLDER):
            if f.lower().endswith(('.jpg','.png','.jpeg')) and f != 'img_metadata.json':
                meta = md.get(f, {})
                images.append({"filename":f,"title":meta.get("title",f),
                               "prompt":meta.get("prompt",""),
                               "created":meta.get("created", os.path.getmtime(os.path.join(IMAGES_FOLDER,f))),
                               "url":f"/api/images/serve/{f}"})
        images.sort(key=lambda x: x['created'], reverse=True)
        return jsonify(images)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route('/api/images/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt','')
        if not prompt: return jsonify({"status":"error","message":"No prompt"}), 400
        timestamp = int(time.time())
        img_filename = f"img_{timestamp}.jpg"
        title = data.get('title', f"Image_{timestamp}")
        md = load_images_metadata()
        md[img_filename] = {"title":title,"prompt":prompt,"created":timestamp,"status":"generating"}
        save_images_metadata(md)
        def gen():
            success = generate_standalone_image(img_filename, prompt, title)
            imd = load_images_metadata()
            if img_filename in imd:
                imd[img_filename]["status"] = "done" if success else "failed"
                save_images_metadata(imd)
        threading.Thread(target=gen, daemon=True).start()
        return jsonify({"status":"generating","filename":img_filename})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 500

@app.route('/api/images/status/<filename>')
def image_status(filename):
    md = load_images_metadata()
    done = md.get(filename,{}).get("status") == "done"
    return jsonify({"ready":done,"url":f"/api/images/serve/{filename}" if done else None})

@app.route('/api/images/serve/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/api/images/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    path = os.path.join(IMAGES_FOLDER, filename)
    if os.path.exists(path): os.remove(path)
    md = load_images_metadata()
    if filename in md: del md[filename]; save_images_metadata(md)
    return jsonify({"ok":True})

@app.route('/api/open-folder')
def open_folder():
    abs_path = os.path.abspath(EXPORT_FOLDER)
    if platform.system() == "Windows": os.startfile(abs_path)
    elif platform.system() == "Darwin": subprocess.Popen(["open", abs_path])
    else: subprocess.Popen(["xdg-open", abs_path])
    return jsonify({"status":"success"})


# ============================================================
# TUI
# ============================================================

def run_tui():
    import tty, termios, select, shutil

    MARGIN_TOP    = 2
    MARGIN_BOTTOM = 2
    MARGIN_LEFT   = 7
    MARGIN_RIGHT  = 7

    def clear():          return "\x1b[2J"
    def home():           return "\x1b[H"
    def hide_cursor():    return "\x1b[?25l"
    def show_cursor():    return "\x1b[?25h"
    def goto(r, c):       return f"\x1b[{r};{c}H"
    def sgr(*codes):      return f"\x1b[{';'.join(str(c) for c in codes)}m"
    def reset():          return sgr(0)
    def fg(n):   return sgr(38,5,n)
    def bg(n):   return sgr(48,5,n)
    def bold():  return sgr(1)

    COL_BG=234; COL_PANEL=236; COL_BORDER=240; COL_ACCENT=99
    COL_TEXT=252; COL_MUTED=244; COL_GOOD=35; COL_WARN=214; COL_ERR=196
    COL_HL_FG=15; COL_HL_BG=99

    def attr_normal():  return fg(COL_TEXT)+bg(COL_PANEL)
    def attr_muted():   return fg(COL_MUTED)+bg(COL_PANEL)
    def attr_accent():  return fg(COL_ACCENT)+bg(COL_PANEL)+bold()
    def attr_hl():      return fg(COL_HL_FG)+bg(COL_HL_BG)+bold()
    def attr_good():    return fg(COL_GOOD)+bg(COL_PANEL)
    def attr_warn():    return fg(COL_WARN)+bg(COL_PANEL)
    def attr_err():     return fg(COL_ERR)+bg(COL_PANEL)

    def trunc(s,n):
        s=str(s)
        if n<=0: return ""
        return s[:n-1]+"…" if len(s)>n else s.ljust(n)

    def pad(s,n):
        s=str(s)
        if n<=0: return ""
        return (s[:n-1]+"…" if len(s)>n else s).ljust(n)

    def get_term_size():
        try:
            sz=os.get_terminal_size()
            return sz.lines, sz.columns
        except:
            return 24,80

    def get_safe_size():
        total_rows,total_cols=get_term_size()
        sr=max(5,total_rows-MARGIN_TOP-MARGIN_BOTTOM)
        sc=max(20,total_cols-MARGIN_LEFT-MARGIN_RIGHT)
        return sr,sc

    VIEWS=["Library","Generate","Chat","Settings"]
    st={
        "view":0, "focus":"nav",
        "lib_list":[], "lib_cursor":0, "lib_scroll":0,
        "gen_fields":["prompt","name","dur","lyrics","model",
                      "ace_steps","ace_cfg","ace_seed","cover_prompt"],
        "gen_field":0, "gen_prompt":"", "gen_name":"", "gen_dur":"30",
        "gen_lyrics":"[Instrumental]",
        "gen_model":0,
        "gen_models":["ACE-Step Python","MusicGen Small","MusicGen Medium","MusicGen Melody"],
        "gen_model_keys":["acestep","small","medium","melody"],
        "gen_ace_steps":"27", "gen_ace_cfg":"15.0", "gen_ace_seed":"",
        "gen_cover_prompt":"",
        "generating":False, "gen_status":"",
        "ai_filling":False, "ai_fill_target":"",
        "chat_messages":[], "chat_input":"",
        "chat_awaiting_duration":False, "chat_pending_prompt":"",
        "chat_pending_name":"", "chat_pending_lyrics":"", "chat_pending_cover":"",
        "settings_cursor":0,
        "settings_keys":  ["generateCovers","skipDurationPrompt","notifyOnGen","discordRpc"],
        "settings_labels":["Generate Cover Art","Skip Duration Prompt","Notify on Generation","Discord Rich Presence"],
        "settings_vals":{},
        "now_playing":"", "volume":80, "audio_proc":None,
        "audio_lock":threading.Lock(),
        "status":"Ready — press Tab to move focus, q in nav to quit.",
        "status_col":COL_MUTED,
        "show_sink":False, "sinks":[], "sink_cursor":0, "active_sink":"",
    }

    def refresh_library():
        try:
            md=load_metadata(); tracks=[]
            for f in os.listdir(EXPORT_FOLDER):
                if not f.endswith(".wav"): continue
                path=os.path.join(EXPORT_FOLDER,f); meta=md.get(f,{})
                tracks.append({"filename":f,"name":meta.get("name",f),
                               "model":meta.get("model","?"),"duration":meta.get("duration",0),
                               "prompt":meta.get("prompt",""),"lyrics":meta.get("lyrics",""),"path":path})
            tracks.sort(key=lambda x:os.path.getmtime(x["path"]),reverse=True)
            st["lib_list"]=tracks
            if st["lib_cursor"]>=len(tracks): st["lib_cursor"]=max(0,len(tracks)-1)
        except: st["lib_list"]=[]

    def load_settings_vals():
        cfg=load_config()
        for k in st["settings_keys"]:
            st["settings_vals"][k]=cfg.get(k,False)

    def play_track(track):
        with st["audio_lock"]:
            if st["audio_proc"] and st["audio_proc"].poll() is None:
                st["audio_proc"].terminate()
                try: st["audio_proc"].wait(timeout=2)
                except: pass
        for player in ("mpv","ffplay","aplay"):
            if not shutil.which(player): continue
            if player=="mpv":
                cmd=["mpv","--no-video","--quiet",f"--volume={st['volume']}"]
                if st["active_sink"] and st["active_sink"]!="default":
                    cmd+=[f"--audio-device=pulse/{st['active_sink']}"]
                cmd.append(track["path"])
            elif player=="ffplay":
                cmd=["ffplay","-nodisp","-autoexit","-volume",str(st["volume"]),track["path"]]
            else:
                cmd=["aplay",track["path"]]
            with st["audio_lock"]:
                try:
                    st["audio_proc"]=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    st["now_playing"]=track["name"]
                    st["status"]=f"Playing: {track['name']}"
                    st["status_col"]=COL_ACCENT
                except Exception as e:
                    st["status"]=f"Playback error: {e}"; st["status_col"]=COL_ERR
            return
        st["status"]="No player found (install mpv/ffplay/aplay)"; st["status_col"]=COL_ERR

    def stop_audio():
        with st["audio_lock"]:
            if st["audio_proc"] and st["audio_proc"].poll() is None:
                st["audio_proc"].terminate()
                try: st["audio_proc"].wait(timeout=2)
                except: pass
            st["now_playing"]=""; st["status"]="Stopped."; st["status_col"]=COL_MUTED

    def detect_sinks():
        try:
            out=subprocess.check_output(["pactl","list","short","sinks"],text=True,stderr=subprocess.DEVNULL)
            sinks=[]
            for line in out.strip().splitlines():
                parts=line.split("\t")
                if len(parts)>=2: sinks.append((parts[1],parts[1].replace("_"," ")))
            st["sinks"]=sinks or [("default","Default Output")]
        except:
            st["sinks"]=[("default","Default Output")]

    _dirty=threading.Event()

    def _call_ai_sync(system_prompt,user_prompt):
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]
        resp,warn=ai_chat(messages)
        content=resp.get("message",{}).get("content","") or resp.get("error","")
        import re; content=re.sub(r'<[^>]+>[\s\S]*?</[^>]+>','',content)
        content=re.sub(r'<[^>]+>','',content)
        return content.strip()

    def do_ai_fill(target,extra_instruction=""):
        st["ai_filling"]=True; st["ai_fill_target"]=target
        st["status"]=f"AI filling: {target}…"; st["status_col"]=COL_WARN; _dirty.set()
        current_prompt=st["gen_prompt"]; current_lyrics=st["gen_lyrics"]; current_cover=st["gen_cover_prompt"]
        try:
            if target in ("prompt","all"):
                sys_p=("You are a music style tag generator. Output ONLY comma-separated style tags. No explanations.")
                user_p=(f"Refine: \"{current_prompt}\"\n" if current_prompt else "Generate a detailed music style prompt.\n")
                if extra_instruction: user_p+=f"Extra: {extra_instruction}"
                result=_call_ai_sync(sys_p,user_p)
                if result: st["gen_prompt"]=result; st["status"]="AI filled: style prompt"
                _dirty.set()
            if target in ("name","all"):
                sys_n="You are a creative music title writer. Output ONLY a short track title. No explanations."
                user_n=(f"Music style: {st['gen_prompt']}\n" if st["gen_prompt"] else "Create a creative track title.\n")
                result=_call_ai_sync(sys_n,user_n)
                if result: st["gen_name"]=result.strip().strip('"').strip("'"); st["status"]="AI filled: track name"
                _dirty.set()
            if target in ("lyrics","all"):
                sys_l="You are a lyricist. Output ONLY lyrics with [Verse]/[Chorus]/[Bridge]/[Outro] markers. No explanations."
                user_l=""
                if st["gen_prompt"]: user_l+=f"Style: {st['gen_prompt']}\n"
                is_draft=current_lyrics and current_lyrics!="[Instrumental]" and len(current_lyrics.strip())>10
                if is_draft: user_l+=f"Draft:\n{current_lyrics}\nComplete and refine."
                else: user_l+="Write complete lyrics."
                if extra_instruction: user_l+=f"\n{extra_instruction}"
                result=_call_ai_sync(sys_l,user_l)
                if result: st["gen_lyrics"]=result; st["status"]="AI filled: lyrics"
                _dirty.set()
            if target in ("cover","all"):
                sys_c="You are a visual art director for album cover art prompts. Output ONLY a visual description. No music tags."
                user_c=""
                if st["gen_prompt"]: user_c+=f"Music style: {st['gen_prompt']}\n"
                if current_cover: user_c+=f"Existing: \"{current_cover}\"\n"
                if not user_c: user_c="Create a compelling album cover art description."
                if extra_instruction: user_c+=f"\nInstruction: {extra_instruction}"
                result=_call_ai_sync(sys_c,user_c)
                if result: st["gen_cover_prompt"]=result; st["status"]="AI filled: cover prompt"
                _dirty.set()
            if target=="all": st["status"]="AI fill complete."; st["status_col"]=COL_GOOD
            else: st["status_col"]=COL_GOOD
        except Exception as e:
            st["status"]=f"AI fill error: {e}"; st["status_col"]=COL_ERR
        finally:
            st["ai_filling"]=False; _dirty.set()

    def do_generate():
        st["generating"]=True; st["gen_status"]="Initialising…"; _dirty.set()
        model_key=st["gen_model_keys"][st["gen_model"]]
        prompt=st["gen_prompt"].strip()
        name=st["gen_name"].strip() or f"TUI Track {int(time.time())}"
        dur=int(st["gen_dur"]) if st["gen_dur"].isdigit() else 30
        lyrics=st["gen_lyrics"] or "[Instrumental]"
        cover=st["gen_cover_prompt"].strip() or prompt
        cfg=load_config(); gen_cover=cfg.get("generateCovers",True)
        try: ace_steps=int(st["gen_ace_steps"]) if st["gen_ace_steps"].isdigit() else 27
        except: ace_steps=27
        try: ace_cfg=float(st["gen_ace_cfg"]) if st["gen_ace_cfg"] else 15.0
        except: ace_cfg=15.0
        try: ace_seed=int(st["gen_ace_seed"]) if st["gen_ace_seed"].strip() else -1
        except: ace_seed=-1
        try:
            if model_key=="acestep":
                st["gen_status"]="ACE-Step Python synthesising…"; _dirty.set()
                fn,fn_name=generate_acestep_track(
                    prompt=prompt,name=name,duration=dur,lyrics=lyrics,
                    inference_steps=ace_steps,guidance_scale=ace_cfg,seed=ace_seed,generate_cover=gen_cover)
            else:
                st["gen_status"]=f"MusicGen ({model_key}) generating…"; _dirty.set()
                fn,fn_name=generate_music_track(
                    prompt=prompt,name=name,model_size=model_key,
                    duration=dur,cover_prompt=cover,generate_cover=gen_cover)
            if fn:
                st["status"]=f"Done: {fn_name}"; st["status_col"]=COL_GOOD; refresh_library()
            else:
                st["status"]="Generation failed."; st["status_col"]=COL_ERR
        except Exception as e:
            st["status"]=f"Error: {e}"; st["status_col"]=COL_ERR
        finally:
            st["generating"]=False; st["gen_status"]=""; _dirty.set()

    def do_chat(user_msg):
        import re
        st["status"]="AI thinking…"; st["status_col"]=COL_WARN; _dirty.set()
        system=("You are a highly creative Music AI Assistant. When ready to generate output EXACTLY:\n"
                "<name>Title</name>\n<generate>style description</generate>\n"
                "<cover>visual description</cover>\n<lyrics>[Verse]\n...\n[Chorus]\n...</lyrics>")
        history_msgs=[{"role":m["role"],"content":m["text"]} for m in st["chat_messages"][-20:]]
        messages=[{"role":"system","content":system}]+history_msgs
        try:
            resp,warn=ai_chat(messages)
            content=resp.get("message",{}).get("content","") or resp.get("error","Error")
            gen_match=re.search(r'<generate>([\s\S]*?)</generate>',content,re.IGNORECASE)
            name_match=re.search(r'<name>([\s\S]*?)</name>',content,re.IGNORECASE)
            cover_match=re.search(r'<cover>([\s\S]*?)</cover>',content,re.IGNORECASE)
            lyrics_match=re.search(r'<lyrics>([\s\S]*?)</lyrics>',content,re.IGNORECASE)
            if gen_match:
                st["chat_pending_prompt"]=gen_match.group(1).strip()
                st["chat_pending_name"]=name_match.group(1).strip() if name_match else f"AI Track {int(time.time())}"
                st["chat_pending_cover"]=cover_match.group(1).strip() if cover_match else st["chat_pending_prompt"]
                st["chat_pending_lyrics"]=lyrics_match.group(1).strip() if lyrics_match else "[Instrumental]"
                display=re.sub(r'<[a-z]+>[\s\S]*?</[a-z]+>','',content,flags=re.IGNORECASE).strip()
                if not display: display=f'I\'ve designed "{st["chat_pending_name"]}"!'
                st["chat_messages"].append({"role":"assistant","text":display})
                st["chat_messages"].append({"role":"assistant","text":
                    f'✦ Ready: "{st["chat_pending_name"]}"\\n  → Enter duration in seconds or "skip" for 30s.'})
                st["chat_awaiting_duration"]=True
                st["status"]=f'Song designed — enter duration.'; st["status_col"]=COL_ACCENT
            else:
                display=re.sub(r'<[^>]+>','',content).strip()
                st["chat_messages"].append({"role":"assistant","text":display})
                st["status"]="AI responded."; st["status_col"]=COL_ACCENT
        except Exception as e:
            st["chat_messages"].append({"role":"assistant","text":f"[Error: {e}]"})
            st["status"]=f"Chat error: {e}"; st["status_col"]=COL_ERR
        _dirty.set()

    def do_chat_generate(duration):
        prompt=st["chat_pending_prompt"]; name=st["chat_pending_name"]
        lyrics=st["chat_pending_lyrics"]; cover=st["chat_pending_cover"]
        cfg=load_config(); gen_cover=cfg.get("generateCovers",True)
        st["chat_messages"].append({"role":"assistant","text":f"Generating {duration}s: \"{name}\"…"})
        st["status"]=f"Generating: {name} ({duration}s)…"; st["status_col"]=COL_WARN; _dirty.set()
        try:
            fn,fn_name=generate_acestep_track(prompt=prompt,name=name,duration=duration,
                                               lyrics=lyrics,generate_cover=gen_cover)
            if fn:
                st["chat_messages"].append({"role":"assistant","text":f'✓ Done! "{fn_name}" in library.'})
                st["status"]=f'Done: "{fn_name}"'; st["status_col"]=COL_GOOD; refresh_library()
            else:
                st["chat_messages"].append({"role":"assistant","text":"Generation failed."})
                st["status"]="Generation failed."; st["status_col"]=COL_ERR
        except Exception as e:
            st["chat_messages"].append({"role":"assistant","text":f"[Error: {e}]"})
            st["status"]=f"Error: {e}"; st["status_col"]=COL_ERR
        st["chat_awaiting_duration"]=False
        st["chat_pending_prompt"]=st["chat_pending_name"]=st["chat_pending_lyrics"]=st["chat_pending_cover"]=""
        _dirty.set()

    def render():
        rows,cols=get_term_size(); safe_rows,safe_cols=get_safe_size(); buf=[]; w=buf.append
        w(home())
        bg_line=bg(COL_BG)+" "*cols+reset()
        for _ in range(rows): w(bg_line)
        w(home())
        safe_top_abs=MARGIN_TOP+1; safe_left_abs=MARGIN_LEFT+1
        hrow=safe_top_abs
        w(goto(hrow,safe_left_abs)); w(fg(COL_ACCENT)+bg(COL_PANEL)+bold())
        title_str="PlamiumAI Studio "; w(title_str); used=len(title_str)
        for i,v in enumerate(VIEWS):
            if i==st["view"]: label=f" {v} "; w(attr_hl()+label+reset()+fg(COL_PANEL)+bg(COL_PANEL))
            else: label=f" {v} "; w(attr_muted()+label)
            used+=len(label)
        if st["now_playing"]:
            np_text=f"  ▶ {st['now_playing']}"
            remaining=safe_cols-used-len(np_text)-2
            if remaining>0: w(" "*remaining); w(fg(COL_ACCENT)+bg(COL_PANEL)+np_text[:safe_cols-used-4])
        vol_text=f"  vol:{st['volume']}%"
        abs_col=safe_left_abs+safe_cols-len(vol_text)
        if abs_col>safe_left_abs+used: w(goto(hrow,abs_col)); w(attr_muted()+vol_text)
        w(reset())
        sep_row=safe_top_abs+1; w(goto(sep_row,safe_left_abs)); w(fg(COL_BORDER)+bg(COL_BG)+"─"*safe_cols+reset())
        content_top_abs=safe_top_abs+2; status_row_abs=safe_top_abs+safe_rows-2
        bot_border_abs=safe_top_abs+safe_rows-1; content_h=status_row_abs-content_top_abs
        if st["view"]==0: render_library(buf,content_top_abs,content_h,safe_cols,safe_left_abs)
        elif st["view"]==1: render_generate(buf,content_top_abs,content_h,safe_cols,safe_left_abs)
        elif st["view"]==2: render_chat(buf,content_top_abs,content_h,safe_cols,safe_left_abs)
        elif st["view"]==3: render_settings(buf,content_top_abs,content_h,safe_cols,safe_left_abs)
        if st["show_sink"]: render_sink_overlay(buf,rows,cols)
        w(goto(status_row_abs,safe_left_abs))
        status_attr=(fg(COL_GOOD) if st["status_col"]==COL_GOOD else fg(COL_WARN) if st["status_col"]==COL_WARN
                     else fg(COL_ERR) if st["status_col"]==COL_ERR else fg(COL_ACCENT) if st["status_col"]==COL_ACCENT
                     else fg(COL_MUTED))
        w(status_attr+bg(COL_PANEL)); status_line=trunc(st['status'],safe_cols//2+10); w(status_line)
        if st["focus"]=="input": hints="  [Enter] confirm  [Esc] cancel"
        elif st["focus"]=="nav": hints="  [←→/↑↓] switch  [Enter] enter  [q] quit"
        else: hints="  [Tab] focus  [↑↓] nav  [Enter] select  [s] stop  [F5] refresh"
        w(fg(COL_MUTED)+bg(COL_PANEL)+trunc(hints,safe_cols-len(status_line)-1)); w(reset())
        w(goto(bot_border_abs,safe_left_abs)); w(fg(COL_BORDER)+bg(COL_BG)+"─"*safe_cols+reset())
        sys.stdout.write("".join(buf)); sys.stdout.flush()

    def draw_box(buf,top,left,h,w_box,title="",focused=False):
        battr=(fg(COL_ACCENT) if focused else fg(COL_BORDER))+bg(COL_BG)
        buf.append(goto(top,left)); buf.append(battr)
        if title:
            t=f" {title} "; inner=w_box-2-len(t)
            buf.append("┌"+t+"─"*max(0,inner)+"┐")
        else: buf.append("┌"+"─"*max(0,w_box-2)+"┐")
        for r in range(1,h-1):
            buf.append(goto(top+r,left)+battr+"│"); buf.append(goto(top+r,left+w_box-1)+battr+"│")
        buf.append(goto(top+h-1,left)+battr+"└"+"─"*max(0,w_box-2)+"┘"); buf.append(reset())

    def render_library(buf,top,h,cols,left):
        tracks=st["lib_list"]; cursor=st["lib_cursor"]
        focused=st["focus"]=="content" and st["view"]==0
        draw_box(buf,top,left,h,cols,"Library",focused)
        if not tracks:
            mid=top+h//2; msg="  Library is empty — generate some music first!"
            buf.append(goto(mid,left+max(0,(cols-len(msg))//2))); buf.append(attr_muted()+msg+reset()); return
        col_num=4; col_name=min(32,cols//4); col_mod=18; col_dur=6
        col_prom=max(10,cols-col_num-col_name-col_mod-col_dur-8)
        buf.append(goto(top+1,left+2)); buf.append(fg(COL_ACCENT)+bg(COL_PANEL)+bold())
        buf.append(pad("#",col_num)+" "+pad("Name",col_name)+" "+pad("Model",col_mod)+" "+pad("Dur",col_dur)+" "+pad("Prompt",col_prom)); buf.append(reset())
        list_top=top+2; visible=h-4
        if cursor<st["lib_scroll"]: st["lib_scroll"]=cursor
        elif cursor>=st["lib_scroll"]+visible: st["lib_scroll"]=cursor-visible+1
        st["lib_scroll"]=max(0,min(st["lib_scroll"],max(0,len(tracks)-visible)))
        for row_i in range(visible):
            abs_i=st["lib_scroll"]+row_i; screen_row=list_top+row_i
            if screen_row>=top+h-1: break
            buf.append(goto(screen_row,left+2))
            if abs_i>=len(tracks): buf.append(bg(COL_BG)+" "*(cols-4)+reset()); continue
            t=tracks[abs_i]; dur=f"{int(t['duration'])}s" if t["duration"] else "?"; is_cur=(abs_i==cursor)
            if is_cur: a=attr_hl(); num_s=pad(f">>{abs_i+1}",col_num)
            else: a=attr_normal(); num_s=pad(str(abs_i+1),col_num)
            buf.append(a+num_s+" "+pad(t["name"],col_name)+" "+pad(t["model"],col_mod)+" "+pad(dur,col_dur)+" "+pad(t["prompt"],col_prom)+reset())
        total=len(tracks); end_vis=min(st["lib_scroll"]+visible,total)
        scroll_hint=f" {st['lib_scroll']+1}-{end_vis}/{total} "
        buf.append(goto(top+h-1,left+cols-len(scroll_hint)-1)); buf.append(fg(COL_MUTED)+bg(COL_BG)+scroll_hint+reset())

    def render_generate(buf,top,h,cols,left):
        focused_box=st["focus"] in ("content","input") and st["view"]==1
        draw_box(buf,top,left,h,cols,"Generate — Manual Studio",focused_box)
        fields=st["gen_fields"]; cur_field=st["gen_field"]
        label_map={
            "prompt":("Style Prompt",st["gen_prompt"]),
            "name":("Track Name",st["gen_name"]),
            "dur":("Duration (s)",st["gen_dur"]),
            "lyrics":("Lyrics",st["gen_lyrics"].split('\n')[0]+("…" if '\n' in st["gen_lyrics"] else "")),
            "model":("Model",st["gen_models"][st["gen_model"]]),
            "ace_steps":("ACE Steps",st["gen_ace_steps"]),
            "ace_cfg":("ACE Guidance",st["gen_ace_cfg"]),
            "ace_seed":("ACE Seed",st["gen_ace_seed"] or "random"),
            "cover_prompt":("Cover Prompt",st["gen_cover_prompt"]),
        }
        label_w=16
        for i,fname in enumerate(fields):
            row=top+1+i
            if row>=top+h-3: break
            label,val=label_map.get(fname,(fname,""))
            is_active=(i==cur_field and st["focus"]=="input")
            is_sel=(i==cur_field and st["focus"]=="content")
            if fname=="model" and is_active: val=f"← {val} →"
            buf.append(goto(row,left+2)); val_w=cols-label_w-6
            if is_active:
                cursor_char="█" if fname!="model" else ""
                buf.append(fg(COL_ACCENT)+bg(COL_PANEL)+bold()+pad(label+":",label_w))
                buf.append(attr_hl()+f" {trunc(val,val_w)}{cursor_char}"+reset())
            elif is_sel:
                buf.append(fg(COL_ACCENT)+bg(COL_PANEL)+bold()+pad(label+":",label_w))
                buf.append(attr_muted()+f" {trunc(val,val_w)}"+reset())
            else:
                buf.append(attr_muted()+pad(label+":",label_w)); buf.append(attr_normal()+f" {trunc(val,val_w)}"+reset())
        ai_row=top+h-3; buf.append(goto(ai_row,left+2))
        if st["ai_filling"]: buf.append(fg(COL_WARN)+bg(COL_PANEL)+f"✦ AI filling: {st['ai_fill_target']}…"+reset())
        else: buf.append(attr_muted()+"[A] AI prompt  [L] AI lyrics  [N] AI name  [C] AI cover  [Z] fill all"+reset())
        hint_row=top+h-2; buf.append(goto(hint_row,left+2))
        if st["generating"]: buf.append(fg(COL_WARN)+bg(COL_PANEL)+f"⏳ {st['gen_status']}"+reset())
        else: buf.append(attr_muted()+"[Enter on prompt] generate  [↑↓] fields  [←→] model  [Esc] back"+reset())

    def render_chat(buf,top,h,cols,left):
        focused_box=st["focus"] in ("content","input") and st["view"]==2
        draw_box(buf,top,left,h,cols,"Music Agent Chat",focused_box)
        chat_h=h-5; messages=st["chat_messages"]; display_lines=[]; max_w=cols-6
        for msg in messages:
            role=msg.get("role","user"); content=msg.get("text",""); prefix="You  › " if role=="user" else "AI   › "
            lines_raw=content.split('\n')
            for li,raw_line in enumerate(lines_raw):
                if len(raw_line)==0: display_lines.append((role,"")); continue
                while len(raw_line)>max_w:
                    display_lines.append((role,(prefix if li==0 and not display_lines else "       ")+raw_line[:max_w]))
                    raw_line=raw_line[max_w:]; li=1
                display_lines.append((role,(prefix if li==0 else "       ")+raw_line))
            display_lines.append(("sep",""))
        visible_lines=display_lines[-chat_h:] if len(display_lines)>chat_h else display_lines
        for i in range(chat_h):
            row=top+1+i; buf.append(goto(row,left+2))
            if i<len(visible_lines):
                role,line=visible_lines[i]
                if role=="user": buf.append(fg(COL_ACCENT)+bg(COL_PANEL)+trunc(line,cols-4)+reset())
                elif role=="assistant":
                    if line.startswith("✦"): buf.append(fg(COL_GOOD)+bg(COL_PANEL)+bold()+trunc(line,cols-4)+reset())
                    else: buf.append(attr_normal()+trunc(line,cols-4)+reset())
                else: buf.append(bg(COL_BG)+" "*(cols-4)+reset())
            else: buf.append(bg(COL_BG)+" "*(cols-4)+reset())
        inp_row=top+h-3; buf.append(goto(inp_row,left+2))
        if st["chat_awaiting_duration"]: prompt_label=fg(COL_WARN)+bg(COL_PANEL)+"Duration (s)▸ "
        else: prompt_label=fg(COL_ACCENT)+bg(COL_PANEL)+"Message    ▸ "
        buf.append(prompt_label)
        inp_display=trunc(st["chat_input"],cols-18); cursor_vis="█" if st["focus"]=="input" else " "
        inp_attr=attr_hl() if st["focus"]=="input" else attr_muted()
        buf.append(inp_attr+inp_display+cursor_vis+reset())
        hint_row=top+h-2; buf.append(goto(hint_row,left+2))
        if st["chat_awaiting_duration"]: buf.append(attr_muted()+"[Enter] generate  [number] duration  'skip'=30s  [Esc] cancel"+reset())
        else: buf.append(attr_muted()+"[Enter] input mode  [Esc] nav — in input: [Enter] send  [Esc] cancel"+reset())

    def render_settings(buf,top,h,cols,left):
        focused_box=st["focus"]=="content" and st["view"]==3
        draw_box(buf,top,left,h,cols,"Settings",focused_box)
        cfg=load_config()
        for i,(key,label) in enumerate(zip(st["settings_keys"],st["settings_labels"])):
            row=top+1+i*2
            if row>=top+h-1: break
            val=st["settings_vals"].get(key,cfg.get(key,False))
            is_cur=(i==st["settings_cursor"] and st["focus"]=="content")
            val_s="ON " if val else "OFF"; val_a=fg(COL_GOOD) if val else fg(COL_ERR)
            buf.append(goto(row,left+3))
            if is_cur:
                buf.append(attr_hl()+f"  {pad(label,36)}"+reset()+val_a+bg(COL_PANEL)+bold()+f"  {val_s}"+reset())
            else:
                buf.append(attr_normal()+f"  {pad(label,36)}"+val_a+bg(COL_PANEL)+f"  {val_s}"+reset())
        buf.append(goto(top+h-2,left+3)); buf.append(attr_muted()+"[↑↓] navigate  [Enter/Space] toggle  [Esc] back"+reset())

    def render_sink_overlay(buf,rows,cols):
        ow=min(54,cols-4); oh=min(len(st["sinks"])+4,rows-4)
        ot=max(1,(rows-oh)//2); ol=max(1,(cols-ow)//2)
        draw_box(buf,ot,ol,oh,ow,"Audio Output Device",True)
        for i,(name,desc) in enumerate(st["sinks"]):
            row=ot+1+i
            if row>=ot+oh-1: break
            is_cur=(i==st["sink_cursor"]); active=(name==st["active_sink"]); suffix=" ◀ active" if active else "         "
            buf.append(goto(row,ol+2))
            if is_cur: buf.append(attr_hl()+pad(desc+suffix,ow-4)+reset())
            else: buf.append(attr_normal()+pad(desc+suffix,ow-4)+reset())
        buf.append(goto(ot+oh-1,ol+2)); buf.append(attr_muted()+"[Enter] select  [Esc] close"+reset())

    K_UP=b'\x1b[A'; K_DOWN=b'\x1b[B'; K_RIGHT=b'\x1b[C'; K_LEFT=b'\x1b[D'
    K_UP2=b'\x1bOA'; K_DOWN2=b'\x1bOB'; K_RIGHT2=b'\x1bOC'; K_LEFT2=b'\x1bOD'
    K_ENTER=b'\r'; K_ENTER2=b'\n'; K_ESC=b'\x1b'; K_TAB=b'\t'
    K_BS=b'\x7f'; K_BS2=b'\x08'; K_F5=b'\x1b[15~'; K_SPACE=b' '

    def is_up(k): return k in (K_UP,K_UP2)
    def is_down(k): return k in (K_DOWN,K_DOWN2)
    def is_left(k): return k in (K_LEFT,K_LEFT2)
    def is_right(k): return k in (K_RIGHT,K_RIGHT2)
    def is_enter(k): return k in (K_ENTER,K_ENTER2)
    def is_bs(k): return k in (K_BS,K_BS2)
    def is_printable(k): return len(k)==1 and 32<=k[0]<=126

    def handle_key(key):
        if st["show_sink"]:
            if is_up(key) and st["sink_cursor"]>0: st["sink_cursor"]-=1
            elif is_down(key) and st["sink_cursor"]<len(st["sinks"])-1: st["sink_cursor"]+=1
            elif is_enter(key) and st["sinks"]:
                name,desc=st["sinks"][st["sink_cursor"]]; st["active_sink"]=name
                st["status"]=f"Output: {desc}"; st["status_col"]=COL_GOOD; st["show_sink"]=False
            elif key in (K_ESC,b'\x1b'): st["show_sink"]=False
            return
        if st["focus"]=="input": handle_input_mode(key); return
        if key==K_F5: refresh_library(); st["status"]="Library refreshed."; st["status_col"]=COL_ACCENT; return
        if key in (b'+',b'='): st["volume"]=min(100,st["volume"]+5); st["status"]=f"Volume: {st['volume']}%"; st["status_col"]=COL_ACCENT; return
        if key==b'-': st["volume"]=max(0,st["volume"]-5); st["status"]=f"Volume: {st['volume']}%"; st["status_col"]=COL_ACCENT; return
        if key==b's': threading.Thread(target=stop_audio,daemon=True).start(); return
        if key==b'o' and st["focus"]!="input": detect_sinks(); st["show_sink"]=True; return
        if key==K_TAB:
            cycle=["nav","content","input"]; idx=cycle.index(st["focus"]) if st["focus"] in cycle else 0
            st["focus"]=cycle[(idx+1)%len(cycle)]; return
        if st["focus"]=="nav":
            if is_left(key) or is_up(key): st["view"]=(st["view"]-1)%len(VIEWS)
            elif is_right(key) or is_down(key): st["view"]=(st["view"]+1)%len(VIEWS)
            elif is_enter(key): st["focus"]="content"
            elif key==b'q': raise KeyboardInterrupt
            return
        if st["focus"]=="content": handle_content_mode(key); return

    def handle_content_mode(key):
        view=st["view"]
        if view==0:
            total=len(st["lib_list"])
            if is_up(key) and st["lib_cursor"]>0: st["lib_cursor"]-=1
            elif is_down(key) and st["lib_cursor"]<total-1: st["lib_cursor"]+=1
            elif is_enter(key) and st["lib_list"]:
                t=st["lib_list"][st["lib_cursor"]]; threading.Thread(target=play_track,args=(t,),daemon=True).start()
            elif key==K_ESC: st["focus"]="nav"
        elif view==1:
            fields=st["gen_fields"]; fi=st["gen_field"]
            if is_up(key): st["gen_field"]=(fi-1)%len(fields)
            elif is_down(key): st["gen_field"]=(fi+1)%len(fields)
            elif is_enter(key): st["focus"]="input"
            elif key==K_ESC: st["focus"]="nav"
            elif key==b'a' and not st["ai_filling"]: threading.Thread(target=do_ai_fill,args=("prompt",),daemon=True).start()
            elif key==b'l' and not st["ai_filling"]: threading.Thread(target=do_ai_fill,args=("lyrics",),daemon=True).start()
            elif key==b'n' and not st["ai_filling"]: threading.Thread(target=do_ai_fill,args=("name",),daemon=True).start()
            elif key==b'c' and not st["ai_filling"]: threading.Thread(target=do_ai_fill,args=("cover",),daemon=True).start()
            elif key==b'z' and not st["ai_filling"]: threading.Thread(target=do_ai_fill,args=("all",),daemon=True).start()
            elif key==b'g' and not st["generating"] and st["gen_prompt"].strip():
                threading.Thread(target=do_generate,daemon=True).start()
        elif view==2:
            if is_enter(key): st["focus"]="input"
            elif key==K_ESC: st["focus"]="nav"
        elif view==3:
            if is_up(key) and st["settings_cursor"]>0: st["settings_cursor"]-=1
            elif is_down(key) and st["settings_cursor"]<len(st["settings_keys"])-1: st["settings_cursor"]+=1
            elif is_enter(key) or key==K_SPACE:
                k=st["settings_keys"][st["settings_cursor"]]
                st["settings_vals"][k]=not st["settings_vals"].get(k,False)
                cfg=load_config(); cfg[k]=st["settings_vals"][k]; save_config(cfg)
                st["status"]=f"{k} = {st['settings_vals'][k]}"; st["status_col"]=COL_ACCENT
            elif key==K_ESC: st["focus"]="nav"

    def handle_input_mode(key):
        view=st["view"]
        if key==K_ESC: st["focus"]="content"; return
        if view==1:
            fields=st["gen_fields"]; fi=st["gen_field"]; fname=fields[fi]
            if is_up(key): st["gen_field"]=(fi-1)%len(fields); return
            if is_down(key) or key==K_TAB: st["gen_field"]=(fi+1)%len(fields); return
            if fname=="model":
                if is_left(key): st["gen_model"]=(st["gen_model"]-1)%len(st["gen_models"])
                elif is_right(key) or is_enter(key): st["gen_model"]=(st["gen_model"]+1)%len(st["gen_models"])
                return
            if is_enter(key):
                if fname in ("prompt","name","dur","ace_steps","ace_cfg","ace_seed","cover_prompt"):
                    if not st["generating"] and st["gen_prompt"].strip():
                        threading.Thread(target=do_generate,daemon=True).start()
                    st["focus"]="content"
                elif fname=="lyrics": st["gen_lyrics"]+="\n"
                return
            if is_bs(key):
                fm={"prompt":"gen_prompt","name":"gen_name","dur":"gen_dur","lyrics":"gen_lyrics",
                    "ace_steps":"gen_ace_steps","ace_cfg":"gen_ace_cfg","ace_seed":"gen_ace_seed","cover_prompt":"gen_cover_prompt"}
                bk=fm.get(fname)
                if bk: st[bk]=st[bk][:-1]
                return
            if is_printable(key):
                char=key.decode("utf-8","ignore")
                fm={"prompt":"gen_prompt","name":"gen_name","dur":"gen_dur","lyrics":"gen_lyrics",
                    "ace_steps":"gen_ace_steps","ace_cfg":"gen_ace_cfg","ace_seed":"gen_ace_seed","cover_prompt":"gen_cover_prompt"}
                bk=fm.get(fname)
                if bk: st[bk]=st[bk]+char
            return
        if view==2:
            if is_enter(key):
                msg=st["chat_input"].strip()
                if not msg: st["focus"]="content"; return
                if st["chat_awaiting_duration"]:
                    if msg.lower()=="skip": duration=30
                    else:
                        try: duration=max(5,min(600,int(msg)))
                        except ValueError:
                            st["chat_messages"].append({"role":"assistant","text":"Please enter a number or 'skip'."})
                            st["chat_input"]=""; _dirty.set(); return
                    st["chat_input"]=""; st["focus"]="content"
                    threading.Thread(target=do_chat_generate,args=(duration,),daemon=True).start()
                else:
                    user_text=msg; st["chat_messages"].append({"role":"user","text":user_text})
                    st["chat_input"]=""; st["focus"]="content"
                    threading.Thread(target=do_chat,args=(user_text,),daemon=True).start()
                return
            if is_bs(key): st["chat_input"]=st["chat_input"][:-1]; return
            if is_printable(key): st["chat_input"]+=key.decode("utf-8","ignore")
            return

    refresh_library(); load_settings_vals(); detect_sinks()
    if st["sinks"]: st["active_sink"]=st["sinks"][0][0]
    fd=sys.stdin.fileno(); old_term=termios.tcgetattr(fd)
    sys.stdout.write(hide_cursor()+clear()); sys.stdout.flush()
    try:
        import select as _select
        tty.setraw(fd); render()
        while True:
            if _select.select([fd],[],[],0.12)[0]:
                data=os.read(fd,16)
                if data:
                    if data==b'q' and st["focus"]=="nav": break
                    try: handle_key(data)
                    except KeyboardInterrupt: break
                    render()
            elif _dirty.is_set():
                _dirty.clear(); render()
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_term)
        sys.stdout.write(show_cursor()+clear()+home()); sys.stdout.flush()
        stop_audio(); print("PlamiumAI Studio closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PlamiumAI Studio")
    parser.add_argument("--tui",     action="store_true", help="Flask server + TUI")
    parser.add_argument("--tuionly", action="store_true", help="TUI only (no Flask)")
    args = parser.parse_args()

    if args.tuionly:
        run_tui()
    elif args.tui:
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
            daemon=True)
        flask_thread.start()
        time.sleep(1.5)
        run_tui()
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)