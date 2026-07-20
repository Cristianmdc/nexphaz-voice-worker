"""NexPhaz own-voice worker — Chatterbox TTS on RunPod Serverless.

Input (event["input"]):
  text       str, required — what to say (keep <= ~500 chars per call; the
             platform splits long scripts into sentence chunks and concats)
  voice_url  str — public URL of the owner's reference sample (preferred;
             the platform passes its HMAC-signed asset URL)
  voice_b64  str — base64 audio as fallback when no public URL exists
  exaggeration, cfg_weight  float, optional — Chatterbox style knobs

Output:
  { "audio_base64": <wav b64>, "sample_rate": int, "seconds": float }

The reference sample is what makes it the OWNER'S voice — Chatterbox is
zero-shot: no per-client training, the sample conditions every generation.
"""
import base64
import os
import tempfile
import time

import requests
import runpod
import torch
import torchaudio

MODEL = None


def _get_model():
    global MODEL
    if MODEL is None:
        from chatterbox.tts import ChatterboxTTS
        t0 = time.time()
        MODEL = ChatterboxTTS.from_pretrained(device="cuda")
        print(f"[worker] model loaded in {time.time()-t0:.1f}s")
    return MODEL


def _fetch_reference(inp):
    """Write the owner's reference sample to a temp file, return its path."""
    raw = None
    url = inp.get("voice_url")
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = r.content
    elif inp.get("voice_b64"):
        raw = base64.b64decode(inp["voice_b64"])
    if not raw:
        return None
    f = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    f.write(raw)
    f.close()
    return f.name


def handler(event):
    inp = event.get("input") or {}
    text = (inp.get("text") or inp.get("prompt") or "").strip()
    if not text:
        return {"error": "text required"}
    if len(text) > 900:
        return {"error": "text too long for one call (max 900 chars) — chunk it"}

    ref_path = None
    try:
        ref_path = _fetch_reference(inp)
    except Exception as e:
        return {"error": f"could not fetch reference sample: {e}"}
    if not ref_path:
        return {"error": "voice_url or voice_b64 required"}

    model = _get_model()
    kwargs = {"audio_prompt_path": ref_path}
    for k in ("exaggeration", "cfg_weight"):
        if inp.get(k) is not None:
            kwargs[k] = float(inp[k])

    t0 = time.time()
    wav = model.generate(text, **kwargs)
    gen_s = time.time() - t0

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    torchaudio.save(out_path, wav, model.sr)
    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    seconds = float(wav.shape[-1]) / float(model.sr)
    print(f"[worker] {len(text)} chars -> {seconds:.1f}s audio in {gen_s:.1f}s")

    for p in (ref_path, out_path):
        try:
            os.unlink(p)
        except OSError:
            pass

    return {"audio_base64": b64, "sample_rate": int(model.sr), "seconds": round(seconds, 2)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
