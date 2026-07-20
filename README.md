# nexphaz-voice-worker

Own-voice TTS worker (Chatterbox, zero-shot voice conditioning) for RunPod Serverless.
Input: { text, voice_url | voice_b64 } → Output: { audio_base64 (wav), sample_rate, seconds }.
Image builds via GitHub Actions → ghcr.io/cristianmdc/nexphaz-voice-worker.
