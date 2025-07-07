import os
import argparse
import subprocess
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

def record(duration: int, fs: int = 16000):
    print(f"[1/4] Recording {duration}s of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return fs, audio

def save_wav(path: str, fs: int, audio: np.ndarray):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())

def transcribe(path: str):
    print("[2/4] Transcribing audio...")
    model = WhisperModel("small", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(path, beam_size=5)
    return "".join(seg.text for seg in segments)

def synthesize(text: str, voice_path: str, out_path: str):
    print(f"[3/4] Synthesizing speech: \"{text}\"")
    subprocess.run([
        "piper",
        "--model", voice_path,
        "--text", text,
        "--output", out_path
    ], check=True)

def play_wav(path: str):
    print("[4/4] Playing back...")
    with wave.open(path, 'rb') as wf:
        fs = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    data = np.frombuffer(frames, dtype=np.int16)
    sd.play(data, fs)
    sd.wait()

def main():
    p = argparse.ArgumentParser("Record→Whisper→Piper round-trip")
    p.add_argument("--listen", type=int, default=5, help="seconds to record")
    args = p.parse_args()

    in_path = "/tmp/assistant_input.wav"
    out_path = "/tmp/assistant_output.wav"

    fs, audio = record(args.listen)
    save_wav(in_path, fs, audio)

    text = transcribe(in_path)
    print("Transcription:", text)

    voice_dir = os.environ.get("PIPER_VOICE_DIR", "/data/piper_models")
    voices = [f for f in os.listdir(voice_dir) if f.endswith(".onnx")]
    if not voices:
        raise RuntimeError(f"No voice models in {voice_dir}")
    voice_path = os.path.join(voice_dir, voices[0])

    synthesize(text, voice_path, out_path)
    play_wav(out_path)

if __name__ == "__main__":
    main()
