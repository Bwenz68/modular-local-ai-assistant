import sounddevice as sd
import numpy as np
import queue
import tempfile
import os
import subprocess
from faster_whisper import WhisperModel


Q = queue.Queue()

def _callback(indata, frames, time, status):
    if status:
        print(status)
    Q.put(indata.copy())

def listen_microphone(duration=5, sample_rate=96000):
    """Record mic input for `duration` seconds, return transcribed text."""
    print(f"🎙️ Listening for {duration} seconds...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        filename = tmpfile.name

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=_callback):
            with open(filename, 'wb') as f:
                recording = []
                for _ in range(0, int(sample_rate / 1024 * duration)):
                    data = Q.get()
                    recording.append(data)
                audio_data = np.concatenate(recording)
                audio_data = np.int16(audio_data * 32767)
                import soundfile as sf
                sf.write(filename, audio_data, sample_rate)

        model = WhisperModel("base.en", compute_type="auto")
        segments, _ = model.transcribe(filename)
        text = " ".join([seg.text for seg in segments])
        print(f"📝 Transcription: {text}")
        return text.strip()

    finally:
        os.remove(filename)


def speak_text(text):
    """Use Piper to speak a response out loud."""
    print(f"🗣️ Speaking: {text}")
    voice = "en_US-lessac-low.onnx"
    command = [
        "piper",
        "--model", f"/data/piper_models/{voice}",
        "--text", text
    ]
    subprocess.run(command)
