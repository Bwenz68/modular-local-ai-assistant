#!/bin/bash

set -e

echo "🧠 Modular AI Assistant - Full System Test"
echo "-----------------------------------------"

echo "⏳ Starting container in detached mode..."
docker compose up -d assistant

echo "✅ Testing GPU / CUDA..."
docker compose run --rm assistant python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "📸 Running YOLOv8 on 'bus.jpg'..."
docker compose run --rm assistant python -m assistant.vision bus.jpg

echo "🗣️ Recording 5 seconds of microphone input using Whisper..."
docker compose exec assistant python -m assistant.speech --listen 5

echo "🔊 Playing test audio with Piper..."
docker compose exec assistant python -c "
from piper_tts import PiperVoice
voice = PiperVoice.load('/data/piper_models/en_US-lessac-low.onnx')
voice.speak('Hello Bill, this is a Piper test using your custom voice model.', speaker=0)
"

echo "📚 Testing LangChain RAG retrieval..."
docker compose run --rm assistant python test_rag.py

echo "🎉 All core modules tested successfully!"
