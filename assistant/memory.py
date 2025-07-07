import os
import json

# 👇 Use this host path for testing outside of Docker
#MEMORY_PATH = "/mnt/ai-data/memory/assistant1/history.json"
MEMORY_PATH = "/data/memory/assistant1/history.json"

def load_history():
    """Load past user-assistant exchanges from the JSON file."""
    if not os.path.exists(MEMORY_PATH):
        return []

    try:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_history(history):
    """Save the full history back to disk."""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

def add_exchange(prompt, response):
    """Append a new user–assistant exchange to memory."""
    history = load_history()
    history.append({
        "prompt": prompt.strip(),
        "response": response.strip()
    })
    save_history(history)
