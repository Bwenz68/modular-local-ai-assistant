import requests
import json

#OLLAMA_URL = 'http://localhost:11434/api/generate'
#OLLAMA_URL = 'http://host.docker.internal:11434/api/generate'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = 'llama3'

def query_ollama(prompt, model=MODEL):
    """
    Sends a prompt to the local Ollama model and streams back the response.

    Parameters:
    - prompt (str): The question or instruction to send to the model.
    - model (str): The model name to use (e.g., "llama3", "mistral", etc.).

    Returns:
    - The full model response as a string.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # set to True if you want to stream responses
    }

    try:
        response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[Error communicating with Ollama API] {e}"
