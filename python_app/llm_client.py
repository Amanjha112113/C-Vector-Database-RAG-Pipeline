import requests
import json
from typing import List, Dict, Optional

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}/api"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "qwen2.5:3b"

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            payload = {
                "model": self.embed_model,
                "prompt": text
            }
            response = requests.post(f"{self.base_url}/embeddings", json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                print(f"Ollama embedding error: {response.status_code} {response.text}")
                return []
        except requests.RequestException as e:
            print(f"Ollama request exception: {e}")
            return []

    def generate(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.gen_model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(f"{self.base_url}/generate", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "")
            return "ERROR: Ollama unavailable."
        except requests.RequestException:
            return "ERROR: Request to Ollama failed. Is it running?"

# Helper for splitting text into chunks for RAG
def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks
