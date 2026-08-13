"""
Optional, ungraded extension — only used when MOCK_LLM=0 is explicitly set.

Thin wrapper around Groq's free-tier API (OpenAI-compatible /chat/completions
endpoint). Reads the API key from the GROQ_API_KEY environment variable.
Any other genuinely-free-tier LLM API can be substituted here without
touching graph.py, since graph.py only calls call_llm(prompt).

Not required for grading: the required baseline (MOCK_LLM unset or 1)
never imports or calls this module's network path.
"""

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    Send `prompt` to the configured LLM and return the raw text response.
    Raises RuntimeError if GROQ_API_KEY is not set or the request fails.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires a GROQ_API_KEY environment variable "
            "(free signup at console.groq.com, no card required)."
        )

    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
