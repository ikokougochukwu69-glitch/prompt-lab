"""Model clients. Each exposes complete(prompt, source_text) -> str.

- MockClient runs offline with a keyword baseline, so the project works
  without an API key. It is NOT a language model and ignores the prompt.
- AnthropicClient and OpenAIClient call real models. Install the matching
  SDK and set the API key environment variable to use them.
"""

import os
import re

POSITIVE_WORDS = {
    "love", "great", "best", "perfect", "perfectly", "recommend", "exceeded",
    "premium", "gorgeous", "comfortable", "good", "excellent", "silky",
    "value", "better", "works",
}
NEGATIVE_WORDS = {
    "broke", "waste", "never", "missing", "jammed", "headache", "rejected",
    "stopped", "bad", "worst", "cracks", "burnt", "slow", "zero", "returned",
    "worth",
}


class MockClient:
    """Keyword-counting baseline. Useful for testing the pipeline offline."""

    name = "mock"

    def complete(self, prompt: str, source_text: str) -> str:
        words = re.findall(r"[a-z']+", source_text.lower())
        score = sum(w in POSITIVE_WORDS for w in words) - sum(
            w in NEGATIVE_WORDS for w in words
        )
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 300):
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("Run: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("Set the ANTHROPIC_API_KEY environment variable.")
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, source_text: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str):
        try:
            import openai
        except ImportError as e:
            raise SystemExit("Run: pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("Set the OPENAI_API_KEY environment variable.")
        self._client = openai.OpenAI()
        self.model = model

    def complete(self, prompt: str, source_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


def get_client(provider: str, model: str | None = None):
    if provider == "mock":
        return MockClient()
    if not model:
        raise SystemExit(f"--model is required for provider '{provider}'.")
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    raise SystemExit(f"Unknown provider '{provider}'. Use mock, anthropic or openai.")
