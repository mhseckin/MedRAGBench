"""
LLM provider abstraction for MedRAGBench.

All calls to an LLM (chat completions) and to the embedding model go through
this module, so that switching providers touches exactly one file.

Today this uses OpenAI:
    * chat:        GPT-5 via the Chat Completions API
    * embeddings:  text-embedding-3-small

To switch chat generation to Anthropic Claude, see the clearly marked
`chat_anthropic` function below and the README section "Switching to Claude".
Note that Anthropic does not provide an embeddings endpoint, so embeddings
stay on OpenAI (or another embeddings provider) even when chat is on Claude.
"""

from __future__ import annotations

import os
import time
from typing import List

from . import config


# --------------------------------------------------------------------------
# Lazy client construction (so importing this module never crashes if a key
# is missing; the error surfaces only when a call is actually made).
# --------------------------------------------------------------------------
_openai_client = None
_anthropic_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        key = os.environ.get(config.OPENAI_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"Environment variable {config.OPENAI_API_KEY_ENV} is not set. "
                "Export your OpenAI API key before launching MedRAGBench."
            )
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        key = os.environ.get(config.ANTHROPIC_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"Environment variable {config.ANTHROPIC_API_KEY_ENV} is not set. "
                "Export your Anthropic API key before launching MedRAGBench."
            )
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


# --------------------------------------------------------------------------
# Embeddings (OpenAI). Anthropic has no embeddings endpoint, so this stays
# on OpenAI regardless of the chat provider.
# --------------------------------------------------------------------------
def embed_texts(texts: List[str], max_retries: int = 4) -> List[List[float]]:
    """Return one embedding vector per input string."""
    if not texts:
        return []
    client = _get_openai_client()
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            # Preserve input order.
            return [d.embedding for d in resp.data]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return []


def embed_text(text: str) -> List[float]:
    out = embed_texts([text])
    return out[0] if out else []


# --------------------------------------------------------------------------
# Chat completion. Routes to the configured provider.
# --------------------------------------------------------------------------
def chat(system: str, user: str, max_retries: int = 4) -> str:
    """Single-turn chat completion returning assistant text."""
    if config.LLM_PROVIDER == "anthropic":
        return _chat_anthropic(system, user, max_retries)
    return _chat_openai(system, user, max_retries)


def _chat_openai(system: str, user: str, max_retries: int) -> str:
    client = _get_openai_client()
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


# ==========================================================================
# >>> SWITCHING TO CLAUDE <<<
# This is the only generation function you need for Anthropic. To activate
# it, set LLM_PROVIDER = "anthropic" in config.py (or export
# MEDRAGBENCH_PROVIDER=anthropic) and set ANTHROPIC_API_KEY. The function is
# already implemented; nothing else in the pipeline changes, because every
# stage calls llm.chat(system, user).
# ==========================================================================
def _chat_anthropic(system: str, user: str, max_retries: int) -> str:
    client = _get_anthropic_client()
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=config.ANTHROPIC_CHAT_MODEL,
                max_tokens=4096,
                system=system,                       # Anthropic takes system separately
                messages=[{"role": "user", "content": user}],
            )
            # Anthropic returns a list of content blocks; concatenate text blocks.
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "".join(parts).strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""
