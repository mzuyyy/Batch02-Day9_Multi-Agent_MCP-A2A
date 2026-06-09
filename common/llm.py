"""Shared LLM factory for all agents.

Uses NVIDIA NIM as an OpenAI-compatible API, so the rest of the code can
continue using LangChain's ChatOpenAI integration.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at NVIDIA NIM."""
    return ChatOpenAI(
        model=os.getenv("NVIDIA_MODEL", "z-ai/glm-5.1"),
        openai_api_key=os.getenv("NVIDIA_API_KEY"),
        openai_api_base=os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
        temperature=0.3,
    )
