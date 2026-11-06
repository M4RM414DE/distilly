#!/usr/bin/env python3
"""Ollama-backed inference client for Distilly's prompt pipeline.

Reads analyzer/builder prompts as system instructions, feeds source material
through them via Ollama's /api/chat endpoint, and returns the generated text.

No cloud tokens burned. Requires a running Ollama instance with a capable model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "glm-5.3-flash:cloud"


def ollama_chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
    num_ctx: int = 32768,
) -> str:
    """Send a single system+user turn to Ollama and return the assistant text."""
    base = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = model or os.environ.get("DISTILLY_MODEL") or DEFAULT_MODEL

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }).encode()

    req = Request(f"{base}/api/chat", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed ({base}): {exc}") from exc

    return data["message"]["content"]


def load_prompt(prompt_path: str, repo_root: Path | None = None) -> str:
    """Load a prompt markdown file, resolving relative to the repo root."""
    root = repo_root or Path(__file__).resolve().parent.parent
    path = root / prompt_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def run_pipeline_step(
    prompt_path: str,
    source_material: str,
    *,
    name: str = "the subject",
    model: str | None = None,
    base_url: str | None = None,
    repo_root: Path | None = None,
) -> str:
    """Run one pipeline step: load prompt as system, source material as user."""
    system = load_prompt(prompt_path, repo_root)
    system = system.replace("{name}", name)
    user = f"Source material for {name}:\n\n{source_material}"
    return ollama_chat(system, user, model=model, base_url=base_url)


def analyze_and_build(
    character: str,
    name: str,
    source_material: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Run the full analyze→build pipeline for both work and persona tracks.

    Returns {"work_analysis", "persona_analysis", "work_content", "persona_content"}.
    """
    kwargs = dict(name=name, model=model, base_url=base_url, repo_root=repo_root)

    print(f"  [1/4] Analyzing work capabilities...", file=sys.stderr)
    work_analysis = run_pipeline_step(
        "prompts/shared/work_analyzer.md", source_material, **kwargs
    )

    print(f"  [2/4] Analyzing persona...", file=sys.stderr)
    persona_analysis = run_pipeline_step(
        f"prompts/{character}/persona_analyzer.md", source_material, **kwargs
    )

    print(f"  [3/4] Building work skill...", file=sys.stderr)
    work_content = run_pipeline_step(
        "prompts/shared/work_builder.md", work_analysis, **kwargs
    )

    print(f"  [4/4] Building persona skill...", file=sys.stderr)
    persona_content = run_pipeline_step(
        f"prompts/{character}/persona_builder.md", persona_analysis, **kwargs
    )

    return {
        "work_analysis": work_analysis,
        "persona_analysis": persona_analysis,
        "work_content": work_content,
        "persona_content": persona_content,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Ollama inference for Distilly")
    parser.add_argument("--model", help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", help=f"Ollama URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--prompt", help="Test a single prompt file")
    parser.add_argument("--input", help="Input text or file path")
    args = parser.parse_args()

    if args.prompt and args.input:
        text = Path(args.input).read_text() if Path(args.input).exists() else args.input
        result = run_pipeline_step(args.prompt, text, model=args.model, base_url=args.base_url)
        print(result)
    else:
        result = ollama_chat(
            "You are a helpful assistant.",
            "Say 'Ollama inference working' and nothing else.",
            model=args.model,
            base_url=args.base_url,
        )
        print(result)
