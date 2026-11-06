#!/usr/bin/env python3
"""CLI runner for Ollama-backed Distilly profile distillation.

Usage:
    python -I tools/ollama_distill.py \
        --source path/to/emails/ \
        --character colleague \
        --name "Jane Smith" \
        --slug jane-smith \
        --model glm-5.3-flash:cloud

Reads source material, runs it through the analyzer→builder prompt pipeline
via a local Ollama instance, and writes the SKILL.md artifact.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import mailbox
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    analyze_and_build,
    load_prompt,
    ollama_chat,
)
from skill_writer import create_skill
from skill_presets import get_character_preset, normalize_character


def read_mbox(path: Path, max_messages: int = 200) -> str:
    """Extract text bodies from an mbox file."""
    mbox = mailbox.mbox(str(path))
    parts = []
    for i, msg in enumerate(mbox):
        if i >= max_messages:
            break
        subject = msg.get("subject", "(no subject)")
        sender = msg.get("from", "unknown")
        date = msg.get("date", "")
        body = _extract_body(msg)
        if body.strip():
            parts.append(f"--- Email {i+1} ---\nFrom: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}\n")
    return "\n".join(parts)


def read_eml(path: Path) -> str:
    """Extract text body from a single .eml file."""
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    subject = msg.get("subject", "(no subject)")
    sender = msg.get("from", "unknown")
    date = msg.get("date", "")
    body = _extract_body(msg)
    return f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body}\n"


def _extract_body(msg: email.message.Message) -> str:
    """Get plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        return ""
    payload = msg.get_payload(decode=True)
    if payload:
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return ""


def read_text_files(directory: Path) -> str:
    """Read all .txt and .md files from a directory."""
    parts = []
    for ext in ("*.txt", "*.md"):
        for f in sorted(directory.glob(ext)):
            parts.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')}\n")
    return "\n".join(parts)


def load_source_material(source: Path, max_messages: int = 200) -> str:
    """Auto-detect source format and load content."""
    if source.is_file():
        if source.suffix == ".mbox":
            return read_mbox(source, max_messages)
        elif source.suffix == ".eml":
            return read_eml(source)
        else:
            return source.read_text(encoding="utf-8", errors="replace")
    elif source.is_dir():
        parts = []
        for eml in sorted(source.glob("*.eml")):
            parts.append(read_eml(eml))
        for mbox in sorted(source.glob("*.mbox")):
            parts.append(read_mbox(mbox, max_messages))
        if not parts:
            parts.append(read_text_files(source))
        return "\n".join(parts)
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def run_merger(
    character: str,
    name: str,
    work_content: str,
    persona_content: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """Run the merger prompt to polish and integrate both halves.

    Returns (merged_work, merged_persona). If no merger prompt exists,
    returns the inputs unchanged.
    """
    preset = get_character_preset(character)
    merger_path = preset["prompt_bundle"].get("merger")
    if not merger_path:
        return work_content, persona_content

    merger_prompt = load_prompt(merger_path, REPO_ROOT)
    user_input = (
        f"# Subject: {name}\n\n"
        f"## Work Content\n\n{work_content}\n\n"
        f"## Persona Content\n\n{persona_content}"
    )

    print("  [5/5] Running merger...", file=sys.stderr)
    merged = ollama_chat(
        merger_prompt.replace("{name}", name),
        user_input,
        model=model,
        base_url=base_url,
    )

    if "## PART A" in merged and "## PART B" in merged:
        parts = merged.split("## PART B")
        merged_work = parts[0].replace("## PART A", "").strip()
        merged_persona = parts[1].strip() if len(parts) > 1 else persona_content
    else:
        merged_work = work_content
        merged_persona = persona_content

    return merged_work, merged_persona


def main():
    parser = argparse.ArgumentParser(
        description="Run Distilly person-profile distillation via local Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Path to emails/docs (file or directory)")
    parser.add_argument("--character", default="colleague", help="Character family (default: colleague)")
    parser.add_argument("--name", required=True, help="Subject's display name")
    parser.add_argument("--slug", help="Skill slug (auto-generated from name if omitted)")
    parser.add_argument("--model", default=None, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=None, help=f"Ollama base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: skills/<character>)")
    parser.add_argument("--max-messages", type=int, default=200, help="Max emails to process (default: 200)")
    parser.add_argument("--skip-merger", action="store_true", help="Skip the merger step")
    parser.add_argument("--description", default=None, help="Skill description")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    character = normalize_character(args.character)

    if not args.slug:
        from skill_writer import slugify
        args.slug = slugify(args.name)

    print(f"Distilly — Ollama-backed distillation", file=sys.stderr)
    print(f"  Subject: {args.name} ({args.slug})", file=sys.stderr)
    print(f"  Character: {character}", file=sys.stderr)
    print(f"  Source: {source}", file=sys.stderr)
    print(f"  Model: {args.model or DEFAULT_MODEL}", file=sys.stderr)
    print(file=sys.stderr)

    print("Loading source material...", file=sys.stderr)
    material = load_source_material(source, args.max_messages)
    if not material.strip():
        print("ERROR: No source material found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(material):,} chars from source", file=sys.stderr)
    print(file=sys.stderr)

    print("Running analysis pipeline...", file=sys.stderr)
    results = analyze_and_build(
        character, args.name, material,
        model=args.model, base_url=args.ollama_url, repo_root=REPO_ROOT,
    )

    work_content = results["work_content"]
    persona_content = results["persona_content"]

    if not args.skip_merger:
        work_content, persona_content = run_merger(
            character, args.name, work_content, persona_content,
            model=args.model, base_url=args.ollama_url,
        )

    preset = get_character_preset(character)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / preset["storage_root"]
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": args.name,
        "display_name": args.name,
        "character": character,
        "description": args.description or f"Person profile for {args.name}",
        "generation": {
            "model": args.model or DEFAULT_MODEL,
            "engine": "ollama-local",
            "source_chars": len(material),
        },
    }

    print(f"\nWriting skill artifacts...", file=sys.stderr)
    skill_dir = create_skill(output_dir, args.slug, meta, work_content, persona_content)
    print(f"  Written to: {skill_dir}", file=sys.stderr)

    analysis_dir = skill_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "work_analysis.md").write_text(results["work_analysis"], encoding="utf-8")
    (analysis_dir / "persona_analysis.md").write_text(results["persona_analysis"], encoding="utf-8")

    print(f"\nDone. Skill created at {skill_dir}", file=sys.stderr)
    print(f"  SKILL.md: {skill_dir / 'SKILL.md'}", file=sys.stderr)
    print(f"  Analysis saved to: {analysis_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
