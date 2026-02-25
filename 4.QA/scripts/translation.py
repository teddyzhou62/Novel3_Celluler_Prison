import argparse
import os
from pathlib import Path

def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return p.read_text(encoding="utf-8")

def optional_read(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_file", default="")
    ap.add_argument("--glossary", default="")
    ap.add_argument("--style", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--cast", default="")
    ap.add_argument("--world_items", default="")
    ap.add_argument("--world_lore", default="")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    src_text = read_text(args.src)
    glossary = optional_read(args.glossary) if args.glossary else ""
    style = optional_read(args.style) if args.style else ""
    names = optional_read(args.names) if args.names else ""
    cast = optional_read(args.cast) if args.cast else ""
    world_items = optional_read(args.world_items) if args.world_items else ""
    world_lore = optional_read(args.world_lore) if args.world_lore else ""

    # ---- OpenAI call (example using openai SDK v1) ----
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = (
        "You are a professional novel localizer. Produce final publish-ready English.\n"
        "English-first readability. Keep Chinese characteristics only when they add meaning.\n"
        "Follow glossary + style guide strictly.\n"
    )

    user = f"""
STYLE GUIDE:
{style}

GLOSSARY (enforce terms):
{glossary}

NAME RULES:
{names}

CAST / VOICE:
{cast}

WORLD ITEMS:
{world_items}

WORLD LORE:
{world_lore}

TASK:
Translate and localize the following Chinese chapter into final English prose.
- Output markdown.
- Keep paragraphs readable.
- Natural dialogue.
- No explanations, no notes—only the chapter text.

CHAPTER:
{src_text}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # swap later
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )

    out_text = resp.choices[0].message.content.strip()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_file.strip():
        out_name = args.out_file.strip()
    else:
        out_name = Path(args.src).name  # reuse basename

    out_path = out_dir / out_name
    out_path.write_text(out_text + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
