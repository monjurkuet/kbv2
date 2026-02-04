#!/usr/bin/env python3
"""Preprocess YouTube transcripts into clean markdown format."""

import re
from pathlib import Path


def clean_transcript(input_file: str, output_file: str):
    """Clean and format YouTube transcript.

    Args:
        input_file: Path to input transcript file
        output_file: Path to output markdown file
    """
    with open(input_file, "r") as f:
        content = f.read()

    # Remove timestamps like [00:00:00] or 00:00:00
    content = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", content)
    content = re.sub(r"\d{2}:\d{2}:\d{2}", "", content)

    # Remove speaker labels like [Speaker: Name] or Speaker:
    content = re.sub(r"\[Speaker:\s*\w+\]", "", content)
    content = re.sub(r"^\w+:\s*", "", content, flags=re.MULTILINE)

    # Remove multiple consecutive newlines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Add title
    title = Path(input_file).stem.replace("_", " ").title()
    markdown_content = f"# {title}\n\n{content}"

    with open(output_file, "w") as f:
        f.write(markdown_content)

    print(f"Preprocessed: {input_file} -> {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess YouTube transcripts")
    parser.add_argument("input", help="Input transcript file")
    parser.add_argument("output", help="Output markdown file")

    args = parser.parse_args()

    clean_transcript(args.input, args.output)
