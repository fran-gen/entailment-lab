#!/usr/bin/env python3
"""Download and extract the SNLI dataset into data/raw/snli."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import requests

SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    shutil.unpack_archive(str(zip_path), str(output_dir))


def copy_split_files(extracted_root: Path, target_dir: Path) -> list[Path]:
    snli_root = extracted_root / "snli_1.0"
    split_files = [
        snli_root / "snli_1.0_train.jsonl",
        snli_root / "snli_1.0_dev.jsonl",
        snli_root / "snli_1.0_test.jsonl",
    ]

    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in split_files:
        if not source.exists():
            raise FileNotFoundError(f"Expected file not found after extraction: {source}")
        destination = target_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SNLI into data/raw/snli")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/snli"),
        help="Directory where SNLI split files will be copied",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if target files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    expected = [
        output_dir / "snli_1.0_train.jsonl",
        output_dir / "snli_1.0_dev.jsonl",
        output_dir / "snli_1.0_test.jsonl",
    ]

    if not args.force and all(path.exists() for path in expected):
        print("SNLI files already present. Use --force to re-download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="snli_download_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive_path = tmp_path / "snli_1.0.zip"

        print(f"Downloading SNLI from {SNLI_URL}...")
        download_file(SNLI_URL, archive_path)

        print("Extracting archive...")
        extract_zip(archive_path, tmp_path)

        print(f"Copying split files to {output_dir}...")
        copied = copy_split_files(tmp_path, output_dir)

    print("Done. Files:")
    for file in copied:
        print(f"- {file}")


if __name__ == "__main__":
    main()
