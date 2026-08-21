"""Download, fingerprint and extract the pinned Olist dataset."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import EXPECTED_FILE_SHA256, FILE_TABLES, ProjectConfig


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def verify_archive(config: ProjectConfig) -> None:
    actual = sha256_file(config.archive_path)
    if actual != config.archive_sha256:
        raise ValueError(
            "Downloaded archive hash does not match the retained dataset-v7 fingerprint: "
            f"expected {config.archive_sha256}, got {actual}"
        )


def extract_and_verify(config: ProjectConfig) -> dict[str, str]:
    """Extract the expected CSVs and fail if any file differs from the verified source."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(config.archive_path) as archive:
        archive_names = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".csv")}
        missing = sorted(set(FILE_TABLES) - set(archive_names))
        if missing:
            raise ValueError(f"Archive is missing expected CSV files: {missing}")
        for filename in FILE_TABLES:
            destination = config.data_dir / filename
            with archive.open(archive_names[filename]) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)

    hashes: dict[str, str] = {}
    for filename, expected in EXPECTED_FILE_SHA256.items():
        path = config.data_dir / filename
        actual = sha256_file(path)
        hashes[filename] = actual
        if actual != expected:
            raise ValueError(f"Source fingerprint mismatch for {filename}: expected {expected}, got {actual}")
    return hashes


def ensure_dataset(config: ProjectConfig) -> dict[str, str]:
    """Make the verified dataset available locally, downloading it only when required."""
    config.validate()
    if not config.archive_path.exists():
        print("Downloading pinned Olist dataset version 7...")
        _download(config.dataset_url, config.archive_path)
    verify_archive(config)

    all_extracted = all((config.data_dir / filename).exists() for filename in FILE_TABLES)
    if all_extracted:
        hashes = {filename: sha256_file(config.data_dir / filename) for filename in FILE_TABLES}
        if hashes == EXPECTED_FILE_SHA256:
            return hashes
    return extract_and_verify(config)
