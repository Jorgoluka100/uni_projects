"""CineIntelligence NoSQL v2 — robust TMDB document ingestion and query benchmark.

The restored notebook demonstrates document-store ideas but used brittle parsing for
JSON-like columns. This extension adds a defensive parser, row quarantine, explicit
movie-document schema, quality checks and repeatable query timing.

Expected input is the common TMDB 5000 movies CSV (or a compatible file) with fields
such as id, title, genres, keywords, production_companies, budget, revenue, runtime
and release_date. No Kaggle credentials are embedded in this repository.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd


JSONISH_COLUMNS = ["genres", "keywords", "production_companies", "production_countries", "spoken_languages"]


@dataclass(frozen=True)
class Config:
    benchmark_repeats: int = 20
    artifact_dir: Path = Path("cine_nosql_artifacts")


def parse_jsonish(value) -> list[dict]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    errors = []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            raise ValueError("expected a list")
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("could not parse JSON-like list: " + " | ".join(errors[:2]))


def normalise_name_list(items: list[dict]) -> list[str]:
    names = []
    for item in items:
        name = item.get("name")
        if name is not None and str(name).strip():
            names.append(str(name).strip())
    return sorted(dict.fromkeys(names))


def build_documents(frame: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    required = {"id", "title"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    documents = []
    quarantine = []
    for row_index, row in frame.iterrows():
        try:
            parsed = {}
            for column in JSONISH_COLUMNS:
                parsed[column] = parse_jsonish(row[column]) if column in frame else []

            movie_id = int(row["id"])
            title = str(row["title"]).strip()
            if not title:
                raise ValueError("blank title")

            release_date = pd.to_datetime(row.get("release_date"), errors="coerce")
            budget = pd.to_numeric(row.get("budget"), errors="coerce")
            revenue = pd.to_numeric(row.get("revenue"), errors="coerce")
            runtime = pd.to_numeric(row.get("runtime"), errors="coerce")
            popularity = pd.to_numeric(row.get("popularity"), errors="coerce")
            vote_average = pd.to_numeric(row.get("vote_average"), errors="coerce")

            doc = {
                "_id": movie_id,
                "title": title,
                "overview": None if pd.isna(row.get("overview")) else str(row.get("overview")),
                "release_year": None if pd.isna(release_date) else int(release_date.year),
                "genres": normalise_name_list(parsed["genres"]),
                "keywords": normalise_name_list(parsed["keywords"]),
                "production_companies": normalise_name_list(parsed["production_companies"]),
                "production_countries": normalise_name_list(parsed["production_countries"]),
                "spoken_languages": normalise_name_list(parsed["spoken_languages"]),
                "budget": None if pd.isna(budget) else float(budget),
                "revenue": None if pd.isna(revenue) else float(revenue),
                "runtime": None if pd.isna(runtime) else float(runtime),
                "popularity": None if pd.isna(popularity) else float(popularity),
                "vote_average": None if pd.isna(vote_average) else float(vote_average),
            }
            documents.append(doc)
        except Exception as exc:
            quarantine.append({"row_index": int(row_index), "error": str(exc), "raw_id": row.get("id")})
    return documents, pd.DataFrame(quarantine)


def validate_documents(documents: list[dict]) -> dict[str, int]:
    ids = [doc["_id"] for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate movie _id after ingestion")
    if any(not doc["title"] for doc in documents):
        raise ValueError("blank title in document set")
    negative_money = sum(
        1
        for doc in documents
        for key in ("budget", "revenue")
        if doc[key] is not None and doc[key] < 0
    )
    if negative_money:
        raise ValueError("negative budget/revenue values found")
    return {
        "documents": len(documents),
        "unique_ids": len(set(ids)),
        "documents_with_genres": sum(bool(doc["genres"]) for doc in documents),
    }


def build_inverted_index(documents: list[dict], field: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for doc in documents:
        for value in doc.get(field, []):
            index.setdefault(value.lower(), []).append(doc["_id"])
    return index


def query_genre_scan(documents: list[dict], genre: str) -> list[int]:
    target = genre.lower()
    return [doc["_id"] for doc in documents if any(g.lower() == target for g in doc["genres"])]


def benchmark_query(documents: list[dict], index: dict[str, list[int]], genre: str, repeats: int) -> dict[str, float]:
    scan_times = []
    index_times = []
    scan_result = query_genre_scan(documents, genre)
    indexed_result = index.get(genre.lower(), [])
    if set(scan_result) != set(indexed_result):
        raise AssertionError("index query and full scan disagree")

    for _ in range(repeats):
        t0 = time.perf_counter()
        query_genre_scan(documents, genre)
        scan_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        index.get(genre.lower(), [])
        index_times.append((time.perf_counter() - t0) * 1000)

    return {
        "scan_median_ms": float(median(scan_times)),
        "index_median_ms": float(median(index_times)),
        "matching_movies": int(len(scan_result)),
    }


def run(csv_path: Path, config: Config = Config()) -> dict[str, object]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path, low_memory=False)
    documents, quarantine = build_documents(frame)
    quality = validate_documents(documents)
    genre_index = build_inverted_index(documents, "genres")

    genre_counts = sorted(((genre, len(ids)) for genre, ids in genre_index.items()), key=lambda x: (-x[1], x[0]))
    benchmark = None
    if genre_counts:
        benchmark = {
            "genre": genre_counts[0][0],
            **benchmark_query(documents, genre_index, genre_counts[0][0], config.benchmark_repeats),
        }

    with (config.artifact_dir / "movies.ndjson").open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    quarantine.to_csv(config.artifact_dir / "quarantine.csv", index=False)

    payload = {
        "config": {**asdict(config), "artifact_dir": str(config.artifact_dir)},
        "source_file": str(csv_path),
        "raw_rows": int(len(frame)),
        "quality": quality,
        "quarantined_rows": int(len(quarantine)),
        "genre_query_benchmark": benchmark,
        "document_schema": {
            "_id": "unique integer movie id",
            "title": "non-empty string",
            "genres/keywords/production_*": "normalised arrays of names",
            "budget/revenue/runtime": "nullable numeric values",
        },
        "limitations": [
            "This benchmark is an in-memory index demonstration, not a production MongoDB benchmark.",
            "TMDB source/licensing and freshness must be documented separately for any public release of derived data.",
        ],
    }
    (config.artifact_dir / "ingestion_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=Path("cine_nosql_artifacts"))
    args = parser.parse_args()
    print(json.dumps(run(args.csv_path, Config(artifact_dir=args.artifact_dir)), indent=2))


if __name__ == "__main__":
    main()
