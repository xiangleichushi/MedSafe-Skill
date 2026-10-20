#!/usr/bin/env python
"""Split a benchmark into inference-safe queries and a hidden evaluation key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PUBLIC_FIELDS = ("id", "query")


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public = []
    key = []
    seen_ids = set()
    for index, item in enumerate(records, start=1):
        qid = item.get("id", index)
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"Missing query text for id={qid}")
        if qid in seen_ids:
            raise ValueError(f"Duplicate query id: {qid}")
        seen_ids.add(qid)
        public.append({"id": qid, "query": query})
        key.append({key: value for key, value in item.items() if key not in PUBLIC_FIELDS})
        key[-1] = {"id": qid, **key[-1]}
    return public, key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public, key = split_records(load_json(args.input))
    save_json(args.public_output, public)
    save_json(args.key_output, key)
    print(json.dumps({
        "input_count": len(public),
        "public_output": str(args.public_output),
        "key_output": str(args.key_output),
        "public_fields": list(PUBLIC_FIELDS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
