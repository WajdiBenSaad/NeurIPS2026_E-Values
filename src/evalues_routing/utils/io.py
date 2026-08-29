from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Any


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(p)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp.replace(p)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def checksums(directory: str | Path, exclude: set[str] | None = None) -> dict[str, str]:
    root = Path(directory)
    excluded = exclude or {'checksums.json', 'validation_report.json', 'COMPLETED'}
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in excluded:
            result[str(p.relative_to(root))] = sha256_file(p)
    return result
