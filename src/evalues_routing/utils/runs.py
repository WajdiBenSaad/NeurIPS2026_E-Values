from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import secrets


def new_run_dir(output_root: str | Path, dataset_name: str, stage: str, seed: int) -> Path:
    base = Path(output_root) / dataset_name / stage
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
    suffix = secrets.token_hex(2)
    run = base / f'run_{stamp}_seed-{seed}_{suffix}'
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=False)
    return run


def assert_not_completed(run_dir: str | Path) -> None:
    marker = Path(run_dir) / 'COMPLETED'
    if marker.exists():
        raise RuntimeError(f'Run is immutable because COMPLETED exists: {run_dir}')


def mark_completed(run_dir: str | Path) -> None:
    (Path(run_dir) / 'COMPLETED').write_text('completed\n', encoding='utf-8')
