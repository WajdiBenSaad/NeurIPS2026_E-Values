from __future__ import annotations

from pathlib import Path
import yaml
from evalues_routing.utils.io import read_json, read_jsonl, checksums


BASE_REQUIRED = {
    'config.yaml', 'environment.json', 'manifest.jsonl', 'translations.jsonl',
    'predictions.jsonl', 'selected_languages.json', 'metrics.json', 'latency.json',
    'checksums.json', 'run.log'
}


def _split_checks(run: Path, stage: str) -> list[str]:
    problems: list[str] = []
    try:
        cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
        manifest = read_jsonl(run / 'manifest.jsonl')
        observed = {r.get('split') for r in manifest}
        if stage == 'discovery':
            expected = set(cfg['dataset']['discovery_splits'])
            forbidden = cfg['dataset']['evaluation_split']
            if forbidden in observed:
                problems.append(f'discovery manifest contains evaluation split {forbidden}')
            if not observed.issubset(expected):
                problems.append(f'discovery splits {sorted(observed)} not subset of {sorted(expected)}')
        elif stage == 'evaluation':
            expected = {cfg['dataset']['evaluation_split']}
            if observed != expected:
                problems.append(f'evaluation splits {sorted(observed)} != {sorted(expected)}')
    except Exception as exc:
        problems.append(f'split validation failed: {exc}')
    return problems


def _artifact_consistency(run: Path) -> list[str]:
    problems: list[str] = []
    try:
        manifest = read_jsonl(run / 'manifest.jsonl')
        predictions = read_jsonl(run / 'predictions.jsonl')
        translations = read_jsonl(run / 'translations.jsonl')
        mids = [r['example_id'] for r in manifest]
        pids = [r['example_id'] for r in predictions]
        tids = [r['example_id'] for r in translations]
        if len(mids) != len(set(mids)):
            problems.append('manifest contains duplicate example_id values')
        if set(mids) != set(pids):
            problems.append('prediction example IDs do not match manifest')
        if set(mids) != set(tids):
            problems.append('translation example IDs do not match manifest')
    except Exception as exc:
        problems.append(f'artifact consistency validation failed: {exc}')
    return problems


def validate_run(run_dir: str | Path, require_completed: bool = True) -> dict:
    run = Path(run_dir)
    stage = run.parent.name
    required = set(BASE_REQUIRED)
    if stage == 'discovery':
        required.add('eprocess_trajectories.jsonl')
    if require_completed:
        required.add('COMPLETED')

    present = {p.name for p in run.iterdir() if p.is_file()}
    missing = sorted(required - present)
    checksum_ok = False
    checksum_mismatches: list[str] = []
    checksum_missing: list[str] = []
    cpath = run / 'checksums.json'
    if cpath.exists():
        stored = read_json(cpath)
        actual = checksums(run)
        checksum_mismatches = sorted(k for k, v in stored.items() if actual.get(k) != v)
        checksum_missing = sorted(k for k in actual if k not in stored)
        checksum_ok = not checksum_mismatches and not checksum_missing

    split_problems = _split_checks(run, stage) if not missing else []
    consistency_problems = _artifact_consistency(run) if not missing else []
    valid = not missing and checksum_ok and not split_problems and not consistency_problems
    return {
        'run_dir': str(run),
        'stage': stage,
        'missing_required_files': missing,
        'checksum_ok': checksum_ok,
        'checksum_mismatches': checksum_mismatches,
        'checksum_untracked_files': checksum_missing,
        'split_problems': split_problems,
        'consistency_problems': consistency_problems,
        'valid': valid,
    }
