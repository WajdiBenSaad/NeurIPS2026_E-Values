from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any
import yaml


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML root must be a mapping: {p}')
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and recursively merge declared include configs."""
    p = Path(path).resolve()
    cfg = load_yaml(p)
    includes = cfg.pop('include_configs', {}) or {}
    merged: dict[str, Any] = {}
    for namespace, include_path in includes.items():
        ip = Path(include_path)
        if not ip.is_absolute():
            # Config paths are repository-root relative by convention.
            repo_root = p.parent.parent if p.parent.name == 'configs' else Path.cwd()
            ip = repo_root / ip
        included = load_yaml(ip)
        # Include files may already use the requested namespace at root.
        payload = included.get(namespace, included)
        merged[namespace] = _deep_merge(merged.get(namespace, {}), payload)
    merged = _deep_merge(merged, cfg)
    output_root = os.environ.get('EVALUES_OUTPUT_ROOT')
    if output_root:
        merged.setdefault('project', {})['output_root'] = output_root
    merged['_meta'] = {'config_path': str(p)}
    return merged


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
