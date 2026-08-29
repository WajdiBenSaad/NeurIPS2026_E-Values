#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
from pathlib import Path

from evalues_routing.utils.config import load_config
from evalues_routing.pipeline import load_rows
from evalues_routing.utils.io import write_jsonl, write_json


def main():
    ap = argparse.ArgumentParser(description='Download/normalize configured dataset splits without model inference.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    splits = list(dict.fromkeys(cfg['dataset']['discovery_splits'] + [cfg['dataset']['evaluation_split']]))
    rows, labels = load_rows(cfg, splits)
    cache_root = Path(os.environ.get('EVALUES_CACHE_ROOT', '.cache/evalues-routing'))
    out = Path(args.output) if args.output else cache_root / 'prepared' / f"{cfg['project']['dataset_name']}_manifest.jsonl"
    write_jsonl(rows, out)
    write_json({'labels': labels, 'n_rows': len(rows), 'splits': splits}, out.with_suffix('.meta.json'))
    print(out)

if __name__ == '__main__':
    main()
