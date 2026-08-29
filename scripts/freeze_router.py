#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
from evalues_routing.utils.io import read_json, write_json
from evalues_routing.artifacts.validation import validate_run


def main():
    ap = argparse.ArgumentParser(description='Freeze a validated discovery router without modifying the completed run.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    report = validate_run(run, require_completed=True)
    if not report['valid'] or report['stage'] != 'discovery':
        raise RuntimeError(f'Only a valid completed discovery run can be frozen: {report}')
    src = read_json(run / 'selected_languages.json')
    frozen = dict(src)
    frozen['frozen'] = True
    frozen['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    dataset = src.get('dataset', run.parent.parent.name)
    out = Path(args.output) if args.output else Path('experiments') / dataset / f'frozen_router_{run.name}.json'
    write_json(frozen, out)
    print(out)

if __name__ == '__main__':
    main()
