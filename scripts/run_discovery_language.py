#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from evalues_routing.artifacts.progress import LanguageCheckpointWriter
from evalues_routing.pipeline import infer_both_paths
from evalues_routing.utils.io import read_json, read_jsonl
from evalues_routing.utils.logging import configure_logging
from evalues_routing.utils.runs import assert_not_completed
from evalues_routing.utils.runtime import release_accelerator_memory, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description='Run exactly one discovery language, checkpoint it, and exit.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--language-code', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    assert_not_completed(run)
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    language_order = [language['code'] for language in cfg['dataset']['languages']]
    if args.language_code not in language_order:
        raise ValueError(f'Unknown language code {args.language_code}; expected one of {language_order}')
    progress_data = read_json(run / 'progress.json')
    completed = {entry['language_code'] for entry in progress_data.get('completed_languages', [])}
    if args.language_code in completed:
        print(f'Language already checkpointed; nothing to do: {args.language_code}')
        return

    manifest = read_jsonl(run / 'manifest.jsonl')
    rows = [row for row in manifest if row['language_code'] == args.language_code]
    if not rows:
        raise RuntimeError(f'No manifest rows found for {args.language_code}')
    labels = cfg['dataset'].get('labels') or sorted({row['gold_label'] for row in manifest})
    seed_everything(int(cfg['project']['seed']), cfg['project'].get('deterministic', True))
    log = configure_logging(run / 'run.log')
    progress = LanguageCheckpointWriter(run, 'discovery', len(language_order), resume=True)
    log.info('Launching isolated language subprocess: %s', args.language_code)
    try:
        infer_both_paths(cfg, rows, labels, log, progress, language_order=language_order)
    except BaseException as exc:
        language = next(item['name'] for item in cfg['dataset']['languages'] if item['code'] == args.language_code)
        progress.language_interrupted(
            language_order.index(args.language_code) + 1,
            args.language_code,
            language,
            len(rows),
            f'{type(exc).__name__}: {exc}',
        )
        raise
    finally:
        release_accelerator_memory()
        log.info('Language subprocess exiting; OS will release all RAM/MPS allocations: %s', args.language_code)


if __name__ == '__main__':
    main()
