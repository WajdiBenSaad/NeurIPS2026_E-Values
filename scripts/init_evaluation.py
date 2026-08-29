#!/usr/bin/env python
from __future__ import annotations

import argparse

from evalues_routing.artifacts.progress import LanguageCheckpointWriter
from evalues_routing.pipeline import load_rows
from evalues_routing.utils.config import dump_yaml, load_config
from evalues_routing.utils.io import read_json, write_json, write_jsonl
from evalues_routing.utils.logging import configure_logging
from evalues_routing.utils.runs import new_run_dir
from evalues_routing.utils.runtime import environment_snapshot, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description='Initialize a manual per-language evaluation run.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--router', required=True, help='Frozen router produced from discovery.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    router_spec = read_json(args.router)
    if not router_spec.get('frozen', False):
        raise RuntimeError('Evaluation requires a frozen router. Run scripts/freeze_router.py first.')
    dataset = cfg['project']['dataset_name']
    if router_spec.get('dataset') not in (None, dataset):
        raise RuntimeError(f"Router dataset {router_spec.get('dataset')} does not match config {dataset}")

    seed = int(cfg['project']['seed'])
    seed_everything(seed, cfg['project'].get('deterministic', True))
    run = new_run_dir(cfg['project']['output_root'], dataset, 'evaluation', seed)
    log = configure_logging(run / 'run.log')
    dump_yaml(cfg, run / 'config.yaml')
    write_json(environment_snapshot(), run / 'environment.json')
    write_json(router_spec, run / 'selected_languages.json')
    rows, _ = load_rows(cfg, [cfg['dataset']['evaluation_split']])
    write_jsonl(rows, run / 'manifest.jsonl')
    language_codes = [language['code'] for language in cfg['dataset']['languages']]
    LanguageCheckpointWriter(run, 'evaluation', len(language_codes))
    log.info(
        'Initialized manual evaluation run: %s languages=%d examples=%d. '
        'Run one scripts/run_evaluation_language.py process per language.',
        run, len(language_codes), len(rows),
    )
    print(run)


if __name__ == '__main__':
    main()
