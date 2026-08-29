#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from evalues_routing.artifacts.checkpoint import finalize_run
from evalues_routing.artifacts.progress import LanguageCheckpointWriter
from evalues_routing.evaluation.latency import latency_summary
from evalues_routing.pipeline import aggregate_discovery_metrics
from evalues_routing.routing.eprocess_router import discover_languages
from evalues_routing.utils.io import read_jsonl, write_json, write_jsonl
from evalues_routing.utils.logging import configure_logging
from evalues_routing.utils.runs import assert_not_completed


def main() -> None:
    parser = argparse.ArgumentParser(description='Assemble and seal a completed manual discovery run.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    assert_not_completed(run)
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    languages = cfg['dataset']['languages']
    predictions = []
    translations = []
    missing = []
    for index, language in enumerate(languages, start=1):
        checkpoint = run / 'language_checkpoints' / f"{index:02d}_{language['code']}"
        prediction_path = checkpoint / 'predictions.jsonl'
        translation_path = checkpoint / 'translations.jsonl'
        summary_path = checkpoint / 'summary.json'
        if not prediction_path.exists() or not translation_path.exists() or not summary_path.exists():
            missing.append(language['code'])
            continue
        predictions.extend(read_jsonl(prediction_path))
        translations.extend(read_jsonl(translation_path))
    if missing:
        raise RuntimeError(f'Cannot finalize; missing language checkpoints: {missing}')

    write_jsonl(translations, run / 'translations.jsonl')
    write_jsonl(predictions, run / 'predictions.jsonl')
    discovered, trajectories = discover_languages(predictions, cfg['statistics'])
    write_jsonl(trajectories, run / 'eprocess_trajectories.jsonl')
    e_selected = sorted(code for code, result in discovered.items() if result['e_selected'])
    p_selected = sorted(code for code, result in discovered.items() if result['p_selected'])
    write_json({
        'dataset': cfg['project']['dataset_name'],
        'source_run': str(run),
        'alpha': cfg['statistics']['paired_test']['alpha'],
        'e_threshold': cfg['statistics']['eprocess']['decision_threshold'],
        'e_selected_languages': e_selected,
        'p_selected_languages': p_selected,
        'fixed_selected_languages': cfg.get('fixed_router', {}).get('translated_languages', []),
        'language_statistics': discovered,
        'frozen': False,
    }, run / 'selected_languages.json')
    metrics = aggregate_discovery_metrics(predictions)
    metrics['language_statistics'] = discovered
    write_json(metrics, run / 'metrics.json')
    write_json({
        'direct': latency_summary(predictions),
        'translated_path_mean_s': sum(float(row['translated_latency_s']) for row in predictions) / len(predictions),
    }, run / 'latency.json')
    progress = LanguageCheckpointWriter(run, 'discovery', len(languages), resume=True)
    progress.finalize()
    log = configure_logging(run / 'run.log')
    log.info('Completed manual discovery run: %s', run)
    finalize_run(run)
    print(run)


if __name__ == '__main__':
    main()
