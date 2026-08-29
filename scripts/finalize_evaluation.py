#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from evalues_routing.artifacts.checkpoint import finalize_run
from evalues_routing.artifacts.progress import LanguageCheckpointWriter
from evalues_routing.evaluation.policy_evaluation import evaluate_policy
from evalues_routing.routing.baseline_routers import DirectOnlyRouter, SelectedLanguageRouter, TranslationOnlyRouter
from evalues_routing.routing.fixed_router import FixedLanguageRouter
from evalues_routing.utils.io import read_json, read_jsonl, write_json, write_jsonl
from evalues_routing.utils.logging import configure_logging
from evalues_routing.utils.runs import assert_not_completed


def main() -> None:
    parser = argparse.ArgumentParser(description='Assemble and seal a completed manual evaluation run.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    assert_not_completed(run)
    cfg = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    router_spec = read_json(run / 'selected_languages.json')
    if not router_spec.get('frozen', False):
        raise RuntimeError('The evaluation run does not contain a frozen router.')

    predictions: list[dict] = []
    translations: list[dict] = []
    missing: list[str] = []
    languages = cfg['dataset']['languages']
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
    eligible = {row['language_code'] for row in predictions if row.get('translate_eligible', True)}
    policies = [
        ('direct_only', DirectOnlyRouter()),
        ('translation_only', TranslationOnlyRouter(eligible)),
        ('fixed_list', FixedLanguageRouter(cfg.get('fixed_router', {}).get('translated_languages', []))),
        ('p_value', SelectedLanguageRouter(router_spec.get('p_selected_languages', []))),
        ('e_value', SelectedLanguageRouter(router_spec.get('e_selected_languages', []))),
    ]
    all_metrics = {}
    for name, router in policies:
        metrics, _ = evaluate_policy(predictions, router, name)
        all_metrics[name] = metrics
    write_json({'policies': all_metrics}, run / 'metrics.json')
    write_json({name: metrics['latency'] for name, metrics in all_metrics.items()}, run / 'latency.json')
    LanguageCheckpointWriter(run, 'evaluation', len(languages), resume=True).finalize()
    log = configure_logging(run / 'run.log')
    log.info('Completed manual evaluation run: %s', run)
    finalize_run(run)
    print(run)


if __name__ == '__main__':
    main()
