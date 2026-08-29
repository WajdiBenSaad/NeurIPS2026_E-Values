#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from evalues_routing.utils.io import read_jsonl, read_json, write_json
from evalues_routing.routing.fixed_router import FixedLanguageRouter
from evalues_routing.routing.baseline_routers import DirectOnlyRouter, TranslationOnlyRouter, SelectedLanguageRouter
from evalues_routing.evaluation.policy_evaluation import evaluate_policy


def main():
    ap = argparse.ArgumentParser(description='Recompute counterfactual policy latency from stored per-path measurements; no model inference.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    preds = read_jsonl(run / 'predictions.jsonl')
    spec = read_json(run / 'selected_languages.json')
    eligible = {r['language_code'] for r in preds if r.get('translate_eligible', True)}
    policies = [
        ('direct_only', DirectOnlyRouter()),
        ('translation_only', TranslationOnlyRouter(eligible)),
        ('fixed_list', FixedLanguageRouter(spec.get('fixed_selected_languages', []))),
        ('p_value', SelectedLanguageRouter(spec.get('p_selected_languages', []))),
        ('e_value', SelectedLanguageRouter(spec.get('e_selected_languages', []))),
    ]
    result = {}
    for name, router in policies:
        m, _ = evaluate_policy(preds, router, name)
        result[name] = m['latency']
    out = Path(args.output) if args.output else Path('reports/generated') / f'{run.name}_latency_recomputed.json'
    write_json(result, out)
    print(out)

if __name__ == '__main__':
    main()
