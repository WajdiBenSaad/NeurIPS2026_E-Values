#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from evalues_routing.artifacts.validation import validate_run
from evalues_routing.statistics.operating_characteristics import (
    exact_test_critical_wins,
    simulate_crossing_times,
    summarize_crossings,
    wilson_interval,
)
from evalues_routing.utils.io import read_jsonl, write_json


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def paired_counts(run_dir: Path) -> list[dict]:
    router = yaml.safe_load((run_dir / 'selected_languages.json').read_text(encoding='utf-8'))
    observed_statistics = router.get('language_statistics', {})
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(run_dir / 'predictions.jsonl'):
        if row.get('translate_eligible', True):
            by_language[str(row['language_code'])].append(row)

    records = []
    for code, rows in sorted(by_language.items()):
        fixes = sum((not bool(row['direct_correct'])) and bool(row['translated_correct']) for row in rows)
        regressions = sum(bool(row['direct_correct']) and (not bool(row['translated_correct'])) for row in rows)
        discordances = int(fixes + regressions)
        records.append({
            'language_code': code,
            'language': rows[0].get('language', code),
            'n_examples': len(rows),
            'fixes': int(fixes),
            'regressions': int(regressions),
            'discordances': discordances,
            'q_hat': fixes / discordances if discordances else np.nan,
            'discordance_rate': discordances / len(rows) if rows else np.nan,
            'observed_e_selected': bool(observed_statistics.get(code, {}).get('e_selected', False)),
            'observed_crossing_discordances': observed_statistics.get(code, {}).get(
                'first_crossing_discordant_index'
            ),
        })
    return records


def build_synthetic(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    section = cfg['synthetic']
    repetitions = int(section['repetitions'])
    horizon = int(section['horizon'])
    budgets = sorted(set(map(int, section['reporting_budgets'])))
    null_p = float(cfg['null_probability'])
    null_probabilities = list(map(float, section['null_probabilities']))
    if null_p not in null_probabilities:
        raise ValueError('synthetic null probabilities must include the boundary null')
    probabilities = [*null_probabilities, *map(float, section['alternative_probabilities'])]
    thresholds = list(map(float, cfg['thresholds']))
    confidence = float(cfg['confidence_level'])
    alpha = float(cfg['primary_alpha'])
    records = []

    if budgets[-1] > horizon:
        raise ValueError('a reporting budget exceeds the synthetic horizon')
    for q in probabilities:
        print(f'[synthetic] q={q:.3f}, repetitions={repetitions}, horizon={horizon}', flush=True)
        crossing_times, _ = simulate_crossing_times(
            q=q, horizon=horizon, thresholds=thresholds, repetitions=repetitions,
            rng=rng, null_p=null_p,
        )
        for budget in budgets:
            critical = exact_test_critical_wins(budget, alpha, null_p)
            fixed_rejections = int((rng.binomial(budget, q, size=repetitions) >= critical).sum())
            fixed_low, fixed_high = wilson_interval(fixed_rejections, repetitions, confidence)
            for threshold in thresholds:
                records.append({
                    'q': q,
                    'is_null': bool(q <= null_p),
                    'is_null_boundary': bool(q == null_p),
                    'discordance_budget': budget,
                    'threshold': threshold,
                    **summarize_crossings(
                        crossing_times[threshold], horizon=budget, repetitions=repetitions,
                        confidence_level=confidence,
                    ),
                    'fixed_exact_rejection_rate': fixed_rejections / repetitions,
                    'fixed_exact_rate_ci_low': fixed_low,
                    'fixed_exact_rate_ci_high': fixed_high,
                    'fixed_exact_critical_wins': critical,
                })
    return pd.DataFrame(records)


def build_discovery_calibrated(
    datasets: dict[str, Path], cfg: dict, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repetitions = int(cfg['discovery_calibrated']['repetitions'])
    null_p = float(cfg['null_probability'])
    thresholds = list(map(float, cfg['thresholds']))
    confidence = float(cfg['confidence_level'])
    count_records, language_records = [], []
    router_null_paths: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)

    for dataset, run_dir in datasets.items():
        for row in paired_counts(run_dir):
            count_records.append({'dataset': dataset, **row})
            horizon = int(row['discordances'])
            if horizon == 0:
                continue
            print(
                f"[resampling] {dataset}/{row['language_code']}: "
                f"q_hat={row['q_hat']:.3f}, discordances={horizon}", flush=True,
            )
            plug_in_paths, _ = simulate_crossing_times(
                q=float(row['q_hat']), horizon=horizon, thresholds=thresholds,
                repetitions=repetitions, rng=rng, null_p=null_p,
            )
            null_paths, _ = simulate_crossing_times(
                q=null_p, horizon=horizon, thresholds=thresholds,
                repetitions=repetitions, rng=rng, null_p=null_p,
            )
            for threshold in thresholds:
                plug_in = summarize_crossings(
                    plug_in_paths[threshold], horizon=horizon, repetitions=repetitions,
                    confidence_level=confidence,
                )
                null = summarize_crossings(
                    null_paths[threshold], horizon=horizon, repetitions=repetitions,
                    confidence_level=confidence,
                )
                rate = float(row['discordance_rate'])
                mean_t = plug_in['mean_crossing_discordances_conditional']
                median_t = plug_in['median_crossing_discordances_conditional']
                language_records.append({
                    'dataset': dataset,
                    **row,
                    'threshold': threshold,
                    'repetitions': repetitions,
                    'empirical_selection_probability': plug_in['crossing_rate'],
                    'empirical_selection_ci_low': plug_in['crossing_rate_ci_low'],
                    'empirical_selection_ci_high': plug_in['crossing_rate_ci_high'],
                    'power_interpretation_applicable': bool(row['q_hat'] > null_p),
                    'null_crossing_rate': null['crossing_rate'],
                    'null_crossing_ci_low': null['crossing_rate_ci_low'],
                    'null_crossing_ci_high': null['crossing_rate_ci_high'],
                    'mean_crossing_discordances_conditional': mean_t,
                    'median_crossing_discordances_conditional': median_t,
                    'estimated_mean_crossing_examples_conditional': (
                        min(float(mean_t) / rate, row['n_examples'])
                        if mean_t is not None and rate > 0 else None
                    ),
                    'estimated_median_crossing_examples_conditional': (
                        min(float(median_t) / rate, row['n_examples'])
                        if median_t is not None and rate > 0 else None
                    ),
                    'restricted_mean_monitored_discordances': plug_in[
                        'restricted_mean_monitored_discordances'
                    ],
                })
                router_null_paths[(dataset, threshold)].append(null_paths[threshold] <= horizon)

    router_records = []
    for (dataset, threshold), paths in sorted(router_null_paths.items()):
        matrix = np.vstack(paths)
        count = int(matrix.any(axis=0).sum())
        low, high = wilson_interval(count, repetitions, confidence)
        router_records.append({
            'dataset': dataset,
            'threshold': threshold,
            'groups': int(matrix.shape[0]),
            'repetitions': repetitions,
            'router_any_false_crossing_rate': count / repetitions,
            'router_any_false_crossing_ci_low': low,
            'router_any_false_crossing_ci_high': high,
            'simulation_model': 'independent null streams; observed discovery discordance horizons',
        })
    return pd.DataFrame(count_records), pd.DataFrame(language_records), pd.DataFrame(router_records)


def make_figures(
    synthetic: pd.DataFrame, languages: pd.DataFrame, output_dir: Path,
    primary_threshold: float, sensitivity_threshold: float,
) -> None:
    figures = output_dir / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    horizon = int(synthetic['discordance_budget'].max())
    available_thresholds = set(map(float, synthetic['threshold'].unique()))
    if primary_threshold not in available_thresholds:
        raise ValueError(f'primary routing threshold {primary_threshold:g} is absent from results')
    if sensitivity_threshold not in available_thresholds:
        raise ValueError(
            f'per-group sensitivity threshold {sensitivity_threshold:g} is absent from results'
        )
    curve = synthetic[
        (synthetic['discordance_budget'] == horizon)
        & (synthetic['threshold'] == primary_threshold)
    ].sort_values('q')
    sensitivity_curve = synthetic[
        (synthetic['discordance_budget'] == horizon)
        & (synthetic['threshold'] == sensitivity_threshold)
    ].sort_values('q')

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        curve['q'], curve['crossing_rate'], marker='o', linewidth=2,
        label=f'Primary e-process (threshold {primary_threshold:g})',
    )
    ax.fill_between(curve['q'], curve['crossing_rate_ci_low'], curve['crossing_rate_ci_high'], alpha=.2)
    if sensitivity_threshold != primary_threshold:
        ax.plot(
            sensitivity_curve['q'], sensitivity_curve['crossing_rate'], marker='^',
            linestyle='--', linewidth=1.5,
            label=f'Per-group sensitivity (threshold {sensitivity_threshold:g})',
        )
    ax.plot(curve['q'], curve['fixed_exact_rejection_rate'], marker='s', label='Fixed-horizon exact test')
    ax.axhline(.05, color='black', linestyle='--', linewidth=1, label=r'$\alpha=0.05$')
    ax.set(
        xlabel=r'Discordant win probability $q$',
        ylabel='Rejection / crossing probability',
        ylim=(-.02, 1.02),
        title=f'Primary routing threshold: {primary_threshold:g}',
    )
    ax.grid(alpha=.25)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'synthetic_power.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)

    alternative = curve[curve['q'] > .5]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(alternative['q'], alternative['median_crossing_discordances_conditional'], marker='o')
    ax.fill_between(
        alternative['q'], alternative['q25_crossing_discordances_conditional'],
        alternative['q75_crossing_discordances_conditional'], alpha=.2, label='Conditional IQR',
    )
    ax.set(
        xlabel=r'Discordant win probability $q$', ylabel='Discordances to crossing',
        title=f'Primary routing threshold {primary_threshold:g}, horizon {horizon}',
    )
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'synthetic_stopping_time.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)

    empirical = languages[languages['threshold'] == primary_threshold].copy()
    empirical['label'] = empirical['dataset'] + '/' + empirical['language_code']
    empirical = empirical.sort_values(['dataset', 'q_hat'])
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = np.where(empirical['dataset'].eq('sib200'), '#1f77b4', '#ff7f0e')
    ax.barh(empirical['label'], empirical['empirical_selection_probability'], color=colors)
    ax.set(xlabel='Discovery-calibrated probability of crossing', xlim=(0, 1.02))
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(figures / f'language_empirical_power.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build e-process Type-I error, power, and stopping-time metrics.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--sib-discovery-run', required=True)
    parser.add_argument('--massive-discovery-run', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--synthetic-repetitions', type=int)
    parser.add_argument('--empirical-repetitions', type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.synthetic_repetitions is not None:
        cfg['synthetic']['repetitions'] = args.synthetic_repetitions
    if args.empirical_repetitions is not None:
        cfg['discovery_calibrated']['repetitions'] = args.empirical_repetitions

    datasets = {
        'sib200': Path(args.sib_discovery_run).resolve(),
        'massive': Path(args.massive_discovery_run).resolve(),
    }
    for dataset, run_dir in datasets.items():
        report = validate_run(run_dir, require_completed=True)
        if not report['valid'] or report['stage'] != 'discovery':
            raise ValueError(f'{dataset} discovery run is not valid and complete: {run_dir}')

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(int(cfg['seed'])).spawn(2)
    synthetic = build_synthetic(cfg, np.random.default_rng(children[0]))
    counts, languages, router = build_discovery_calibrated(
        datasets, cfg, np.random.default_rng(children[1])
    )
    write_csv(synthetic, output_dir / 'synthetic_operating_characteristics.csv')
    write_csv(counts, output_dir / 'observed_discovery_counts.csv')
    write_csv(languages, output_dir / 'language_operating_characteristics.csv')
    write_csv(router, output_dir / 'router_type1_error.csv')
    make_figures(
        synthetic, languages, output_dir,
        primary_threshold=float(cfg['primary_routing_threshold']),
        sensitivity_threshold=float(cfg['per_group_sensitivity_threshold']),
    )
    write_json({
        'analysis': 'eprocess_operating_characteristics',
        'config': cfg,
        'source_runs': {key: str(value) for key, value in datasets.items()},
        'interpretation': {
            'test_data_used': False,
            'synthetic': 'Bernoulli discordant outcomes at the reported q values.',
            'discovery_calibrated': 'Plug-in bootstrap using discovery q_hat and observed discordance horizons.',
            'router_null': 'Independent null streams with observed group-specific horizons.',
            'stopping_time': 'Conditional summaries exclude non-crossers; restricted means assign them the full horizon.',
        },
    }, output_dir / 'manifest.json')
    print(f'Wrote e-process metrics to {output_dir}')


if __name__ == '__main__':
    main()
