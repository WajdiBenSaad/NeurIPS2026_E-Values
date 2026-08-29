#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from evalues_routing.artifacts.validation import validate_run
from evalues_routing.evaluation.policy_evaluation import evaluate_policy
from evalues_routing.routing.baseline_routers import (
    DirectOnlyRouter,
    SelectedLanguageRouter,
    TranslationOnlyRouter,
)
from evalues_routing.routing.eprocess_router import discover_languages
from evalues_routing.routing.fixed_router import FixedLanguageRouter
from evalues_routing.utils.io import read_json, read_jsonl
from evalues_routing.utils.config import load_yaml


POLICY_ORDER = ['direct_only', 'fixed_list', 'p_value', 'e_value', 'translation_only']
POLICY_LABELS = {
    'direct_only': 'Direct only',
    'fixed_list': 'Fixed list',
    'p_value': 'Paired p-value',
    'e_value': 'E-value',
    'translation_only': 'Translation only',
}
COLORS = {
    'direct_only': '#4C78A8',
    'fixed_list': '#F58518',
    'p_value': '#54A24B',
    'e_value': '#E45756',
    'translation_only': '#B279A2',
}


def validate_pair(discovery: Path, evaluation: Path, expected_dataset: str) -> None:
    for run, expected_stage in ((discovery, 'discovery'), (evaluation, 'evaluation')):
        report = validate_run(run, require_completed=True)
        if not report['valid'] or report['stage'] != expected_stage:
            raise RuntimeError(f'Invalid {expected_dataset} {expected_stage} run {run}: {report}')
        router = read_json(run / 'selected_languages.json')
        if router.get('dataset') != expected_dataset:
            raise RuntimeError(
                f'Expected dataset {expected_dataset}, found {router.get("dataset")} in {run}'
            )


def stable_seed(*parts: str | int) -> int:
    value = '|'.join(map(str, parts)).encode('utf-8')
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def stable_order_key(example_id: str, seed: int) -> str:
    return hashlib.sha256(f'{seed}|{example_id}'.encode('utf-8')).hexdigest()


def language_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row['language_code']].append(row)
    return dict(groups)


def language_order(config: dict) -> list[str]:
    return [item['code'] for item in config['dataset']['languages']]


def pair_counts(rows: list[dict]) -> tuple[int, int]:
    fixed = sum((not bool(row['direct_correct'])) and bool(row['translated_correct']) for row in rows)
    regressed = sum(bool(row['direct_correct']) and (not bool(row['translated_correct'])) for row in rows)
    return int(fixed), int(regressed)


def exact_binomial_log_p(fixed: int, regressed: int) -> float:
    """Stable log p-value for the one-sided exact discordant-pair test."""
    n = fixed + regressed
    if n == 0:
        return 0.0
    values = np.arange(fixed, n + 1, dtype=float)
    log_terms = (
        gammaln(n + 1.0)
        - gammaln(values + 1.0)
        - gammaln(n - values + 1.0)
        - n * math.log(2.0)
    )
    return float(min(0.0, logsumexp(log_terms)))


def holm_adjust_log(log_pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(log_pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = float('-inf')
    for rank, (name, log_pvalue) in enumerate(ordered, start=1):
        candidate = min(0.0, log_pvalue + math.log(m - rank + 1))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def pvalue_display(log_pvalue: float) -> str:
    log10_pvalue = log_pvalue / math.log(10.0)
    if log10_pvalue < -300:
        return '<1e-300'
    return f'{math.exp(log_pvalue):.4g}'


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
    scores = np.divide(
        2.0 * np.diag(matrix),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    return float(scores.mean())


def paired_bootstrap_delta(
    rows: list[dict], replicates: int, confidence: float, seed: int
) -> dict[str, float]:
    """Paired multinomial bootstrap for translated-minus-direct accuracy and Macro-F1."""
    labels = sorted(
        {row['gold_label'] for row in rows}
        | {row['direct_prediction'] for row in rows}
        | {row['translated_prediction'] for row in rows}
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    counts = Counter(
        (
            label_to_id[row['gold_label']],
            label_to_id[row['direct_prediction']],
            label_to_id[row['translated_prediction']],
        )
        for row in rows
    )
    cells = list(counts)
    observed = np.asarray([counts[cell] for cell in cells], dtype=float)
    probabilities = observed / observed.sum()
    gold = np.asarray([cell[0] for cell in cells], dtype=int)
    direct = np.asarray([cell[1] for cell in cells], dtype=int)
    translated = np.asarray([cell[2] for cell in cells], dtype=int)
    n_labels = len(labels)
    n = len(rows)
    rng = np.random.default_rng(seed)
    accuracy_delta = np.empty(replicates, dtype=float)
    macro_delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        weights = rng.multinomial(n, probabilities)
        direct_matrix = np.bincount(
            gold * n_labels + direct, weights=weights, minlength=n_labels * n_labels
        ).reshape(n_labels, n_labels)
        translated_matrix = np.bincount(
            gold * n_labels + translated, weights=weights, minlength=n_labels * n_labels
        ).reshape(n_labels, n_labels)
        accuracy_delta[index] = (
            np.trace(translated_matrix) - np.trace(direct_matrix)
        ) / n
        macro_delta[index] = (
            macro_f1_from_confusion(translated_matrix)
            - macro_f1_from_confusion(direct_matrix)
        )
    tail = (1.0 - confidence) / 2.0
    return {
        'Accuracy CI low': float(np.quantile(accuracy_delta, tail)),
        'Accuracy CI high': float(np.quantile(accuracy_delta, 1.0 - tail)),
        'Macro-F1 CI low': float(np.quantile(macro_delta, tail)),
        'Macro-F1 CI high': float(np.quantile(macro_delta, 1.0 - tail)),
    }


def policy_accuracy_inference(
    rows: list[dict],
    selected_languages: list[str],
    dataset_label: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Stratified paired bootstrap for frozen e-router accuracy versus direct-only."""
    selected = set(selected_languages)
    grouped = language_groups(rows)
    rng = np.random.default_rng(seed)
    bootstrap_totals = np.zeros(replicates, dtype=float)
    observed_total = 0
    total_examples = 0
    for group in grouped.values():
        effects = np.asarray([
            int(
                row['translated_correct']
                if row.get('translate_eligible', True) and row['language_code'] in selected
                else row['direct_correct']
            )
            - int(row['direct_correct'])
            for row in group
        ], dtype=int)
        counts = np.asarray([
            np.sum(effects == -1),
            np.sum(effects == 0),
            np.sum(effects == 1),
        ], dtype=int)
        draws = rng.multinomial(len(group), counts / len(group), size=replicates)
        bootstrap_totals += draws[:, 2] - draws[:, 0]
        observed_total += int(effects.sum())
        total_examples += len(group)
    bootstrap_delta = 100.0 * bootstrap_totals / total_examples
    observed_delta = 100.0 * observed_total / total_examples
    tail = (1.0 - confidence) / 2.0
    return pd.DataFrame([{
        'Dataset': dataset_label,
        'Comparison': 'E-router - direct-only',
        'N': total_examples,
        'Delta accuracy (pp)': observed_delta,
        'CI low (pp)': float(np.quantile(bootstrap_delta, tail)),
        'CI high (pp)': float(np.quantile(bootstrap_delta, 1.0 - tail)),
        'Confidence level': confidence,
        'Bootstrap replicates': replicates,
        'Resampling': 'paired examples within language',
    }])


def save_table(
    frame: pd.DataFrame,
    name: str,
    reports_tables: Path,
    paper_tables: Path,
    *,
    latex: bool = True,
) -> None:
    reports_tables.mkdir(parents=True, exist_ok=True)
    paper_tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(reports_tables / f'{name}.csv', index=False)
    if latex:
        latex_text = frame.to_latex(
            index=False,
            float_format=lambda value: f'{value:.4g}',
            escape=True,
            na_rep='--',
        )
        (reports_tables / f'{name}.tex').write_text(latex_text, encoding='utf-8')
        shutil.copy2(reports_tables / f'{name}.tex', paper_tables / f'{name}.tex')


def save_figure(fig: plt.Figure, name: str, reports_figures: Path, paper_figures: Path) -> None:
    reports_figures.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    pdf = reports_figures / f'{name}.pdf'
    png = reports_figures / f'{name}.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=220, bbox_inches='tight')
    shutil.copy2(pdf, paper_figures / pdf.name)
    plt.close(fig)


def discovery_table(metrics: dict, selected: dict, order: list[str]) -> pd.DataFrame:
    rows = []
    for code in order:
        stat = metrics['language_statistics'][code]
        performance = metrics['per_language'][code]
        rows.append({
            'Language': stat['language'],
            'Code': code,
            'N': stat['n_examples'],
            'Direct accuracy': performance['direct']['accuracy'],
            'Translated accuracy': performance['translated']['accuracy'],
            'Delta accuracy': performance['delta_accuracy'],
            'Direct Macro-F1': performance['direct']['macro_f1'],
            'Translated Macro-F1': performance['translated']['macro_f1'],
            'Delta Macro-F1': performance['delta_macro_f1'],
            'Fixed': stat['fixed'],
            'Regressed': stat['regressed'],
            'Discordant': stat['discordant_pairs'],
            'p-value': stat['p_value'],
            'Final E': stat['final_e_value'],
            'Max E (descriptive)': stat['max_e_value'],
            'First crossing': stat['first_crossing_discordant_index'],
            'E-route': bool(stat['e_selected']),
            'BH p-route': bool(stat.get('bh_selected', False)),
            'e-BH final-E route': bool(stat.get('e_bh_selected', False)),
        })
    return pd.DataFrame(rows)


def evaluation_language_table(
    rows: list[dict], selected: dict, discovery_metrics: dict, order: list[str]
) -> pd.DataFrame:
    grouped = language_groups(rows)
    output = []
    e_selected = set(selected['e_selected_languages'])
    for code in order:
        group = grouped[code]
        fixed, regressed = pair_counts(group)
        n = len(group)
        direct_accuracy = sum(bool(row['direct_correct']) for row in group) / n
        translated_accuracy = sum(bool(row['translated_correct']) for row in group) / n
        gold = [row['gold_label'] for row in group]
        direct_predictions = [row['direct_prediction'] for row in group]
        translated_predictions = [row['translated_prediction'] for row in group]
        labels = sorted(set(gold) | set(direct_predictions) | set(translated_predictions))
        direct_cm = confusion_matrix(gold, direct_predictions, labels=labels)
        translated_cm = confusion_matrix(gold, translated_predictions, labels=labels)
        direct_f1 = macro_f1_from_confusion(direct_cm)
        translated_f1 = macro_f1_from_confusion(translated_cm)
        selected_for_translation = code in e_selected
        eligible = bool(group[0].get('translate_eligible', True))
        direction_correct = (
            (selected_for_translation and translated_accuracy > direct_accuracy)
            or ((not selected_for_translation) and translated_accuracy <= direct_accuracy)
        )
        discovery = discovery_metrics['per_language'][code]
        output.append({
            'Language': group[0]['language'],
            'Code': code,
            'N': n,
            'Discovery delta accuracy': discovery['delta_accuracy'],
            'Test direct accuracy': direct_accuracy,
            'Test translated accuracy': translated_accuracy,
            'Test delta accuracy': translated_accuracy - direct_accuracy,
            'Test direct Macro-F1': direct_f1,
            'Test translated Macro-F1': translated_f1,
            'Test delta Macro-F1': translated_f1 - direct_f1,
            'Fixed': fixed,
            'Regressed': regressed,
            'Eligible': eligible,
            'Frozen route': 'translate' if selected_for_translation else 'direct',
            'Accuracy direction correct': direction_correct,
        })
    return pd.DataFrame(output)


def significance_table(
    rows: list[dict], selected: dict, order: list[str], replicates: int, confidence: float
) -> pd.DataFrame:
    grouped = language_groups(rows)
    raw_log_pvalues = {}
    for code in order:
        group = grouped[code]
        if group[0].get('translate_eligible', True):
            fixed, regressed = pair_counts(group)
            raw_log_pvalues[code] = exact_binomial_log_p(fixed, regressed)
    adjusted_log = holm_adjust_log(raw_log_pvalues)
    output = []
    for code in order:
        group = grouped[code]
        fixed, regressed = pair_counts(group)
        n = len(group)
        direct_accuracy = sum(bool(row['direct_correct']) for row in group) / n
        translated_accuracy = sum(bool(row['translated_correct']) for row in group) / n
        direct_labels = [row['direct_prediction'] for row in group]
        translated_labels = [row['translated_prediction'] for row in group]
        gold = [row['gold_label'] for row in group]
        all_labels = sorted(set(gold) | set(direct_labels) | set(translated_labels))
        direct_f1 = macro_f1_from_confusion(confusion_matrix(gold, direct_labels, labels=all_labels))
        translated_f1 = macro_f1_from_confusion(
            confusion_matrix(gold, translated_labels, labels=all_labels)
        )
        ci = paired_bootstrap_delta(
            group,
            replicates,
            confidence,
            stable_seed(selected['dataset'], code, 'paired-bootstrap'),
        )
        log_pvalue = raw_log_pvalues.get(code, 0.0)
        adjusted_log_pvalue = adjusted_log.get(code, 0.0)
        output.append({
            'Language': group[0]['language'],
            'Code': code,
            'N': n,
            'Delta accuracy': translated_accuracy - direct_accuracy,
            **ci,
            'Delta Macro-F1': translated_f1 - direct_f1,
            'Fixed': fixed,
            'Regressed': regressed,
            'Exact paired p-value': pvalue_display(log_pvalue),
            'Exact paired -log10(p)': -log_pvalue / math.log(10.0),
            'Holm-adjusted p-value': pvalue_display(adjusted_log_pvalue),
            'Holm-adjusted -log10(p)': -adjusted_log_pvalue / math.log(10.0),
            'Holm significant': bool(adjusted_log_pvalue <= math.log(0.05)),
        })
    return pd.DataFrame(output)


def policy_table(dataset_label: str, evaluation_metrics: dict) -> pd.DataFrame:
    direct = evaluation_metrics['direct_only']
    output = []
    for policy in POLICY_ORDER:
        metrics = evaluation_metrics[policy]
        output.append({
            'Dataset': dataset_label,
            'Policy': policy,
            'Policy label': POLICY_LABELS[policy],
            'Accuracy': metrics['accuracy'],
            'Delta accuracy vs direct': metrics['accuracy'] - direct['accuracy'],
            'Macro-F1': metrics['macro_f1'],
            'Delta Macro-F1 vs direct': metrics['macro_f1'] - direct['macro_f1'],
            'Translated examples': metrics['translated_examples'],
            'Translation rate': metrics['translation_rate'],
            'Mean latency (s)': metrics['latency']['mean_s'],
            'Median latency (s)': metrics['latency']['median_s'],
            'P95 latency (s)': metrics['latency']['p95_s'],
        })
    return pd.DataFrame(output)


def threshold_sensitivity(
    rows: list[dict], discovery_metrics: dict, selected: dict, alpha_values: list[float]
) -> pd.DataFrame:
    stats = discovery_metrics['language_statistics']
    eligible = {
        row['language_code'] for row in rows if row.get('translate_eligible', True)
    }
    output = []
    primary_alpha = 0.05 / len(eligible) if eligible else 0.05
    primary_threshold = 1.0 / primary_alpha
    settings = [('Primary FWER=0.05', primary_alpha, primary_threshold)]
    settings.extend(
        (f'Per-group sensitivity alpha={alpha:g}', alpha, 1.0 / alpha)
        for alpha in alpha_values
    )
    primary_languages = {
        code
        for code in eligible
        if stats[code]['max_log_e_value'] >= math.log(primary_threshold)
    }
    for label, alpha, threshold in settings:
        log_threshold = math.log(threshold)
        chosen = sorted(
            code
            for code in eligible
            if stats[code]['max_log_e_value'] >= log_threshold
        )
        metrics, _ = evaluate_policy(rows, SelectedLanguageRouter(chosen), label)
        output.append({
            'Setting': label,
            'Nominal per-language alpha': alpha,
            'Threshold': threshold,
            'Selected count': len(chosen),
            'Selected languages': ', '.join(chosen),
            'Accuracy': metrics['accuracy'],
            'Macro-F1': metrics['macro_f1'],
            'Translation rate': metrics['translation_rate'],
            'Mean latency (s)': metrics['latency']['mean_s'],
            'Matches primary router': set(chosen) == primary_languages,
        })
    return pd.DataFrame(output)


def budget_sensitivity(
    discovery_rows: list[dict],
    evaluation_rows: list[dict],
    config: dict,
    budgets: list[int],
) -> pd.DataFrame:
    grouped = language_groups(discovery_rows)
    order_seed = int(config['statistics']['eprocess']['trajectory_order_seed'])
    ordered = {
        code: sorted(group, key=lambda row: stable_order_key(row['example_id'], order_seed))
        for code, group in grouped.items()
    }
    full_budget = min(len(group) for group in ordered.values())
    values = sorted({value for value in budgets if value < full_budget} | {full_budget})
    output = []
    for budget in values:
        subset = [row for code in ordered for row in ordered[code][:budget]]
        stats_config = deepcopy(config['statistics'])
        stats_config['eprocess']['decision_threshold'] = 280.0
        discovered, _ = discover_languages(subset, stats_config)
        chosen = sorted(code for code, stat in discovered.items() if stat['e_selected'])
        metrics, _ = evaluate_policy(
            evaluation_rows, SelectedLanguageRouter(chosen), f'budget_{budget}'
        )
        output.append({
            'Examples per language': budget,
            'Selected count': len(chosen),
            'Selected languages': ', '.join(chosen),
            'Test accuracy': metrics['accuracy'],
            'Test Macro-F1': metrics['macro_f1'],
            'Test translation rate': metrics['translation_rate'],
            'Test mean latency (s)': metrics['latency']['mean_s'],
        })
    return pd.DataFrame(output)


def ordering_stability(
    discovery_rows: list[dict], config: dict, order: list[str], repetitions: int
) -> pd.DataFrame:
    selected_counts = Counter()
    crossings: dict[str, list[int]] = defaultdict(list)
    base_seed = int(config['statistics']['eprocess']['trajectory_order_seed'])
    for offset in range(repetitions):
        stats_config = deepcopy(config['statistics'])
        stats_config['eprocess']['trajectory_order_seed'] = base_seed + offset
        stats_config['eprocess']['decision_threshold'] = 280.0
        discovered, _ = discover_languages(discovery_rows, stats_config)
        for code, stat in discovered.items():
            if stat['e_selected']:
                selected_counts[code] += 1
                crossings[code].append(int(stat['first_crossing_discordant_index']))
    grouped = language_groups(discovery_rows)
    output = []
    for code in order:
        values = crossings.get(code, [])
        output.append({
            'Language': grouped[code][0]['language'],
            'Code': code,
            'Order repetitions': repetitions,
            'Selected repetitions': selected_counts[code],
            'Selection frequency': selected_counts[code] / repetitions,
            'Median first crossing': float(np.median(values)) if values else np.nan,
            'Minimum first crossing': min(values) if values else np.nan,
            'Maximum first crossing': max(values) if values else np.nan,
        })
    return pd.DataFrame(output)


def metric_target_sensitivity(
    evaluation_rows: list[dict], discovery_metrics: dict, selected: dict, order: list[str]
) -> pd.DataFrame:
    """Descriptive only: compare the primary router with the sign of discovery Macro-F1."""
    primary = set(selected['e_selected_languages'])
    macro_positive = {
        code
        for code in order
        if discovery_metrics['per_language'][code]['delta_macro_f1'] > 0
        and any(
            row['language_code'] == code and row.get('translate_eligible', True)
            for row in evaluation_rows
        )
    }
    rows = []
    for name, chosen in (
        ('Primary accuracy e-process', primary),
        ('Post-hoc validation Macro-F1 sign', macro_positive),
    ):
        metrics, _ = evaluate_policy(
            evaluation_rows, SelectedLanguageRouter(sorted(chosen)), name
        )
        rows.append({
            'Analysis': name,
            'Inferential status': 'primary' if name.startswith('Primary') else 'descriptive post-hoc',
            'Selected count': len(chosen),
            'Selected languages': ', '.join(sorted(chosen)),
            'Test accuracy': metrics['accuracy'],
            'Test Macro-F1': metrics['macro_f1'],
            'Test translation rate': metrics['translation_rate'],
            'Test mean latency (s)': metrics['latency']['mean_s'],
        })
    return pd.DataFrame(rows)


def cost_utility_table(policy_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for penalty in (0.0, 0.01, 0.025, 0.05, 0.10, 0.20):
        for _, row in policy_frame.iterrows():
            rows.append({
                'Dataset': row['Dataset'],
                'Penalty lambda': penalty,
                'Policy': row['Policy'],
                'Utility (accuracy - lambda*translation rate)': (
                    row['Accuracy'] - penalty * row['Translation rate']
                ),
            })
    result = pd.DataFrame(rows)
    maxima = result.groupby(['Dataset', 'Penalty lambda'])[
        'Utility (accuracy - lambda*translation rate)'
    ].transform('max')
    result['Best utility'] = np.isclose(
        result['Utility (accuracy - lambda*translation rate)'], maxima
    )
    return result


def class_level_table(
    rows: list[dict], selected: dict, fixed_languages: list[str], order: list[str]
) -> pd.DataFrame:
    grouped = language_groups(rows)
    primary_router = SelectedLanguageRouter(selected['e_selected_languages'])
    fixed_router = FixedLanguageRouter(fixed_languages)
    output = []
    for code in order:
        group = grouped[code]
        labels = sorted(
            {row['gold_label'] for row in group}
            | {row['direct_prediction'] for row in group}
            | {row['translated_prediction'] for row in group}
        )
        gold = [row['gold_label'] for row in group]
        predictions = {
            'Direct': [row['direct_prediction'] for row in group],
            'Translated': [row['translated_prediction'] for row in group],
            'E-route': [
                row['translated_prediction']
                if primary_router.route(code) == 'translate' and row.get('translate_eligible', True)
                else row['direct_prediction']
                for row in group
            ],
            'Fixed-list': [
                row['translated_prediction']
                if fixed_router.route(code) == 'translate' and row.get('translate_eligible', True)
                else row['direct_prediction']
                for row in group
            ],
        }
        computed = {}
        for path, prediction in predictions.items():
            precision, recall, f1, support = precision_recall_fscore_support(
                gold, prediction, labels=labels, zero_division=0
            )
            computed[path] = (precision, recall, f1, support)
        for index, label in enumerate(labels):
            row = {
                'Language': group[0]['language'],
                'Code': code,
                'Class': label,
                'Support': int(computed['Direct'][3][index]),
            }
            for path in predictions:
                row[f'{path} precision'] = computed[path][0][index]
                row[f'{path} recall'] = computed[path][1][index]
                row[f'{path} F1'] = computed[path][2][index]
            output.append(row)
    return pd.DataFrame(output)


def write_confusion_tables(
    rows: list[dict], selected: dict, fixed_languages: list[str], dataset: str, destination: Path
) -> None:
    eligible = {row['language_code'] for row in rows if row.get('translate_eligible', True)}
    routers = {
        'direct_only': DirectOnlyRouter(),
        'fixed_list': FixedLanguageRouter(fixed_languages),
        'e_value': SelectedLanguageRouter(selected['e_selected_languages']),
        'translation_only': TranslationOnlyRouter(eligible),
    }
    labels = sorted({row['gold_label'] for row in rows})
    destination.mkdir(parents=True, exist_ok=True)
    for policy, router in routers.items():
        _, routed = evaluate_policy(rows, router, policy)
        matrix = confusion_matrix(
            [row['gold_label'] for row in routed],
            [row['policy_prediction'] for row in routed],
            labels=labels,
        )
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
        frame.index.name = 'Gold label'
        frame.to_csv(destination / f'appendix_confusion_{dataset}_{policy}.csv')


def plot_policy_tradeoff(frame: pd.DataFrame, reports: Path, paper: Path) -> None:
    datasets = list(dict.fromkeys(frame['Dataset']))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.2, 2.15 * len(datasets)), squeeze=False)
    for row_index, dataset in enumerate(datasets):
        subset = frame[frame['Dataset'] == dataset]
        for column_index, metric in enumerate(('Accuracy', 'Macro-F1')):
            ax = axes[row_index, column_index]
            points: dict[tuple[float, float], list[pd.Series]] = defaultdict(list)
            for _, row in subset.iterrows():
                points[(round(float(row['Translation rate']), 10), round(float(row[metric]), 10))].append(row)
            for (translation_rate, value), point_rows in points.items():
                policies = [str(row['Policy']) for row in point_rows]
                representative = 'e_value' if 'e_value' in policies else policies[0]
                label = ' / '.join(POLICY_LABELS[policy] for policy in policies)
                ax.scatter(
                    100 * translation_rate, value, s=72,
                    color=COLORS[representative], zorder=3,
                )
                right_edge = translation_rate >= 0.8
                fixed_list = label == POLICY_LABELS['fixed_list']
                below_point = fixed_list or (right_edge and len(policies) > 1)
                ax.annotate(
                    label,
                    (100 * translation_rate, value),
                    xytext=(
                        -5 if right_edge else 5,
                        -12 if below_point else 6,
                    ),
                    textcoords='offset points',
                    ha='right' if right_edge else 'left', fontsize=9,
                    va='top' if below_point else 'bottom',
                )
            ax.set_title(f'{dataset}: {metric}', fontsize=11)
            ax.set_xlabel('Translation rate (%)', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='both', labelsize=9)
            y_min = float(subset[metric].min())
            y_max = float(subset[metric].max())
            y_span = max(y_max - y_min, 0.01)
            ax.set_ylim(y_min - 0.06 * y_span, y_max + 0.24 * y_span)
            ax.grid(alpha=0.25)
    save_figure(fig, 'policy_quality_translation_tradeoff', reports, paper)


def plot_cost_utility(frame: pd.DataFrame, reports: Path, paper: Path) -> None:
    datasets = list(dict.fromkeys(frame['Dataset']))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8), squeeze=False)
    for index, dataset in enumerate(datasets):
        ax = axes[0, index]
        subset = frame[frame['Dataset'] == dataset]
        for policy in POLICY_ORDER:
            policy_rows = subset[subset['Policy'] == policy]
            ax.plot(
                policy_rows['Penalty lambda'],
                policy_rows['Utility (accuracy - lambda*translation rate)'],
                marker='o', linewidth=1.2, color=COLORS[policy], label=POLICY_LABELS[policy],
            )
        ax.set_title(dataset)
        ax.set_xlabel('Translation penalty $\\lambda$')
        ax.set_ylabel('Accuracy-based utility')
        ax.grid(alpha=0.25)
    axes[0, -1].legend(frameon=False, fontsize=7)
    save_figure(fig, 'posthoc_cost_utility', reports, paper)


def plot_language_effects(
    frame: pd.DataFrame, dataset: str, reports: Path, paper: Path
) -> None:
    subset = frame.sort_values('Test delta accuracy')
    y = np.arange(len(subset))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.2), sharey=True)
    for ax, column, title in (
        (axes[0], 'Test delta accuracy', 'Accuracy difference'),
        (axes[1], 'Test delta Macro-F1', 'Macro-F1 difference'),
    ):
        values = subset[column].to_numpy()
        colors = np.where(values >= 0, '#54A24B', '#E45756')
        ax.barh(y, values, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel('Translated minus direct')
        ax.grid(axis='x', alpha=0.25)
    axes[0].set_yticks(y, subset['Language'])
    fig.suptitle(f'{dataset}: per-language intervention effects')
    save_figure(fig, f'language_effects_{dataset.lower().replace("-", "")}', reports, paper)


def plot_validation_test_consistency(
    frame: pd.DataFrame, dataset: str, reports: Path, paper: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(
        frame['Discovery delta accuracy'], frame['Test delta accuracy'],
        c=np.where(frame['Frozen route'].eq('translate'), '#E45756', '#4C78A8'), s=48,
    )
    for _, row in frame.iterrows():
        ax.annotate(
            row['Code'],
            (row['Discovery delta accuracy'], row['Test delta accuracy']),
            xytext=(3, 3), textcoords='offset points', fontsize=7,
        )
    low = min(frame['Discovery delta accuracy'].min(), frame['Test delta accuracy'].min())
    high = max(frame['Discovery delta accuracy'].max(), frame['Test delta accuracy'].max())
    ax.plot([low, high], [low, high], linestyle='--', color='gray', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlabel('Discovery accuracy difference')
    ax.set_ylabel('Test accuracy difference')
    ax.set_title(f'{dataset}: discovery-to-test consistency')
    ax.grid(alpha=0.2)
    save_figure(fig, f'validation_test_consistency_{dataset.lower().replace("-", "")}', reports, paper)


def plot_sensitivity(
    frame: pd.DataFrame,
    x: str,
    dataset: str,
    name: str,
    reports: Path,
    paper: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for ax, y, title in (
        (axes[0], 'Test accuracy' if 'Test accuracy' in frame else 'Accuracy', 'Accuracy'),
        (axes[1], 'Test Macro-F1' if 'Test Macro-F1' in frame else 'Macro-F1', 'Macro-F1'),
        (
            axes[2],
            'Test translation rate' if 'Test translation rate' in frame else 'Translation rate',
            'Translation rate',
        ),
    ):
        ax.plot(frame[x], frame[y], marker='o', color='#4C78A8')
        ax.set_xlabel(x)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    fig.suptitle(f'{dataset}: {name.replace("_", " ")}')
    save_figure(fig, f'{name}_{dataset.lower().replace("-", "")}', reports, paper)


def plot_ordering_stability(
    frame: pd.DataFrame, dataset: str, reports: Path, paper: Path
) -> None:
    subset = frame.sort_values('Selection frequency')
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.barh(subset['Language'], subset['Selection frequency'], color='#72B7B2')
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Selection frequency across prespecified orders')
    ax.set_title(f'{dataset}: routing-order stability')
    ax.grid(axis='x', alpha=0.25)
    save_figure(fig, f'ordering_stability_{dataset.lower().replace("-", "")}', reports, paper)


def plot_eprocess_trajectories(
    discovery: Path, selected: dict, dataset: str, reports: Path, paper: Path
) -> None:
    trajectories = read_jsonl(discovery / 'eprocess_trajectories.jsonl')
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        grouped[row['language_code']].append(row)
    chosen = set(selected['e_selected_languages'])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for code, points in grouped.items():
        is_selected = code in chosen
        ax.plot(
            [point['t_discordant'] for point in points],
            [point['log_e_value'] for point in points],
            linewidth=1.2 if is_selected else 0.8,
            alpha=0.9 if is_selected else 0.45,
            label=points[0]['language'],
        )
    sensitivity_threshold = 20.0
    primary_threshold = 280.0
    ax.axhline(
        math.log(primary_threshold), linestyle='--', color='black', linewidth=1.2,
        label='Primary FWER threshold: 280',
    )
    ax.axhline(
        math.log(sensitivity_threshold), linestyle=':', color='#666666', linewidth=1,
        label='Per-group sensitivity threshold: 20',
    )
    ax.set_xlabel('Discordant pairs observed')
    ax.set_ylabel('Log e-process value')
    ax.set_title(f'{dataset}: complete discovery trajectories')
    ax.legend(frameon=False, fontsize=6, ncol=3)
    ax.grid(alpha=0.2)
    save_figure(fig, f'eprocess_trajectories_all_{dataset.lower().replace("-", "")}', reports, paper)


def plot_accuracy_macro_alignment(
    frame: pd.DataFrame, dataset: str, reports: Path, paper: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    ax.scatter(frame['Test delta accuracy'], frame['Test delta Macro-F1'], color='#4C78A8', s=48)
    for _, row in frame.iterrows():
        ax.annotate(
            row['Code'], (row['Test delta accuracy'], row['Test delta Macro-F1']),
            xytext=(3, 3), textcoords='offset points', fontsize=7,
        )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Accuracy difference')
    ax.set_ylabel('Macro-F1 difference')
    ax.set_title(f'{dataset}: target-metric alignment')
    ax.grid(alpha=0.2)
    save_figure(fig, f'accuracy_macro_alignment_{dataset.lower().replace("-", "")}', reports, paper)


def write_generated_text(
    datasets: dict,
    policy_frame: pd.DataFrame,
    evaluation_frames: dict[str, pd.DataFrame],
    reports_generated: Path,
    paper_generated: Path,
) -> None:
    reports_generated.mkdir(parents=True, exist_ok=True)
    paper_generated.mkdir(parents=True, exist_ok=True)
    macros = []
    paragraphs = []
    conclusions = []
    for dataset, (_, _, label) in datasets.items():
        subset = policy_frame[policy_frame['Dataset'] == label].set_index('Policy')
        language_frame = evaluation_frames[dataset]
        e_metrics = subset.loc['e_value']
        direct_metrics = subset.loc['direct_only']
        fixed_metrics = subset.loc['fixed_list']
        full_metrics = subset.loc['translation_only']
        selected_count = int(language_frame['Frozen route'].eq('translate').sum())
        direction_count = int(language_frame['Accuracy direction correct'].sum())
        dataset_macro = 'SibTwoHundred' if dataset == 'sib200' else 'Massive'
        for policy in POLICY_ORDER:
            name = ''.join(part.capitalize() for part in policy.split('_'))
            row = subset.loc[policy]
            macros.extend([
                f'\\newcommand{{\\{dataset_macro}{name}Accuracy}}{{{100 * row["Accuracy"]:.2f}\\%}}',
                f'\\newcommand{{\\{dataset_macro}{name}FOne}}{{{row["Macro-F1"]:.4f}}}',
                f'\\newcommand{{\\{dataset_macro}{name}TranslateRate}}{{{100 * row["Translation rate"]:.1f}\\%}}',
            ])
        paragraphs.append(
            f'On {label}, the frozen e-value router selected {selected_count} language(s) for translation. '
            f'It obtained {100 * e_metrics["Accuracy"]:.2f}\\% accuracy and Macro-F1 '
            f'{e_metrics["Macro-F1"]:.4f} at a {100 * e_metrics["Translation rate"]:.1f}\\% '
            f'translation rate, compared with {100 * direct_metrics["Accuracy"]:.2f}\\%/'
            f'{direct_metrics["Macro-F1"]:.4f} for direct-only, '
            f'{100 * fixed_metrics["Accuracy"]:.2f}\\%/{fixed_metrics["Macro-F1"]:.4f} '
            f'for fixed-list routing, and {100 * full_metrics["Accuracy"]:.2f}\\%/'
            f'{full_metrics["Macro-F1"]:.4f} for translation-only. '
            f'The frozen decision matched the observed test accuracy direction for '
            f'{direction_count}/{len(language_frame)} evaluated languages.'
        )
        conclusions.append(
            f'{label}: {100 * e_metrics["Accuracy"]:.2f}\\% accuracy, '
            f'{e_metrics["Macro-F1"]:.4f} Macro-F1, and '
            f'{100 * e_metrics["Translation rate"]:.1f}\\% translation'
        )
    conclusion = (
        'Across independently discovered and frozen routers, ' + '; '.join(conclusions) + '. '
        'The contrast between selective translation on SIB-200 and broad translation on MASSIVE '
        'supports a task-specific rather than universal routing boundary.'
    )
    generated = {
        'results_macros.tex': '\n'.join(macros) + '\n',
        'results_summary.tex': ' '.join(paragraphs) + '\n',
        'conclusion_summary.tex': conclusion + '\n',
    }
    for name, text in generated.items():
        (reports_generated / name).write_text(text, encoding='utf-8')
        shutil.copy2(reports_generated / name, paper_generated / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build comprehensive main-paper and appendix artifacts from completed runs.'
    )
    parser.add_argument('--sib-discovery-run', required=True)
    parser.add_argument('--sib-evaluation-run', required=True)
    parser.add_argument('--massive-discovery-run', required=True)
    parser.add_argument('--massive-evaluation-run', required=True)
    parser.add_argument('--reports-root', default='reports')
    parser.add_argument('--paper-root', default='paper')
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--order-repetitions', type=int, default=50)
    args = parser.parse_args()

    datasets = {
        'sib200': (
            Path(args.sib_discovery_run), Path(args.sib_evaluation_run), 'SIB-200'
        ),
        'massive': (
            Path(args.massive_discovery_run), Path(args.massive_evaluation_run), 'MASSIVE'
        ),
    }
    reports_root = Path(args.reports_root)
    paper_root = Path(args.paper_root)
    reports_tables = reports_root / 'tables'
    reports_figures = reports_root / 'figures'
    reports_generated = reports_root / 'generated'
    paper_tables = paper_root / 'tables'
    paper_figures = paper_root / 'figures'
    paper_generated = paper_root / 'generated'
    appendix_data = reports_root / 'appendix_data'
    for directory in (
        reports_tables, reports_figures, reports_generated, appendix_data,
        paper_tables, paper_figures, paper_generated,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    policy_frames = []
    policy_inference_frames = []
    evaluation_frames: dict[str, pd.DataFrame] = {}
    threshold_frames: dict[str, pd.DataFrame] = {}
    budget_frames: dict[str, pd.DataFrame] = {}
    ordering_frames: dict[str, pd.DataFrame] = {}
    manifest = {
        'bootstrap_replicates': args.bootstrap_replicates,
        'order_repetitions': args.order_repetitions,
        'datasets': {},
        'notes': [
            'Primary routing targets paired accuracy.',
            'Macro-F1-sign and cost-utility analyses are descriptive post-hoc sensitivity analyses.',
            'The running maximum is reported descriptively; inference uses the threshold-crossing event.',
        ],
    }

    for dataset, (discovery, evaluation, label) in datasets.items():
        validate_pair(discovery, evaluation, dataset)
        config = load_yaml(discovery / 'config.yaml')
        order = language_order(config)
        discovery_rows = read_jsonl(discovery / 'predictions.jsonl')
        evaluation_rows = read_jsonl(evaluation / 'predictions.jsonl')
        discovery_metrics = read_json(discovery / 'metrics.json')
        evaluation_metrics = read_json(evaluation / 'metrics.json')['policies']
        selected = read_json(evaluation / 'selected_languages.json')
        fixed_languages = selected.get('fixed_selected_languages', [])

        discovery_frame = discovery_table(discovery_metrics, selected, order)
        evaluation_frame = evaluation_language_table(
            evaluation_rows, selected, discovery_metrics, order
        )
        significance_frame = significance_table(
            evaluation_rows,
            selected,
            order,
            args.bootstrap_replicates,
            float(config['statistics']['uncertainty']['confidence_level']),
        )
        policies = policy_table(label, evaluation_metrics)
        policy_inference = policy_accuracy_inference(
            evaluation_rows,
            selected['e_selected_languages'],
            label,
            args.bootstrap_replicates,
            float(config['statistics']['uncertainty']['confidence_level']),
            stable_seed(dataset, args.bootstrap_replicates, 'policy-accuracy-ci'),
        )
        thresholds = threshold_sensitivity(
            evaluation_rows, discovery_metrics, selected, [0.01, 0.025, 0.05, 0.10]
        )
        budgets = budget_sensitivity(
            discovery_rows,
            evaluation_rows,
            config,
            [50, 100, 200, 400, 800, 1200, 1600],
        )
        ordering = ordering_stability(
            discovery_rows, config, order, args.order_repetitions
        )
        metric_targets = metric_target_sensitivity(
            evaluation_rows, discovery_metrics, selected, order
        )
        classes = class_level_table(
            evaluation_rows, selected, fixed_languages, order
        )

        save_table(discovery_frame, f'appendix_discovery_{dataset}', reports_tables, paper_tables)
        save_table(evaluation_frame, f'main_language_evaluation_{dataset}', reports_tables, paper_tables)
        save_table(significance_frame, f'appendix_paired_inference_{dataset}', reports_tables, paper_tables)
        save_table(thresholds, f'appendix_threshold_sensitivity_{dataset}', reports_tables, paper_tables)
        save_table(budgets, f'appendix_budget_sensitivity_{dataset}', reports_tables, paper_tables)
        save_table(ordering, f'appendix_ordering_stability_{dataset}', reports_tables, paper_tables)
        save_table(metric_targets, f'appendix_metric_target_sensitivity_{dataset}', reports_tables, paper_tables)
        save_table(classes, f'appendix_class_metrics_{dataset}', reports_tables, paper_tables, latex=False)
        write_confusion_tables(
            evaluation_rows, selected, fixed_languages, dataset, appendix_data
        )

        plot_language_effects(evaluation_frame, label, reports_figures, paper_figures)
        plot_validation_test_consistency(evaluation_frame, label, reports_figures, paper_figures)
        plot_sensitivity(
            budgets, 'Examples per language', label, 'budget_sensitivity', reports_figures, paper_figures
        )
        plot_sensitivity(
            thresholds, 'Threshold', label, 'threshold_sensitivity', reports_figures, paper_figures
        )
        plot_ordering_stability(ordering, label, reports_figures, paper_figures)
        plot_eprocess_trajectories(discovery, selected, label, reports_figures, paper_figures)
        plot_accuracy_macro_alignment(evaluation_frame, label, reports_figures, paper_figures)

        policy_frames.append(policies)
        policy_inference_frames.append(policy_inference)
        evaluation_frames[dataset] = evaluation_frame
        threshold_frames[dataset] = thresholds
        budget_frames[dataset] = budgets
        ordering_frames[dataset] = ordering
        manifest['datasets'][dataset] = {
            'discovery_run': str(discovery.resolve()),
            'evaluation_run': str(evaluation.resolve()),
            'discovery_examples': len(discovery_rows),
            'evaluation_examples': len(evaluation_rows),
            'selected_languages': selected['e_selected_languages'],
        }

    policies = pd.concat(policy_frames, ignore_index=True)
    save_table(policies, 'main_policy_evaluation', reports_tables, paper_tables)
    policy_inference = pd.concat(policy_inference_frames, ignore_index=True)
    save_table(
        policy_inference,
        'main_policy_accuracy_inference',
        reports_tables,
        paper_tables,
    )
    utility = cost_utility_table(policies)
    save_table(utility, 'appendix_cost_utility', reports_tables, paper_tables)
    plot_policy_tradeoff(policies, reports_figures, paper_figures)
    plot_cost_utility(utility, reports_figures, paper_figures)

    cross_rows = []
    for dataset, (_, _, label) in datasets.items():
        frame = evaluation_frames[dataset]
        policy = policies[(policies['Dataset'] == label) & (policies['Policy'] == 'e_value')].iloc[0]
        cross_rows.append({
            'Dataset': label,
            'Selected languages': int(frame['Frozen route'].eq('translate').sum()),
            'Accuracy-direction agreement': int(frame['Accuracy direction correct'].sum()),
            'Evaluated languages': len(frame),
            'E-router accuracy': policy['Accuracy'],
            'E-router Macro-F1': policy['Macro-F1'],
            'Translation rate': policy['Translation rate'],
            'Mean latency (s)': policy['Mean latency (s)'],
        })
    save_table(pd.DataFrame(cross_rows), 'main_cross_dataset_summary', reports_tables, paper_tables)

    write_generated_text(
        datasets, policies, evaluation_frames, reports_generated, paper_generated
    )
    manifest['generated_files'] = sorted(
        str(path.relative_to(reports_root))
        for path in reports_root.rglob('*')
        if path.is_file()
    )
    (reports_generated / 'analysis_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
    )
    index_lines = [
        '# Generated paper-analysis artifacts', '',
        'All artifacts are derived from validated stored predictions; no model inference or translation is run.', '',
        '## Main paper', '',
        '- `tables/main_policy_evaluation.*`: overall policy comparison.',
        '- `tables/main_policy_accuracy_inference.*`: paired policy-level accuracy confidence intervals.',
        '- `tables/main_language_evaluation_*.{csv,tex}`: complete test results by language.',
        '- `tables/main_cross_dataset_summary.*`: cross-dataset headline results.',
        '- `figures/policy_quality_translation_tradeoff.*`: quality-cost policy chart.', '',
        '## Appendix', '',
        '- Discovery evidence, paired inference, threshold, budget, order, metric-target, and class-level tables.',
        '- Full confusion matrices in `appendix_data/`.',
        '- Language effects, discovery-to-test consistency, e-process trajectories, and stability charts.', '',
        '## Generated manuscript text', '',
        '- `generated/results_macros.tex`',
        '- `generated/results_summary.tex`',
        '- `generated/conclusion_summary.tex`',
        '- `generated/analysis_manifest.json`',
    ]
    (reports_root / 'ANALYSIS_INDEX.md').write_text('\n'.join(index_lines) + '\n', encoding='utf-8')
    print(reports_root.resolve())


if __name__ == '__main__':
    main()
