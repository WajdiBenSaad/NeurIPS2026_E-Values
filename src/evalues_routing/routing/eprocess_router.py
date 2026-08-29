from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from evalues_routing.statistics.eprocess import trajectory, first_crossing
from evalues_routing.statistics.paired_tests import exact_one_sided_pvalue
from evalues_routing.statistics.multiplicity import benjamini_hochberg, e_bh


def _stable_order_key(example_id: str, seed: int) -> str:
    """Outcome-independent deterministic permutation key for sequential trajectories."""
    return hashlib.sha256(f'{seed}|{example_id}'.encode('utf-8')).hexdigest()


def discover_languages(predictions: list[dict], stats_cfg: dict) -> tuple[dict, list[dict]]:
    """Discover translation-worthy languages from discovery-split paired predictions.

    The primary e-process decision is *ever-crossed*: once E_t reaches the fixed
    threshold under the prespecified example ordering, the language is selected.
    This matches the anytime-valid stopping interpretation.
    """
    by_lang = defaultdict(list)
    for row in predictions:
        by_lang[row['language_code']].append(row)

    alpha = float(stats_cfg['paired_test']['alpha'])
    null_p = float(stats_cfg['eprocess']['null_win_probability'])
    threshold = float(stats_cfg['eprocess']['decision_threshold'])
    order_seed = int(stats_cfg['eprocess'].get('trajectory_order_seed', 20260817))
    results = {}
    all_trajectories = []
    pvals = {}
    final_evalues = {}

    for code, rows in sorted(by_lang.items()):
        discordant_rows = []
        for row in rows:
            if (not row['direct_correct']) and row['translated_correct']:
                discordant_rows.append((row['example_id'], 1))
            elif row['direct_correct'] and (not row['translated_correct']):
                discordant_rows.append((row['example_id'], 0))
        discordant_rows.sort(key=lambda pair: _stable_order_key(pair[0], order_seed))
        discordant = [outcome for _, outcome in discordant_rows]

        points = trajectory(discordant, null_p)
        final = points[-1]
        crossing = first_crossing(points, threshold)
        max_point = max(points, key=lambda p: p.log_e)
        p = exact_one_sided_pvalue(final.wins, final.losses)
        pvals[code] = p
        final_evalues[code] = final.e_value
        results[code] = {
            'language_code': code,
            'language': rows[0]['language'],
            'n_examples': len(rows),
            'discordant_pairs': len(discordant),
            'fixed': final.wins,
            'regressed': final.losses,
            'p_value': p,
            'p_selected': bool(p <= alpha),
            'final_e_value': final.e_value,
            'final_log_e_value': final.log_e,
            'max_e_value': max_point.e_value,
            'max_log_e_value': max_point.log_e,
            'e_threshold': threshold,
            'e_selected': bool(crossing is not None),
            'first_crossing_discordant_index': crossing,
            'trajectory_order_seed': order_seed,
        }
        for pt in points:
            all_trajectories.append({
                'language_code': code,
                'language': rows[0]['language'],
                't_discordant': pt.t,
                'wins': pt.wins,
                'losses': pt.losses,
                'log_e_value': pt.log_e,
                'e_value': pt.e_value,
            })

    if stats_cfg.get('multiplicity', {}).get('report_bh', True):
        bh = benjamini_hochberg(pvals, float(stats_cfg['multiplicity'].get('bh_q', 0.05)))
        ebh = e_bh(final_evalues, float(stats_cfg['multiplicity'].get('bh_q', 0.05)))
        for code in results:
            results[code]['bh_selected'] = bool(bh[code])
            results[code]['e_bh_selected'] = bool(ebh[code])
    return results, all_trajectories
