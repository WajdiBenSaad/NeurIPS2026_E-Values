from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.stats import binomtest

from evalues_routing.statistics.eprocess import log_uniform_mixture_evalue


@lru_cache(maxsize=None)
def _cached_log_evalue(wins: int, losses: int, null_p: float) -> float:
    """Reuse the authoritative e-value implementation across Monte Carlo paths."""
    return log_uniform_mixture_evalue(wins, losses, null_p)


def wilson_interval(successes: int, repetitions: int, confidence_level: float) -> tuple[float, float]:
    if repetitions <= 0:
        raise ValueError('repetitions must be positive')
    interval = binomtest(successes, repetitions).proportion_ci(
        confidence_level=confidence_level, method='wilson'
    )
    return float(interval.low), float(interval.high)


def exact_test_critical_wins(horizon: int, alpha: float, null_p: float = 0.5) -> int:
    """Smallest terminal win count rejected by the fixed-horizon exact test."""
    if horizon <= 0:
        return 1
    for wins in range(horizon + 1):
        if binomtest(wins, n=horizon, p=null_p, alternative='greater').pvalue <= alpha:
            return wins
    return horizon + 1


def simulate_crossing_times(
    *, q: float, horizon: int, thresholds: list[float], repetitions: int,
    rng: np.random.Generator, null_p: float = 0.5,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Simulate first crossings in discordance time and terminal win counts.

    ``horizon + 1`` denotes no crossing. E-values are evaluated exclusively by
    the production implementation in ``statistics.eprocess``.
    """
    if not 0 <= q <= 1:
        raise ValueError('q must be in [0,1]')
    if horizon <= 0 or repetitions <= 0:
        raise ValueError('horizon and repetitions must be positive')
    if not thresholds or any(threshold <= 1 for threshold in thresholds):
        raise ValueError('all thresholds must exceed one')

    unique = sorted(set(float(value) for value in thresholds))
    log_thresholds = {value: math.log(value) for value in unique}
    crossings = {value: np.full(repetitions, horizon + 1, dtype=np.int32) for value in unique}
    terminal_wins = np.zeros(repetitions, dtype=np.int32)

    for repetition in range(repetitions):
        wins = 0
        pending = set(unique)
        for t, outcome in enumerate(rng.binomial(1, q, size=horizon), start=1):
            wins += int(outcome)
            if pending:
                log_e = _cached_log_evalue(wins, t - wins, float(null_p))
                crossed = [value for value in pending if log_e >= log_thresholds[value]]
                for value in crossed:
                    crossings[value][repetition] = t
                    pending.remove(value)
        terminal_wins[repetition] = wins
    return crossings, terminal_wins


def summarize_crossings(
    crossing_times: np.ndarray, *, horizon: int, repetitions: int, confidence_level: float,
) -> dict[str, float | int | None]:
    crossed = crossing_times <= horizon
    count = int(crossed.sum())
    low, high = wilson_interval(count, repetitions, confidence_level)
    observed = crossing_times[crossed].astype(float)
    restricted = np.minimum(crossing_times, horizon).astype(float)
    return {
        'crossings': count,
        'repetitions': repetitions,
        'crossing_rate': count / repetitions,
        'crossing_rate_ci_low': low,
        'crossing_rate_ci_high': high,
        'mean_crossing_discordances_conditional': float(observed.mean()) if count else None,
        'median_crossing_discordances_conditional': float(np.median(observed)) if count else None,
        'q25_crossing_discordances_conditional': float(np.quantile(observed, 0.25)) if count else None,
        'q75_crossing_discordances_conditional': float(np.quantile(observed, 0.75)) if count else None,
        'restricted_mean_monitored_discordances': float(restricted.mean()),
    }
