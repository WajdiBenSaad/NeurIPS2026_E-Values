from __future__ import annotations

from dataclasses import dataclass
import math
from scipy.special import betaln, betainc


@dataclass(frozen=True)
class EProcessPoint:
    t: int
    wins: int
    losses: int
    log_e: float
    e_value: float


def log_uniform_mixture_evalue(wins: int, losses: int, null_p: float = 0.5) -> float:
    """Log e-value for a one-sided Bernoulli mixture alternative.

    The alternative mixes p uniformly on [null_p, 1]. For null_p=0.5 and
    Bernoulli observations X_t indicating a translation win among discordant
    pairs, each fixed-p likelihood ratio against p0=0.5 is a nonnegative
    supermartingale for every true q<=0.5. A nonnegative mixture preserves the
    e-process property.

    E = (1/(1-p0)) * integral_{p0}^1 p^w (1-p)^l dp / [p0^w (1-p0)^l]
    """
    if wins < 0 or losses < 0:
        raise ValueError('wins/losses must be nonnegative')
    if not 0 < null_p < 1:
        raise ValueError('null_p must be in (0,1)')
    a, b = wins + 1.0, losses + 1.0
    tail = 1.0 - float(betainc(a, b, null_p))
    if tail <= 0.0:
        return float('-inf')
    log_integral = float(betaln(a, b)) + math.log(tail)
    log_prior_density = -math.log(1.0 - null_p)
    log_null_lik = wins * math.log(null_p) + losses * math.log(1.0 - null_p)
    return float(log_prior_density + log_integral - log_null_lik)


def uniform_mixture_evalue(wins: int, losses: int, null_p: float = 0.5) -> float:
    log_e = log_uniform_mixture_evalue(wins, losses, null_p)
    if log_e == float('-inf'):
        return 0.0
    # Cap only the floating representation; decisions use log_e when necessary.
    return float(math.exp(min(log_e, 700.0)))


def trajectory(outcomes: list[int], null_p: float = 0.5) -> list[EProcessPoint]:
    """Compute the e-process after each discordant outcome (1=translation win, 0=loss)."""
    wins = losses = 0
    points = [EProcessPoint(0, 0, 0, 0.0, 1.0)]
    for t, x in enumerate(outcomes, start=1):
        if x not in (0, 1):
            raise ValueError('outcomes must contain only 0/1')
        wins += int(x == 1)
        losses += int(x == 0)
        log_e = log_uniform_mixture_evalue(wins, losses, null_p)
        points.append(EProcessPoint(t, wins, losses, log_e, float(math.exp(min(log_e, 700.0))) if math.isfinite(log_e) else 0.0))
    return points


def first_crossing(points: list[EProcessPoint], threshold: float) -> int | None:
    if threshold <= 1:
        raise ValueError('threshold should exceed 1 for evidential routing')
    log_thr = math.log(threshold)
    for p in points:
        if p.log_e >= log_thr:
            return p.t
    return None
