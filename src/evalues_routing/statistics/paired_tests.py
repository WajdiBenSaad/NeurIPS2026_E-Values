from __future__ import annotations

from scipy.stats import binomtest


def discordant_counts(rows: list[dict]) -> tuple[int, int]:
    fixed = sum((not bool(r['direct_correct'])) and bool(r['translated_correct']) for r in rows)
    regressed = sum(bool(r['direct_correct']) and (not bool(r['translated_correct'])) for r in rows)
    return int(fixed), int(regressed)


def exact_one_sided_pvalue(fixed: int, regressed: int) -> float:
    """Exact one-sided paired test conditional on discordant pairs.

    Under H0, translation wins with probability <= 0.5 among discordant pairs.
    The boundary p=0.5 gives the valid one-sided exact binomial p-value.
    """
    n = fixed + regressed
    if n == 0:
        return 1.0
    return float(binomtest(fixed, n=n, p=0.5, alternative='greater').pvalue)
