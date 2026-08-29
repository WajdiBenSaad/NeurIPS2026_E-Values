from __future__ import annotations


def benjamini_hochberg(pvalues: dict[str, float], q: float = 0.05) -> dict[str, bool]:
    """Benjamini-Hochberg p-value decisions."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    k = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            k = i
    selected = {name for name, _ in items[:k]}
    return {name: name in selected for name in pvalues}


def e_bh(evalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """e-BH decisions: reject the largest k e-values with e_(k) >= m/(alpha*k).

    Implemented for sensitivity/future work; it is not the primary 4-page routing rule.
    """
    items = sorted(evalues.items(), key=lambda kv: kv[1], reverse=True)
    m = len(items)
    k = 0
    for i, (_, e) in enumerate(items, start=1):
        if e >= m / (alpha * i):
            k = i
    selected = {name for name, _ in items[:k]}
    return {name: name in selected for name in evalues}
