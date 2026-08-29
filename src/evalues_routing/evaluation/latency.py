from __future__ import annotations

import statistics


def latency_summary(rows: list[dict], route_key: str | None = None) -> dict:
    vals = []
    for r in rows:
        if route_key is None:
            vals.append(float(r.get('direct_latency_s', 0.0)))
        else:
            route = r[route_key]
            vals.append(float(r['translated_latency_s'] if route == 'translate' else r['direct_latency_s']))
    if not vals:
        return {'n': 0, 'mean_s': None, 'median_s': None, 'p95_s': None}
    s = sorted(vals)
    idx = min(len(s)-1, int(round(0.95*(len(s)-1))))
    return {
        'n': len(vals),
        'mean_s': float(statistics.fmean(vals)),
        'median_s': float(statistics.median(vals)),
        'p95_s': float(s[idx]),
    }
