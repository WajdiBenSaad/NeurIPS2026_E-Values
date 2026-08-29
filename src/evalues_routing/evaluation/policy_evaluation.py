from __future__ import annotations

from evalues_routing.evaluation.metrics import classification_metrics
from evalues_routing.evaluation.latency import latency_summary


def evaluate_policy(rows: list[dict], router, name: str) -> tuple[dict, list[dict]]:
    routed = []
    translated = 0
    for r in rows:
        route = router.route(r['language_code'])
        if route == 'translate' and not r.get('translate_eligible', True):
            route = 'direct'
        pred = r['translated_prediction'] if route == 'translate' else r['direct_prediction']
        translated += int(route == 'translate')
        nr = dict(r)
        nr['policy'] = name
        nr['route'] = route
        nr['policy_prediction'] = pred
        routed.append(nr)
    metrics = classification_metrics(routed, 'policy_prediction')
    metrics.update({
        'policy': name,
        'translated_examples': translated,
        'translation_rate': translated / len(routed) if routed else 0.0,
        'latency': latency_summary(routed, 'route'),
    })
    return metrics, routed
