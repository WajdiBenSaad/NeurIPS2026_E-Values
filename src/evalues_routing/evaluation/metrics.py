from __future__ import annotations

from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(rows: list[dict], prediction_key: str) -> dict:
    if not rows:
        return {'n': 0, 'accuracy': None, 'macro_f1': None}
    y_true = [r['gold_label'] for r in rows]
    y_pred = [r[prediction_key] for r in rows]
    return {
        'n': len(rows),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def per_language_metrics(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r['language_code']].append(r)
    result = {}
    for code, group in sorted(groups.items()):
        direct = classification_metrics(group, 'direct_prediction')
        translated = classification_metrics(group, 'translated_prediction')
        result[code] = {
            'language': group[0]['language'],
            'direct': direct,
            'translated': translated,
            'delta_macro_f1': translated['macro_f1'] - direct['macro_f1'],
            'delta_accuracy': translated['accuracy'] - direct['accuracy'],
        }
    return result
