from __future__ import annotations

from typing import Iterable

SIB200_LABELS = [
    'science/technology', 'travel', 'politics', 'sports',
    'health', 'entertainment', 'geography'
]


def load_sib200(config: dict, splits: Iterable[str]) -> list[dict]:
    """Load configured SIB-200 languages and normalize rows to the project schema."""
    from datasets import load_dataset

    rows: list[dict] = []
    max_rows = config['dataset'].get('max_rows_per_language')
    per_split_limits = config['dataset'].get('max_rows_per_language_per_split') or {}
    if max_rows is not None and per_split_limits:
        raise ValueError('Configure either max_rows_per_language or max_rows_per_language_per_split, not both.')
    for lang in config['dataset']['languages']:
        code = lang['code']
        collected = 0
        for split in splits:
            split_collected = 0
            split_limit = per_split_limits.get(split)
            ds = load_dataset(config['dataset']['id'], code, split=split, revision=config['dataset'].get('revision'))
            for idx, ex in enumerate(ds):
                rows.append({
                    'dataset': 'sib200',
                    'language': lang['name'],
                    'language_code': code,
                    'translation_code': code,
                    'split': split,
                    'example_id': f"{code}:{split}:{ex.get('index_id', idx)}",
                    'text': ex['text'],
                    'gold_id': int(ex['label']) if 'label' in ex else SIB200_LABELS.index(ex['category']),
                    'gold_label': SIB200_LABELS[int(ex['label'])] if 'label' in ex else ex['category'],
                    'translate_eligible': bool(lang.get('translate', True)),
                })
                collected += 1
                split_collected += 1
                if split_limit is not None and split_collected >= int(split_limit):
                    break
                if max_rows is not None and collected >= int(max_rows):
                    break
            if max_rows is not None and collected >= int(max_rows):
                break
    return rows
