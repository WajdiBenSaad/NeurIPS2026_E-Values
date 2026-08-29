from __future__ import annotations

from typing import Iterable


def humanize_intent(name: str) -> str:
    """Turn intent identifiers into short English prototype strings."""
    return name.replace('_', ' ').strip()


def _load_locale(dataset_config: dict, locale: str, split: str):
    """Load one MASSIVE locale, with a processed MTEB fallback for Datasets 4.x."""
    from datasets import load_dataset
    first = None
    if dataset_config.get('source', 'amazon') == 'amazon':
        try:
            ds = load_dataset(
                dataset_config['id'], locale, split=split,
                revision=dataset_config.get('revision'), trust_remote_code=True,
            )
            return ds, 'amazon'
        except Exception as error:
            first = error
    try:
        ds = load_dataset(
            dataset_config.get('fallback_id', 'mteb/amazon_massive_intent'),
            split=split, revision=dataset_config.get('fallback_revision'),
        )
        lang = locale.split('-')[0].lower()
        if 'lang' not in ds.column_names:
            raise RuntimeError('MTEB MASSIVE fallback does not expose lang column')
        ds = ds.filter(lambda x: str(x['lang']).lower() == lang)
        if len(ds) == 0:
            raise RuntimeError(f'No MTEB MASSIVE rows for lang={lang}')
        return ds, 'mteb'
    except Exception as second:
        raise RuntimeError(
            f'Unable to load MASSIVE locale={locale}, split={split}. '
            f'Primary error: {first}; fallback error: {second}'
        ) from second


def load_massive(config: dict, splits: Iterable[str]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    max_rows = config['dataset'].get('max_rows_per_language')

    for lang in config['dataset']['languages']:
        locale = lang['code']
        collected = 0
        for split in splits:
            ds, source = _load_locale(config['dataset'], locale, split)
            label_names = None
            if source == 'amazon':
                feature = ds.features.get('intent') if hasattr(ds, 'features') else None
                if feature is not None and getattr(feature, 'names', None):
                    label_names = list(feature.names)

            for idx, ex in enumerate(ds):
                if source == 'amazon':
                    raw_intent = ex['intent']
                    if isinstance(raw_intent, str):
                        intent_name = raw_intent
                    else:
                        if label_names is None:
                            raise RuntimeError('MASSIVE intent label names unavailable')
                        intent_name = label_names[int(raw_intent)]
                    text = ex['utt']
                    ex_id = str(ex.get('id', f'{locale}:{split}:{idx}'))
                else:
                    intent_name = str(ex.get('label_text', ex.get('label')))
                    text = str(ex['text'])
                    ex_id = str(ex.get('id', f'{locale}:{split}:{idx}'))

                rows.append({
                    'dataset': 'massive',
                    'language': lang['name'],
                    'language_code': locale,
                    'translation_code': lang.get('nllb_code', locale),
                    'split': split,
                    'example_id': f'{locale}:{split}:{ex_id}',
                    'text': text,
                    'gold_id': -1,
                    'gold_label': humanize_intent(intent_name),
                    'translate_eligible': bool(lang.get('translate', True)),
                })
                collected += 1
                if max_rows is not None and collected >= int(max_rows):
                    break
            if max_rows is not None and collected >= int(max_rows):
                break

    labels = sorted({r['gold_label'] for r in rows})
    return rows, labels
