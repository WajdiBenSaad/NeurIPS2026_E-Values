from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

from evalues_routing.data.sib200 import load_sib200, SIB200_LABELS
from evalues_routing.data.massive import load_massive
from evalues_routing.models.classifiers import PrototypeClassifier
from evalues_routing.models.translators import TranslationManager
from evalues_routing.evaluation.metrics import per_language_metrics
from evalues_routing.utils.runtime import select_device


def load_rows(cfg: dict, splits: Iterable[str]) -> tuple[list[dict], list[str]]:
    task = cfg['dataset']['task']
    if task == 'topic_classification':
        rows = load_sib200(cfg, splits)
        labels = cfg['dataset'].get('labels', SIB200_LABELS)
        return rows, labels
    if task == 'intent_classification':
        return load_massive(cfg, splits)
    raise ValueError(f'Unsupported task: {task}')


def infer_both_paths(
    cfg: dict,
    rows: list[dict],
    labels: list[str],
    logger=None,
    progress=None,
    language_order: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run direct and translated paths and return prediction rows plus translation records."""
    device = select_device(cfg['project'].get('device', 'auto'))
    multi_cfg = cfg['models']['multilingual_encoder']
    eng_cfg = cfg['models']['english_encoder']
    direct_clf = PrototypeClassifier(multi_cfg['id'], labels, device, multi_cfg.get('batch_size', 128), multi_cfg.get('revision'))
    english_clf = PrototypeClassifier(eng_cfg['id'], labels, device, eng_cfg.get('batch_size', 128), eng_cfg.get('revision'))
    translator = TranslationManager(cfg, device)

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r['language_code']].append(r)

    predictions = []
    translations = []
    configured_order = language_order or list(by_lang)
    total_languages = len(configured_order)
    for code, group in by_lang.items():
        if code not in configured_order:
            raise ValueError(f'Language {code} is absent from the configured language order.')
        language_index = configured_order.index(code) + 1
        language = group[0]['language']
        language_start = perf_counter()
        if logger:
            logger.info(
                '[%d/%d] Starting language=%s code=%s n=%d device=%s',
                language_index, total_languages, language, code, len(group), device,
            )
        if progress:
            progress.language_started(language_index, code, language, len(group))
        texts = [r['text'] for r in group]
        if logger:
            logger.info('[%d/%d] Direct classification started: %s', language_index, total_languages, code)
        direct = direct_clf.predict(texts)
        if logger:
            logger.info(
                '[%d/%d] Direct classification finished: %s (%.3f s/example)',
                language_index, total_languages, code, direct.per_example_seconds,
            )

        eligible = group[0].get('translate_eligible', True)
        if eligible:
            source_code = group[0]['translation_code']
            if logger:
                logger.info('[%d/%d] Translation started: %s source=%s', language_index, total_languages, code, source_code)
            tb = translator.translate(texts, source_code)
            translated_texts = tb.texts
            if logger:
                logger.info(
                    '[%d/%d] Translation finished: %s backend=%s model=%s (%.3f s/example)',
                    language_index, total_languages, code, tb.backend, tb.model_id, tb.per_example_seconds,
                )
            translated_pred = english_clf.predict(translated_texts)
            translated_latency = tb.per_example_seconds + translated_pred.per_example_seconds
        else:
            translated_texts = texts
            translated_pred = direct
            tb = None
            translated_latency = direct.per_example_seconds

        language_predictions = []
        language_translations = []
        for i, r in enumerate(group):
            ttext = translated_texts[i]
            translation_row = {
                'example_id': r['example_id'],
                'language_code': code,
                'source_text': r['text'],
                'translated_text': ttext,
                'backend': tb.backend if tb else 'identity',
                'translation_model': tb.model_id if tb else 'identity',
                'translation_latency_s': tb.per_example_seconds if tb else 0.0,
            }
            prediction_row = {
                **r,
                'translated_text': ttext,
                'direct_prediction': direct.labels[i],
                'direct_score': direct.scores[i],
                'direct_correct': direct.labels[i] == r['gold_label'],
                'translated_prediction': translated_pred.labels[i],
                'translated_score': translated_pred.scores[i],
                'translated_correct': translated_pred.labels[i] == r['gold_label'],
                'direct_latency_s': direct.per_example_seconds,
                'translated_latency_s': translated_latency,
            }
            language_translations.append(translation_row)
            language_predictions.append(prediction_row)

        translations.extend(language_translations)
        predictions.extend(language_predictions)
        elapsed_seconds = perf_counter() - language_start
        fixes = sum(row['translated_correct'] and not row['direct_correct'] for row in language_predictions)
        regressions = sum(row['direct_correct'] and not row['translated_correct'] for row in language_predictions)
        checkpoint = None
        if progress:
            checkpoint = progress.language_completed(
                language_index, code, language, language_predictions,
                language_translations, elapsed_seconds,
            )
        if logger:
            logger.info(
                '[%d/%d] Completed language=%s code=%s elapsed=%.1f s fixes=%d regressions=%d checkpoint=%s',
                language_index, total_languages, language, code, elapsed_seconds,
                fixes, regressions, checkpoint,
            )
    return predictions, translations


def aggregate_discovery_metrics(predictions: list[dict]) -> dict:
    return {'per_language': per_language_metrics(predictions)}
