from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalues_routing.evaluation.metrics import per_language_metrics
from evalues_routing.utils.io import read_json, write_json, write_jsonl


class LanguageCheckpointWriter:
    """Persist each completed language before inference advances to the next."""

    def __init__(self, run_dir: str | Path, stage: str, total_languages: int, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.stage = stage
        self.total_languages = int(total_languages)
        progress_path = self.run_dir / 'progress.json'
        if resume and progress_path.exists():
            existing = read_json(progress_path)
            if int(existing['total_languages']) != self.total_languages or existing['stage'] != stage:
                raise RuntimeError(f'Progress metadata does not match requested run: {progress_path}')
            self.completed_languages = list(existing.get('completed_languages', []))
            self.started_at_utc = existing['started_at_utc']
        else:
            self.completed_languages = []
            self.started_at_utc = datetime.now(timezone.utc).isoformat()
            self._write_progress(status='running', current_language=None)

    @property
    def checkpoint_root(self) -> Path:
        return self.run_dir / 'language_checkpoints'

    def _write_progress(self, status: str, current_language: dict[str, Any] | None) -> None:
        write_json({
            'stage': self.stage,
            'status': status,
            'started_at_utc': self.started_at_utc,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'total_languages': self.total_languages,
            'completed_count': len(self.completed_languages),
            'current_language': current_language,
            'completed_languages': self.completed_languages,
        }, self.run_dir / 'progress.json')

    def language_started(self, index: int, language_code: str, language: str, n_examples: int) -> None:
        self._write_progress(status='running', current_language={
            'index': int(index),
            'language_code': language_code,
            'language': language,
            'n_examples': int(n_examples),
        })

    def language_completed(
        self,
        index: int,
        language_code: str,
        language: str,
        predictions: list[dict],
        translations: list[dict],
        elapsed_seconds: float,
    ) -> Path:
        checkpoint = self.checkpoint_root / f'{index:02d}_{language_code}'
        write_jsonl(predictions, checkpoint / 'predictions.jsonl')
        write_jsonl(translations, checkpoint / 'translations.jsonl')
        metrics = per_language_metrics(predictions)[language_code]
        fixes = sum(bool(row['translated_correct']) and not bool(row['direct_correct']) for row in predictions)
        regressions = sum(bool(row['direct_correct']) and not bool(row['translated_correct']) for row in predictions)
        summary = {
            'index': int(index),
            'total_languages': self.total_languages,
            'language_code': language_code,
            'language': language,
            'n_examples': len(predictions),
            'elapsed_seconds': float(elapsed_seconds),
            'fixed': fixes,
            'regressed': regressions,
            'metrics': metrics,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        write_json(summary, checkpoint / 'summary.json')
        self.completed_languages.append({
            'index': int(index),
            'language_code': language_code,
            'language': language,
            'n_examples': len(predictions),
            'elapsed_seconds': float(elapsed_seconds),
            'checkpoint': str(checkpoint),
        })
        self._write_progress(status='running', current_language=None)
        return checkpoint

    def language_interrupted(
        self, index: int, language_code: str, language: str, n_examples: int, error: str,
    ) -> None:
        self._write_progress(status='interrupted', current_language={
            'index': int(index),
            'language_code': language_code,
            'language': language,
            'n_examples': int(n_examples),
            'error': error,
        })

    def finalize(self) -> None:
        self._write_progress(status='completed', current_language=None)
