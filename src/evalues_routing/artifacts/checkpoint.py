from __future__ import annotations

from pathlib import Path
from evalues_routing.utils.io import write_json, checksums
from evalues_routing.utils.runs import mark_completed
from evalues_routing.artifacts.validation import validate_run


def finalize_run(run_dir: str | Path) -> dict:
    """Checksum, validate, write the validation report, then seal the run.

    No file is written to the run after the COMPLETED marker is created.
    """
    run = Path(run_dir)
    write_json(checksums(run), run / 'checksums.json')
    report = validate_run(run, require_completed=False)
    write_json(report, run / 'validation_report.json')
    if not report['valid']:
        raise RuntimeError(f'Run failed validation before sealing: {report}')
    mark_completed(run)
    return report
