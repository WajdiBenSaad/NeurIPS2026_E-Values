from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger('evalues_routing')
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
