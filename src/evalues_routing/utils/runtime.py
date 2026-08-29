from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def select_device(requested: str = 'auto') -> str:
    import torch
    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(False)  # generation kernels may lack strict deterministic variants
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def release_accelerator_memory() -> None:
    """Best-effort cleanup; subprocess exit then releases the full address space."""
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def git_commit(repo_root: str | Path = '.') -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_snapshot() -> dict:
    packages = ['torch','transformers','datasets','sentence-transformers','numpy','scipy','pandas','scikit-learn','PyYAML']
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'packages': versions,
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'git_commit': git_commit(),
    }
