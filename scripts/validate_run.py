#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from evalues_routing.statistics.eprocess import uniform_mixture_evalue, trajectory
from evalues_routing.statistics.paired_tests import exact_one_sided_pvalue
from evalues_routing.artifacts.validation import validate_run
from evalues_routing.utils.io import write_json


def self_check() -> dict:
    checks = {}
    checks['evalue_origin_is_one'] = abs(uniform_mixture_evalue(0, 0) - 1.0) < 1e-12
    checks['strong_wins_raise_evidence'] = uniform_mixture_evalue(20, 0) > 20
    checks['balanced_pairs_not_strong'] = uniform_mixture_evalue(10, 10) < 20
    checks['pvalue_strong_wins'] = exact_one_sided_pvalue(20, 0) < 0.05
    tr = trajectory([1, 1, 0, 1, 1])
    checks['trajectory_counts'] = tr[-1].wins == 4 and tr[-1].losses == 1
    return {'checks': checks, 'valid': all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--self-check', action='store_true')
    ap.add_argument('--report', default=None, help='Optional external JSON report path; completed runs are never modified.')
    args = ap.parse_args()
    if args.self_check:
        report = self_check()
        print(report)
        if not report['valid']:
            raise SystemExit(1)
        return
    if not args.run_dir:
        ap.error('--run-dir or --self-check is required')
    report = validate_run(args.run_dir, require_completed=True)
    if args.report:
        write_json(report, args.report)
    print(report)
    if not report['valid']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
