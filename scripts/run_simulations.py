#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
from evalues_routing.utils.config import load_config
from evalues_routing.statistics.eprocess import trajectory, first_crossing
from evalues_routing.utils.io import write_json


def main():
    ap = argparse.ArgumentParser(description='Monte Carlo validity/power self-check for the configured Bernoulli e-process.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    stats = cfg['statistics']
    sim = cfg.get('simulations', {})
    rng = np.random.default_rng(int(sim.get('seed', 20260817)))
    reps = int(sim.get('repetitions', 2000))
    horizon = int(sim.get('horizon', 250))
    threshold = float(stats['eprocess']['decision_threshold'])
    null_p = float(stats['eprocess']['null_win_probability'])

    def crossing_rate(p):
        crosses = 0
        crossing_times = []
        for _ in range(reps):
            x = rng.binomial(1, p, size=horizon).tolist()
            c = first_crossing(trajectory(x, null_p), threshold)
            if c is not None:
                crosses += 1
                crossing_times.append(c)
        return {
            'probability': p,
            'crossing_rate': crosses/reps,
            'median_crossing_time': float(np.median(crossing_times)) if crossing_times else None,
        }

    result = {
        'repetitions': reps,
        'horizon': horizon,
        'threshold': threshold,
        'nominal_alpha_bound': 1/threshold,
        'null': [crossing_rate(float(p)) for p in sim.get('null_probabilities', [0.5])],
        'alternative': [crossing_rate(float(p)) for p in sim.get('alternative_probabilities', [0.65])],
    }
    write_json(result, args.output)
    print(args.output)

if __name__ == '__main__':
    main()
