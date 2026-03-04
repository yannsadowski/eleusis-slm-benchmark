"""Estimate parallel evaluation time based on historical results.

Scans the results/ directory for historical timing data and projects
how long a new parallel evaluation would take given the current config.

Usage:
  python scripts/estimate_parallel_time.py eval_models.txt [config.yaml]
  python scripts/estimate_parallel_time.py eval_models.txt config.yaml --results-dir results/
"""

import argparse
import json
from pathlib import Path

import yaml


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = seconds / 60
        return f"{m:.0f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


def load_models_file(path: str) -> list[str]:
    """Read model keys from file (skip empty lines and comments)."""
    models = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                models.append(line)
    return models


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def count_rules(config: dict) -> int:
    """Count rules from config (resolving num_rules=0 via the library)."""
    num_rules = config.get('game', {}).get('num_rules', 0)
    if num_rules == 0:
        library_path = config.get('rules', {}).get('library_path', 'rules.json')
        try:
            with open(library_path) as f:
                data = json.load(f)
            num_rules = len(data.get('rules', []))
        except (FileNotFoundError, json.JSONDecodeError):
            num_rules = 0
    return num_rules


def find_historical_avg(model_key: str, results_dir: Path) -> tuple[float | None, int]:
    """Find average wall-clock seconds per round for a model from past runs.

    Returns:
        (avg_seconds_per_round, number_of_runs_found)
    """
    per_round_times = []

    for path in results_dir.rglob('results.json'):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if data.get('config', {}).get('player_model') != model_key:
            continue

        total_seconds = data.get('statistics', {}).get('total_wall_clock_seconds', 0)
        completed_rounds = len(data.get('rounds', []))

        if completed_rounds > 0 and total_seconds > 0:
            per_round_times.append(total_seconds / completed_rounds)

    if not per_round_times:
        return None, 0

    return sum(per_round_times) / len(per_round_times), len(per_round_times)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate parallel evaluation time from historical results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/estimate_parallel_time.py eval_models.txt
  python scripts/estimate_parallel_time.py eval_models.txt test.yaml
""",
    )
    parser.add_argument('models_file', help='File with model keys (one per line)')
    parser.add_argument('config_file', nargs='?', default='config.yaml',
                        help='Config YAML file (default: config.yaml)')
    parser.add_argument('--results-dir', default='results',
                        help='Results directory to scan (default: results/)')
    args = parser.parse_args()

    models = load_models_file(args.models_file)
    config = load_config(args.config_file)

    num_rules = count_rules(config)
    num_rounds_per_rule = config.get('game', {}).get('num_rounds_per_rule', 1)
    total_rounds = num_rules * num_rounds_per_rule

    results_dir = Path(args.results_dir)

    print()
    print('=' * 62)
    print('  PARALLEL EVALUATION — TIME ESTIMATE')
    print('=' * 62)
    print(f'  Config  : {args.config_file}')
    print(f'  Models  : {len(models)}')
    print(f'  Rounds  : {num_rules} rules × {num_rounds_per_rule} rounds/rule = {total_rounds} rounds per model')
    print('=' * 62)
    print()

    # (model_key, avg_seconds_per_round | None, estimated_total | None, run_count)
    model_estimates: list[tuple[str, float | None, float | None, int]] = []

    for model in models:
        avg, run_count = find_historical_avg(model, results_dir)
        estimated = avg * total_rounds if (avg is not None and total_rounds > 0) else None
        model_estimates.append((model, avg, estimated, run_count))

    col = max(len(m) for m in models) + 2

    for model, avg, estimated, run_count in model_estimates:
        if estimated is not None:
            runs_label = f"{run_count} historical run{'s' if run_count > 1 else ''}"
            source = f"({avg:.0f}s/round avg, {runs_label})"
            print(f'  {model:<{col}} → {format_duration(estimated):>8}   {source}')
        else:
            print(f'  {model:<{col}} → {"unknown":>8}   (no historical data)')

    print()

    known = [(m, e) for m, _, e, _ in model_estimates if e is not None]
    unknown_models = [m for m, _, e, _ in model_estimates if e is None]

    if known:
        bottleneck_model, parallel_time = max(known, key=lambda x: x[1])
        print(f'  Estimated parallel wall time : {format_duration(parallel_time)}')
        print(f'  Bottleneck model             : {bottleneck_model}')
        if unknown_models:
            print(f'  No data for                  : {", ".join(unknown_models)}')
            print('  → actual time may be longer if those models are slow')
    else:
        print('  No historical data found — cannot estimate duration.')
        print('  Run at least one evaluation per model to enable estimates.')

    print()
    print('=' * 62)
    print()


if __name__ == '__main__':
    main()
