"""Estimate parallel evaluation time based on historical results.

Two modes:

  Estimate mode (default): projects duration of a future run from historical data.
    python scripts/estimate_parallel_time.py eval_models.txt [config.yaml]

  Progress mode: shows live progress + ETA for evaluations currently running.
    python scripts/estimate_parallel_time.py --progress [--results-dir results/]
"""

import argparse
import json
import time
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


def load_in_progress_runs(results_dir: Path) -> list[dict]:
    """Find all results.json files for evaluations that are not yet complete.

    Returns list of dicts with progress data, sorted by start time (most recent first).
    """
    runs = []

    for path in results_dir.rglob('results.json'):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        checkpoint = data.get('checkpoint', {})
        completed = checkpoint.get('completed_rounds', 0)
        total = checkpoint.get('total_rounds', 0)

        if completed == 0 or total == 0:
            continue

        player_model = data.get('config', {}).get('player_model', '?')
        total_seconds = data.get('statistics', {}).get('total_wall_clock_seconds', 0)

        runs.append({
            'path': path,
            'folder': path.parent.name,
            'player_model': player_model,
            'completed': completed,
            'total': total,
            'done': completed >= total,
            'total_seconds': total_seconds,
            'timestamp': data.get('timestamp', ''),
        })

    # Sort: in-progress first, then by timestamp descending
    runs.sort(key=lambda r: (r['done'], r['timestamp']), reverse=True)
    return runs


def show_progress(results_dir: Path, watch: bool, interval: int) -> None:
    """Display progress + ETA for evaluations currently running (or recently completed)."""

    def _render():
        runs = load_in_progress_runs(results_dir)
        now = time.strftime('%H:%M:%S')

        in_progress = [r for r in runs if not r['done']]
        recently_done = [r for r in runs if r['done']][:5]

        print()
        print('=' * 68)
        print(f'  PARALLEL EVALUATION — LIVE PROGRESS   [{now}]')
        print('=' * 68)

        if not runs:
            print('  No evaluation data found in', results_dir)
            print('=' * 68)
            print()
            return

        if in_progress:
            print(f'  IN PROGRESS ({len(in_progress)} model{"s" if len(in_progress) > 1 else ""})')
            print()

            col = max(len(r['player_model']) for r in in_progress) + 2

            for r in in_progress:
                completed = r['completed']
                total = r['total']
                elapsed = r['total_seconds']
                avg_per_round = elapsed / completed if completed > 0 else 0
                remaining = total - completed
                eta_seconds = avg_per_round * remaining

                pct = completed / total * 100
                bar_width = 20
                filled = int(bar_width * completed / total)
                bar = '█' * filled + '░' * (bar_width - filled)

                model = r['player_model']
                progress_str = f"{completed}/{total} ({pct:.0f}%)"
                eta_str = format_duration(eta_seconds) if avg_per_round > 0 else '?'
                elapsed_str = format_duration(elapsed)

                print(f'  {model:<{col}}  [{bar}]  {progress_str:<12}  elapsed {elapsed_str:<8}  ETA {eta_str}')

        if recently_done:
            if in_progress:
                print()
            print(f'  COMPLETED (last {len(recently_done)})')
            print()
            col = max(len(r['player_model']) for r in recently_done) + 2
            for r in recently_done:
                total_str = format_duration(r['total_seconds'])
                avg = r['total_seconds'] / r['completed'] if r['completed'] > 0 else 0
                print(f'  {r["player_model"]:<{col}}  {r["completed"]}/{r["total"]} rounds done  total {total_str}  ({avg:.0f}s/round avg)')

        print()
        print('=' * 68)
        print()

    if watch:
        try:
            while True:
                _render()
                print(f'  Refreshing every {interval}s — Ctrl+C to stop')
                print()
                time.sleep(interval)
        except KeyboardInterrupt:
            print('\nStopped.')
    else:
        _render()


def show_estimate(models_file: str, config_file: str, results_dir: Path) -> None:
    """Estimate duration of a future parallel run from historical timing data."""
    models = load_models_file(models_file)
    config = load_config(config_file)

    num_rules = count_rules(config)
    num_rounds_per_rule = config.get('game', {}).get('num_rounds_per_rule', 1)
    total_rounds = num_rules * num_rounds_per_rule

    print()
    print('=' * 62)
    print('  PARALLEL EVALUATION — TIME ESTIMATE')
    print('=' * 62)
    print(f'  Config  : {config_file}')
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


def main():
    parser = argparse.ArgumentParser(
        description='Estimate parallel evaluation time / show live progress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate duration of a future run (uses historical data)
  python scripts/estimate_parallel_time.py eval_models.txt
  python scripts/estimate_parallel_time.py eval_models.txt test.yaml

  # Show progress + ETA for evaluations currently running
  python scripts/estimate_parallel_time.py --progress
  python scripts/estimate_parallel_time.py --progress --watch
  python scripts/estimate_parallel_time.py --progress --watch --interval 30
""",
    )
    parser.add_argument('models_file', nargs='?',
                        help='File with model keys (one per line) — required in estimate mode')
    parser.add_argument('config_file', nargs='?', default='config.yaml',
                        help='Config YAML file (default: config.yaml)')
    parser.add_argument('--results-dir', default='results',
                        help='Results directory to scan (default: results/)')
    parser.add_argument('--progress', action='store_true',
                        help='Show live progress + ETA for in-progress evaluations')
    parser.add_argument('--watch', action='store_true',
                        help='Refresh progress display continuously (use with --progress)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Refresh interval in seconds for --watch (default: 60)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.progress:
        show_progress(results_dir, watch=args.watch, interval=args.interval)
    else:
        if not args.models_file:
            parser.error('models_file is required in estimate mode (or use --progress)')
        show_estimate(args.models_file, args.config_file, results_dir)


if __name__ == '__main__':
    main()
