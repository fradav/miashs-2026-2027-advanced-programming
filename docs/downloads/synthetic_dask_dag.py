"""Synthetic Dask DAG generator - simple starter

This script builds a random, acyclic task graph using dask.delayed.
It can visualize the resulting graph (requires graphviz) and optionally
run a small computation to exercise the scheduler.

Usage examples:
  python Projects/synthetic_dask_dag.py --n 50 --max-deps 3 --out graph.png
  python Projects/synthetic_dask_dag.py --n 20 --max-deps 2 --run --scheduler threads

This is intentionally small and meant as a starting point for Project (14): Synthetic Dask DAGs.
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Any, Dict, List

try:
    from dask import delayed
except Exception:  # pragma: no cover - helpful message when dask isn't installed
    raise SystemExit("dask is required: pip install 'dask[complete]' or 'dask'")


def make_random_dag(n: int, max_deps: int = 2, seed: int | None = None, min_time: float = 0.01, max_time: float = 0.2):
    """Generate a simple DAG of dask.delayed tasks.

    - n: number of tasks
    - max_deps: maximum number of dependencies a task may have (only to earlier tasks)
    - seed: random seed
    - min_time/max_time: simulated work time (seconds)

    Returns a tuple (tasks, roots) where `tasks` is a dict index->delayed and
    `roots` is a list of delayed objects that have no dependents (useful as outputs).
    """
    rand = random.Random(seed)

    # lightweight work function
    def work(i: int, sleep_for: float):
        # tiny simulated CPU work + sleep to emulate duration
        s = 0
        for j in range(10):
            s += (i + j) % 7
        time.sleep(sleep_for)
        return i + s

    tasks: Dict[int, Any] = {}
    # track dependents count to find roots (nodes with no dependents)
    dependents_count: Dict[int, int] = {i: 0 for i in range(n)}

    for i in range(n):
        # only allow deps on earlier tasks to ensure acyclic graph
        possible = list(range(0, i))
        k = rand.randint(0, min(len(possible), max_deps))
        deps = rand.sample(possible, k=k) if k > 0 else []

        sleep_for = rand.uniform(min_time, max_time)

        dep_objs = [tasks[d] for d in deps]

        # helper that accepts dependency results but ignores them
        def work_with_deps(idx: int, t: float, *_) -> int:
            return work(idx, t)

        if deps:
            tasks[i] = delayed(work_with_deps)(i, sleep_for, *dep_objs)
            for d in deps:
                dependents_count[d] += 1
        else:
            tasks[i] = delayed(work)(i, sleep_for)

    # roots: tasks that have zero dependents (no one depends on them)
    roots = [tasks[i] for i, c in dependents_count.items() if c == 0]
    return tasks, roots


def visualize(roots: List[Any], out: str | None = None):
    """Visualize the graph defined by roots.

    If `out` is provided, writes to that path (e.g., PNG). Otherwise returns the graphviz object.
    """
    from dask import visualize as dask_visualize

    if out:
        dask_visualize(roots, filename=out)
        return out

    return dask_visualize(roots)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="number of tasks")
    parser.add_argument("--max-deps", type=int, default=3, help="maximum dependencies per task")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="path to write graph visualization (png/pdf)")
    parser.add_argument("--run", action="store_true", help="actually compute the DAG (small run)")
    parser.add_argument("--scheduler", choices=["threads", "processes", "sync"], default="sync", help="dask scheduler to use for the smoke run")
    args = parser.parse_args()

    tasks, roots = make_random_dag(args.n, args.max_deps, args.seed)

    print(f"Generated DAG with {len(tasks)} tasks; {len(roots)} roots (final outputs)")

    if args.out:
        print(f"Visualizing to {args.out} (requires graphviz installed)")
        try:
            visualize(roots, out=args.out)
            print("Wrote visualization")
        except Exception as e:
            print("Visualization failed:", e)

    if args.run:
        # small smoke run
        print(f"Executing graph with scheduler={args.scheduler} (this may take a moment)")
        from dask import config
        from dask import compute

        # choose simple scheduler
        if args.scheduler == "sync":
            with config.set(scheduler="synchronous"):
                res = compute(*roots)
        else:
            res = compute(*roots, scheduler=args.scheduler)

        print("Run finished; sample outputs:", res[:5])


if __name__ == "__main__":
    main()
