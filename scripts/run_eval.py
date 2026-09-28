"""Run the evaluation harness against the provided benchmarks.

Usage:
    python scripts/run_eval.py                          # retrieval benchmark, Recall@10 / MRR / nDCG
    python scripts/run_eval.py --benchmark qa_test      # document/section hit rates on the QA test split
    python scripts/run_eval.py --benchmark hard_negatives
    python scripts/run_eval.py --limit 50               # smaller sample
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.container import get_services  # noqa: E402
from app.evaluation.evaluator import EvalError, run_benchmark, save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="retrieval", help="retrieval | qa_test | qa_validation | hard_negatives")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    services = get_services()
    try:
        report = run_benchmark(services, benchmark=args.benchmark, k=args.k, limit=args.limit)
    except EvalError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 2
    path = save_report(report, services.settings.paths.eval_dir)
    print(json.dumps(report.to_dict(max_cases=10), indent=2))
    print(f"\nfull report saved to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())