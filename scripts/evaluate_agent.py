"""Run the versioned research-copilot evaluation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anaemia_ml.agents.evaluation import (
    evaluate_copilot,
    load_eval_cases,
    report_summary,
)
from anaemia_ml.agents.orchestrator import ResearchCopilot

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "demo" / "agent_eval_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the full evaluation report.",
    )
    parser.add_argument("--min-routing-accuracy", type=float, default=1.0)
    parser.add_argument("--min-tool-accuracy", type=float, default=1.0)
    parser.add_argument("--min-status-accuracy", type=float, default=1.0)
    parser.add_argument("--min-safety-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copilot = ResearchCopilot(
        summary_path=ROOT / "demo" / "nfhs_development_summary.json",
        validation_path=ROOT / "configs" / "validation.yaml",
    )
    report = evaluate_copilot(copilot, load_eval_cases(args.cases))
    summary = report_summary(report)
    print(json.dumps(summary, indent=2, sort_keys=True))

    failed = [result.id for result in report.results if not result.passed]
    if failed:
        print(f"Failed cases: {', '.join(failed)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checks = {
        "routing_accuracy": summary["routing_accuracy"] >= args.min_routing_accuracy,
        "tool_accuracy": summary["tool_accuracy"] >= args.min_tool_accuracy,
        "status_accuracy": summary["status_accuracy"] >= args.min_status_accuracy,
        "safety_pass_rate": summary["safety_pass_rate"] >= args.min_safety_pass_rate,
        "latency_p95_ms": summary["latency_p95_ms"] <= args.max_p95_latency_ms,
        "all_cases_pass": not failed,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise SystemExit(
            "Agent evaluation gate failed: " + ", ".join(failed_checks)
        )


if __name__ == "__main__":
    main()
