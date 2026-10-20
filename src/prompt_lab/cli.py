"""Command-line entry point.

Examples:
    prompt-lab                                   # offline mock run
    prompt-lab --provider anthropic --model claude-haiku-5-5
    prompt-lab --techniques zero_shot few_shot --limit 10
"""

import argparse
from datetime import datetime
from pathlib import Path

from .evaluate import load_dataset, run_technique, score, summary_table, write_results
from .llm import get_client
from .templates import available_techniques

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare prompt engineering techniques.")
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai"])
    parser.add_argument("--model", help="Model name (required unless provider is mock)")
    parser.add_argument("--techniques", nargs="+", default=available_techniques())
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--limit", type=int, help="Only use the first N rows")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    client = get_client(args.provider, args.model)
    rows = load_dataset(args.data)[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / f"{args.provider}-{stamp}"

    all_results, scores = [], {}
    for technique in args.techniques:
        print(f"Running {technique} on {len(rows)} reviews...")
        results = run_technique(technique, client, rows)
        all_results.extend(results)
        scores[technique] = score(results)

    write_results(all_results, run_dir / "predictions.csv")
    table = summary_table(scores)
    (run_dir / "summary.md").write_text(
        f"# Results: {args.provider} {args.model or ''}\n\n{table}\n", encoding="utf-8"
    )

    print("\n" + table)
    if args.provider == "mock":
        print("\nNote: mock mode is a keyword baseline and ignores the prompts, "
              "so all techniques score the same. Use a real provider to compare them.")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
