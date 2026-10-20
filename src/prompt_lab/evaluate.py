"""Run each technique over the dataset and score the results."""

import csv
from collections import Counter
from pathlib import Path

from . import LABELS
from .parsing import parse_label
from .templates import render


def load_dataset(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"Row {row['id']} has invalid label '{row['label']}'")
    return rows


def run_technique(technique: str, client, rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        prompt = render(technique, row["text"])
        output = client.complete(prompt, source_text=row["text"])
        predicted = parse_label(output)
        results.append({
            "technique": technique,
            "id": row["id"],
            "text": row["text"],
            "expected": row["label"],
            "predicted": predicted or "",
            "correct": predicted == row["label"],
            "raw_output": output.strip(),
        })
    return results


def score(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    unparsed = sum(not r["predicted"] for r in results)

    per_label = {}
    for label in LABELS:
        subset = [r for r in results if r["expected"] == label]
        hits = sum(r["correct"] for r in subset)
        per_label[label] = hits / len(subset) if subset else None

    confusions = Counter(
        (r["expected"], r["predicted"] or "unparsed")
        for r in results if not r["correct"]
    )
    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "unparsed": unparsed,
        "per_label": per_label,
        "top_confusions": confusions.most_common(3),
    }


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summary_table(scores: dict[str, dict]) -> str:
    """Markdown table comparing techniques, best first."""
    header = "| Technique | Accuracy | Positive | Negative | Neutral | Unparsed |"
    sep = "|---|---|---|---|---|---|"

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    lines = [header, sep]
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["accuracy"]):
        pl = s["per_label"]
        lines.append(
            f"| {name} | {pct(s['accuracy'])} | {pct(pl['positive'])} | "
            f"{pct(pl['negative'])} | {pct(pl['neutral'])} | {s['unparsed']} |"
        )
    return "\n".join(lines)
