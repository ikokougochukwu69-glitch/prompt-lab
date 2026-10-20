"""Turn free-form model output into one of the allowed labels."""

import re

from . import LABELS

_ANSWER_LINE = re.compile(r"answer\s*:\s*(\w+)", re.IGNORECASE)
_LABEL_WORD = re.compile(r"\b(" + "|".join(LABELS) + r")\b", re.IGNORECASE)


def parse_label(output: str) -> str | None:
    """Extract a sentiment label, or None if the output can't be parsed.

    Order of preference:
    1. An explicit "Answer: <label>" line (chain-of-thought format).
    2. The last label word in the output (models often restate the answer
       at the end after discussing other labels).
    """
    if not output:
        return None

    answers = _ANSWER_LINE.findall(output)
    if answers and answers[-1].lower() in LABELS:
        return answers[-1].lower()

    words = _LABEL_WORD.findall(output)
    return words[-1].lower() if words else None
