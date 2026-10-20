"""Load and render prompt templates from the prompts/ folder."""

from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def available_techniques(prompts_dir: Path = PROMPTS_DIR) -> list[str]:
    """Return technique names, one per .txt file in the prompts folder."""
    return sorted(p.stem for p in prompts_dir.glob("*.txt"))


def load_template(technique: str, prompts_dir: Path = PROMPTS_DIR) -> Template:
    path = prompts_dir / f"{technique}.txt"
    if not path.exists():
        options = ", ".join(available_techniques(prompts_dir))
        raise ValueError(f"Unknown technique '{technique}'. Options: {options}")
    return Template(path.read_text(encoding="utf-8"))


def render(technique: str, text: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Fill a template's $text placeholder with the review text."""
    return load_template(technique, prompts_dir).substitute(text=text.strip())
