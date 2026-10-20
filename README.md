# prompt-lab

A small, testable framework for comparing **prompt engineering techniques** on the same task, using measured results instead of guesswork.

The task is product-review sentiment classification (positive / negative / neutral). The project runs each prompting technique over a labeled dataset and reports accuracy per technique and per label, plus the reviews each one got wrong.

## Techniques compared

| Technique | File | Idea |
|---|---|---|
| Zero-shot | `prompts/zero_shot.txt` | Plain instruction, no examples |
| Few-shot | `prompts/few_shot.txt` | Three labeled examples before the input |
| Chain-of-thought | `prompts/chain_of_thought.txt` | Reason about cues first, then `Answer: <label>` |
| Role-based | `prompts/role_based.txt` | Expert persona with known pitfalls spelled out |

To add a technique, drop a new `.txt` file in `prompts/` with a `$text` placeholder. It's picked up automatically.

## Dataset

`data/reviews.csv` holds 30 hand-written reviews (10 positive, 11 negative, 9 neutral). They deliberately include hard cases:

- **Sarcasm**, e.g. "Oh great, another phone case that cracks..."
- **Mixed opinions**, e.g. "The screen is gorgeous, but the speakers are so bad I returned it."
- **Flat factual statements** that look like content but carry no sentiment

## Quick start

```bash
git clone https://github.com/<ikokougochukwu69-glitch>/prompt-lab.git
cd prompt-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest            # run the tests
prompt-lab        # offline run with the mock baseline
```

### Run against a real model

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...        # Windows: set ANTHROPIC_API_KEY=...
prompt-lab --provider anthropic --model claude-haiku-5-5
```

OpenAI works the same way: `pip install -e ".[openai]"`, set `OPENAI_API_KEY`, and pass `--provider openai --model <model-name>`.

Useful flags:

- `--techniques zero_shot few_shot` runs a subset of techniques.
- `--limit 10` uses only the first 10 reviews (cheap trial runs).
- `--data path/to/other.csv` uses your own dataset (columns: `id,text,label`).

Each run writes to `results/<provider>-<timestamp>/`:

- `predictions.csv`: every prompt's raw model output, parsed label, and whether it was correct
- `summary.md`: comparison table, best technique first

> **About mock mode:** the mock client is a keyword-counting baseline so the pipeline runs without an API key. It ignores the prompts, so every technique gets the same score. Its keyword list was also written with this dataset in view, so its score is optimistic and is not a fair baseline. It exists to test the plumbing. Use a real provider to compare techniques.

## Project structure

```
prompt-lab/
├── data/reviews.csv          # labeled evaluation set
├── prompts/                  # one template per technique
├── src/prompt_lab/
│   ├── templates.py          # load + render templates
│   ├── llm.py                # mock / Anthropic / OpenAI clients
│   ├── parsing.py            # free-form output -> label
│   ├── evaluate.py           # run techniques, score, save
│   └── cli.py                # command-line entry point
└── tests/
```

## Design notes

- **Temperature 0** for real providers, so differences come from the prompt rather than sampling noise.
- **Robust parsing.** Models don't always answer in one word. The parser looks for an `Answer:` line first, then falls back to the last label mentioned. Outputs it can't parse are counted separately rather than silently scored as wrong.
- **Per-label accuracy** matters because a prompt can look good overall while failing a whole class. Neutral is usually the weak spot.

## Limitations

- 30 examples is enough to spot large differences, but not to call a 3–5 point gap meaningful. Use a larger dataset for real decisions.
- Chain-of-thought costs more tokens per call. Weigh accuracy gains against cost.

## License

MIT
