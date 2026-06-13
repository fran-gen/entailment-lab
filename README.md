# Entailment Lab

A local-first framework for evaluating LLMs on semantic relations such as entailment, contradiction, and neutrality.

## Why Entailment Lab

Natural Language Inference (NLI) is a well-established task, but practical tooling for **systematic local-model evaluation** is still fragmented.

Entailment Lab focuses on engineering usability:

- local-first model evaluation (`Ollama`, `llama.cpp`, custom adapters)
- reproducible runs across datasets, prompts, and decoding settings
- comparable outputs and metrics across models
- extensibility for custom semantic stress tests (negation, quantifiers, temporal reasoning, etc.)

## Scope and Positioning

Entailment Lab does **not** claim to introduce a new task.
It packages NLI-style evaluation into a practical workflow for developers and applied ML teams who need reliable semantic benchmarking for local LLMs.

Given a `(premise, hypothesis)` pair, the model predicts one label:

- `entailment`
- `contradiction`
- `neutral`

## Core Features (MVP)

- JSONL dataset support for immediate custom benchmarking
- local model adapters (starting with Ollama)
- strict prompt templates for deterministic classification
- robust output parsing with invalid-output tracking
- essential metrics:
  - accuracy
  - macro-F1
  - per-label precision/recall/F1
  - confusion matrix
  - invalid output rate
- saved per-example predictions for error analysis

## Planned Extensions

- Support for  SNLI dataset
- Prompt suite comparisons (zero-shot / few-shot / strict JSON)
- robustness slices (negation, lexical overlap traps, quantifiers)
- report generation (Markdown + plots)
- CI regression checks for model updates

## Repository Layout (Target)

```text
entailment-lab/
├── README.md
├── pyproject.toml
├── configs/
│   ├── models.yaml
│   ├── datasets.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── custom/
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── reports/
├── src/
│   └── entailment_lab/
│       ├── cli.py
│       ├── datasets/
│       ├── models/
│       ├── prompts/
│       ├── runners/
│       ├── metrics/
│       └── reporting/
└── tests/
```

## Example Internal Data Format

```json
{
  "id": "example_001",
  "premise": "John is slicing vegetables in the kitchen.",
  "hypothesis": "John is slicing vegetables.",
  "label": "entailment",
  "metadata": {
    "source": "custom",
    "phenomenon": "subsentence"
  }
}
```

## Example CLI (Planned)

```bash
entailment-lab evaluate \
  --model ollama:llama3.1:8b \
  --dataset data/custom/smoke_test.jsonl \
  --prompt strict_zero_shot \
  --output results/predictions/llama3.1-8b-smoke.jsonl
```

```bash
entailment-lab compare \
  --results results/predictions/*.jsonl \
  --metric macro_f1
```

## Evaluation Philosophy

- prioritize **macro-F1** and confusion analysis over raw accuracy alone
- never silently drop malformed model outputs
- keep prompts explicit and constrained to reduce parser ambiguity
- treat benchmark runs as versioned artifacts for reproducibility

## Getting Started

This repository is currently in bootstrap phase.

### Pixi + Notebook

Install the environment and start Jupyter from Pixi:

```bash
pixi run lab
```

If you are opening the notebook from an editor such as VS Code and do not see the project environment in the kernel picker, register the Pixi interpreter as a named kernel:

```bash
pixi run install-kernel
```

After that, select `Python (entailment-lab)` as the notebook kernel.

Immediate next steps:

1. scaffold the Python package (`src/entailment_lab`)
2. add JSONL dataset loader
3. add Ollama adapter
4. implement strict label parser and baseline metrics
5. wire a minimal CLI command: `evaluate`

## License

TBD

## Contributing

Contributions are welcome. If you want to add a new dataset loader, model adapter, or stress-test suite, open an issue with a brief proposal first.
