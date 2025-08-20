# Contrastive Embedder Training Template

This repository provides a modular framework for training and evaluating contrastive embedding models using pair-based contrastive learning.

## Documentation

Detailed guides and customization instructions are organized by topic:

- [Models](src/documents/MODELS/README.md) — available architectures, model customization, and configuration.
- [Datasets](src/documents/DATASETS/README.md) — dataset formats, integration steps, and config examples.
- [Transfer Learning](src/documents/TRANSFER_LEARNING/README.md) — fine-tuning strategies and best practices.
- [Training](src/documents/TRAIN.md) — detailed instructions for using `train.py`.
- [Evaluation](src/documents/EVAL.md) — detailed instructions for using `eval.py`.

---

## Quick Start

**Train with default settings**

```bash
python src/train.py
```

**Train with a specific dataset and model**

```bash
python src/train.py dataset=my_dataset model=my_model
```

---

## Evaluation

**Evaluate the latest run from an experiment**

```bash
python src/eval.py --experiment-name "osnet_ain_contrastive_reid_training"
```

**Evaluate a specific run**

```bash
python src/eval.py --run-id <RUN_ID>
```

**List available experiments**

```bash
python src/eval.py --list-experiments
```

**List runs from a specific experiment**

```bash
python src/eval.py --list-runs "experiment_name"
```

---

## Evaluation Options

| Option                   | Description                       |
| ------------------------ | --------------------------------- |
| `--run-id TEXT`          | Specific MLflow run ID            |
| `--experiment-name TEXT` | Experiment name (uses latest run) |
| `--tracking-uri TEXT`    | Custom MLflow server              |
| `--no-plots`             | Skip saving plots                 |
| `--no-mlflow-log`        | Don't log results to MLflow       |

---

## Configuration

This repository uses [Hydra](https://hydra.cc) for configuration management.

**Key config directories:**

- `configs/config.yaml` — base config
- `configs/dataset/` — dataset configs
- `configs/model/` — model configs

You can override configuration values on the command line. Example:

```bash
python src/train.py dataset=my_dataset train.epochs=50
```

---

## Output & Logging

- **Checkpoints** are saved in `outputs/<date>/<time>/checkpoints/`
- **Logs & metrics** are tracked via [MLflow](https://mlflow.org/)

---

## Notes & Links

- For dataset integration and examples, see `src/documents/DATASETS/README.md`.
- For model customization, see `src/documents/MODELS/README.md`.
- For transfer-learning / pruning workflows, see `src/documents/TRANSFER_LEARNING/README.md`.
- If you want, I can also create `src/documents/TRAIN.md` and `src/documents/EVAL.md` matching this style.
