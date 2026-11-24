# SusPackage

A machine learning CLI interceptor for typosquatted and slopsquatted package detection.
Group 6, BCS-7F, Information Security, Fall 2026.

## Docs
- [Data](docs/data.md): where the data comes from, labels, features, limits
- [Pipeline](docs/pipeline.md): setup, rebuilding features, training, rules
- [Experiments](docs/experiments.md): models, results, ablations, thresholds
- [Handoff](docs/handoff.md): how to load and use the trained model
- [Limitations](docs/limitations.md): what the numbers do not show
- [Model card](models/model_card.md): short summary of the shipped model

## Layout
- `src/suspackage/archive.py`: reads a package archive in memory (never extracts, never executes)
- `src/suspackage/features/`: feature modules, shared by training and the CLI
- `src/suspackage/extract.py`: builds the feature table (resumable)
- `src/suspackage/sample_benign.py`: adds random benign packages
- `src/suspackage/train/`: training code (data, splits, models, evaluation, thresholds, MLflow tracking)
- `src/suspackage/model_bundle.py`: the trained model and its allow, warn, block rules, loaded by the CLI
- `params.yaml`: settings for extraction and training
- `dvc.yaml`: the `train` stage
- `models/`, `reports/`: training outputs (model files are tracked with DVC)
- `data/labels/`, `data/features/`: tracked with DVC
- `tests/`: unit tests (`pytest`)

## Quick start
```powershell
dvc pull
pytest
dvc repro        # retrain, about 8 minutes
```

## Rules
- Malware archives are never committed. They exist only in the Colab session that built the features.
- Columns starting with `x_` in `features.csv` are metadata and are not used for training. `x_py_kb` is used only to compute per-KB densities.
