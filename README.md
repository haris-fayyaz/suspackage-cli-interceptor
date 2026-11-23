# SusPackage

A machine learning CLI interceptor for typosquatted and slopsquatted package detection.
Group 6, BCS-7F, Information Security, Fall 2026.

## Docs
- [Data](docs/data.md): where the data comes from, labels, features, limits
- [Pipeline](docs/pipeline.md): setup, rebuilding features, rules

## Layout
- `src/suspackage/archive.py`: reads a package archive in memory (never extracts, never executes)
- `src/suspackage/features/`: feature modules, shared by training and the CLI
- `src/suspackage/extract.py`: builds the feature table (resumable)
- `src/suspackage/sample_benign.py`: adds random benign packages
- `params.yaml`: settings for extraction and sampling
- `data/labels/`, `data/features/`: tracked with DVC
- `tests/`: unit tests (`pytest`)

## Rules
- Malware archives are never committed. They exist only in the Colab session that built the features.
- Columns starting with `x_` in `features.csv` are metadata and are not used for training.# SusPackage

A machine learning CLI interceptor for typosquatted and slopsquatted package detection.
Group 6, BCS-7F, Information Security, Fall 2026.

## Docs
- [Data](docs/data.md): where the data comes from, labels, features, limits
- [Pipeline](docs/pipeline.md): setup, rebuilding features, rules

## Layout
- `src/suspackage/archive.py`: reads a package archive in memory (never extracts, never executes)
- `src/suspackage/features/`: feature modules, shared by training and the CLI
- `src/suspackage/extract.py`: builds the feature table (resumable)
- `src/suspackage/sample_benign.py`: adds random benign packages
- `params.yaml`: settings for extraction and sampling
- `data/labels/`, `data/features/`: tracked with DVC
- `tests/`: unit tests (`pytest`)

## Rules
- Malware archives are never committed. They exist only in the Colab session that built the features.
- Columns starting with `x_` in `features.csv` are metadata and are not used for training.