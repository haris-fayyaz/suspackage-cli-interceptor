# SusPackage

A machine learning CLI interceptor for typosquatted and slopsquatted package detection.
Group 6, BCS-7F, Information Security, Fall 2026.

## Layout
- `src/suspackage/archive.py`: reads a package archive in memory (never extracts, never executes).
- `src/suspackage/features/`: feature modules (setup_py, code, metadata, name), shared by training and the CLI.
- `src/suspackage/extract.py`: builds `features.csv` from the label table (resumable).
- `params.yaml`, `dvc.yaml`: parameters and the data pipeline.
- `data/labels/`: label tables (paths and labels only, no malware).
- `docs/data_sources/`: READMEs of the source datasets.

## Rules
- Malware archives are never committed. They live only in the Colab runtime.
- Columns starting with `x_` in `features.csv` are metadata and are excluded from training.
