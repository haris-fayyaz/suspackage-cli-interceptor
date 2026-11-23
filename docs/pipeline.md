# Pipeline

## Tools
- **Git:** code, docs, and small text files. Repo: https://dagshub.com/haris-fayyaz/suspackage-cli-interceptor
- **DVC:** the label folder and the feature folder. Git stores only their `.dvc` pointers.
- **MLflow:** experiment runs and metrics (training stage).

## Set up on your machine
```powershell
git clone https://dagshub.com/haris-fayyaz/suspackage-cli-interceptor.git
cd suspackage-cli-interceptor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
dvc pull
```
- `dvc pull` downloads `data/labels` and `data/features`.
- You do not need the malware archives to train or to run the CLI.

## Rebuild the features (Colab only)
- Only needed if the features or the label table change.
- The raw archives live in Colab, so the build runs there.
```python
!python -m suspackage.extract          # full run, resumes if stopped
!python -m suspackage.sample_benign    # adds random benign packages
```
- Then download `data/features/features.csv`, copy it to your repo, and run `dvc add data/features`.

## Where things live
- `src/suspackage/archive.py`: reads archives in memory.
- `src/suspackage/features/`: feature code, shared by training and the CLI.
- `src/suspackage/extract.py`: builds `features.csv`.
- `src/suspackage/sample_benign.py`: adds random benign packages.
- `params.yaml`: settings for extraction and sampling.
- `tests/`: unit tests. Run with `pytest`.

## Rules
- Never run or install code from the malware datasets.
- Never commit archives, zip files or raw samples.
- Keep tokens in Colab Secrets or `.dvc/config.local`, never in Git.
- Run `dvc push` before `git push`, so teammates never see a pointer whose data is missing.
