# Data

## What we are building
- A checker that reads a Python package before pip installs it.
- It answers allow, warn or block.
- To learn that, the model needs labeled examples: known bad packages (1) and known normal packages (0).

## Where the data comes from
- **Malicious, pypi_malregistry:** about 10,000 known malicious PyPI packages. Source: https://github.com/lxyeternal/pypi_malregistry
- **Malicious, DataDog:** about 2,200 packages, reviewed by hand. Source: https://github.com/DataDog/malicious-software-packages-dataset (only `samples/pypi/malicious_intent` is used).
- **Benign, popular:** the 15,000 most downloaded PyPI packages, sampled evenly across the ranking. Source: https://hugovk.dev/top-pypi-packages/
- **Benign, random:** about 3,700 random PyPI packages, first uploaded at least 180 days ago. These are not checked by hand.

## What each folder in the source repos means
- `package/version/archive`: one folder per package, one per version, and the archive is the package's code.
- `.tar.gz` and `.zip`: source packages. These contain `setup.py` and `PKG-INFO`, which we read.
- `.whl`: pre-built wheels. They have no `setup.py`, so we skip them.
- `malicious_intent` (DataDog): packages built to be malware. Used.
- `compromised_lib` (DataDog): real packages that were hijacked later. Not used, because their code looks normal.
- `manifest.json`: a list of names, not code. Not used.

## The label table (`data/labels/labels_final.csv`)
- One row per package version.
- Columns:
  - `name`: package name
  - `version`: version (empty for benign)
  - `source`: which dataset it came from
  - `label`: 1 = malicious, 0 = benign
  - `path`: where the archive was on the machine that built it (for reference only)
  - `family`: groups versions of one package so train and test never share one
  - `rank`, `download_count`: only for popular benign packages
- `skipped.csv` lists the packages we could not download, and why.

## The feature table (`data/features/features.csv`)
- One row per label row, with the same `name`, `label` and `source`.
- Each feature is a count or a yes/no value, computed from the archive without running it.
- Columns starting with `x_` are for us only. Do not train on them.
- `ok = 0` means the archive could not be read. Those rows are dropped before training.

## Counts (from the last full run)
- Malicious: about 12,290 rows (after dropping wheels and corrupt archives).
- Benign: about 14,670 rows (popular plus random).
- Failed: 903 wheels and 3 corrupt archives.

## Known limits
- Most malicious samples were found by one tool (GuardDog), so the data may miss other kinds of malware.
- Many malicious packages are near-copies of each other, so we split train and test by `family`.
- Some malicious packages are proof-of-concept tests, not real attacks. They are still labeled malicious.
- Popular benign packages are polished. The random benign packages balance this, but they are not verified.
- Wheel-only packages are missing from the benign side.

## Rules
- Malware archives are never committed to Git or DVC.
- They exist only in the Colab session that built the features.
- Git stores the label table and the feature table as DVC pointers.
