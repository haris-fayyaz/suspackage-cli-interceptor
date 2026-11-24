"""Load the feature table, pick model inputs and build split groups."""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)

META_COLS = ("row_id", "name", "version", "source", "label", "family", "ok", "error")
DENSITY_BASE = "x_py_kb"  # the only x_ column the model may read, and only for densities
MALICIOUS_SOURCES = ("datadog", "pypi_malregistry")
BENIGN_SOURCES = ("top_pypi", "pypi_random")

# Counts that scale with package size: per-KB densities are built for these.
DENSITY_COLS = (
    "c_exec_eval", "c_subprocess", "c_socket", "c_http", "c_b64_decode", "c_decompress",
    "c_ctypes", "c_environ", "c_sensitive_paths", "c_webhook_urls", "c_ip_literals",
)
# Count-like columns that get log1p for the linear model.
COUNT_COLS = DENSITY_COLS + (
    "sp_cmdclass", "sp_subprocess", "sp_network", "sp_exec_eval", "sp_base64",
    "longest_blob_len", "meta_n_deps", "name_len", "name_digits", "name_hyphens", "name_edit_dist",
)


def is_feature_column(col: str) -> bool:
    """A model input is anything that is not metadata and does not start with x_."""
    return col not in META_COLS and not col.startswith("x_")


def load_features(path: str | Path) -> pd.DataFrame:
    """Read features.csv. low_memory=False because `version` has mixed types."""
    return pd.read_csv(path, low_memory=False)


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose archive was read (ok == 1), with a fresh 0..n-1 index."""
    out = df[df["ok"] == 1].reset_index(drop=True)
    log.info("usable rows: %d of %d (malicious %d, benign %d)",
             len(out), len(df), int((out.label == 1).sum()), int((out.label == 0).sum()))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns in file order (no metadata, no x_ columns)."""
    return [c for c in df.columns if is_feature_column(c)]


def input_columns(df: pd.DataFrame) -> list[str]:
    """Columns the model pipeline reads: features plus x_py_kb (used only for densities)."""
    cols = feature_columns(df)
    if DENSITY_BASE in df.columns:
        cols.append(DENSITY_BASE)
    return cols


def feature_vector_hash(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """One 64-bit hash per row of the feature vector. Equal vectors share a hash."""
    return pd.util.hash_pandas_object(df[list(cols)], index=False).to_numpy()


def build_split_groups(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Split groups: rows are linked when they share a family OR an identical feature vector.

    Connected components of that graph become the groups. A split on these groups cannot
    put a near-copy in train and its twin in test.
    """
    n = len(df)
    fam = pd.factorize(df["family"].astype(str))[0]
    vec = pd.factorize(feature_vector_hash(df, cols))[0]
    n_fam, n_vec = int(fam.max()) + 1, int(vec.max()) + 1
    rows = np.arange(n)
    r = np.concatenate([rows, rows])
    c = np.concatenate([n + fam, n + n_fam + vec])
    size = n + n_fam + n_vec
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(size, size))
    _, comp = connected_components(graph, directed=False)
    return pd.factorize(comp[:n])[0]


def group_stats(groups: np.ndarray) -> dict:
    """Size distribution of split groups, for the report."""
    sizes = np.bincount(groups)
    top = np.sort(sizes)[::-1]
    return {
        "n_groups": int(len(sizes)),
        "max_size": int(top[0]),
        "p99_size": float(np.percentile(sizes, 99)),
        "mean_size": float(sizes.mean()),
        "rows_in_top20_groups": int(top[:20].sum()),
        "top20_sizes": [int(s) for s in top[:20]],
        "share_largest": float(top[0] / sizes.sum()),
    }


def family_stats(df: pd.DataFrame) -> dict:
    """Size distribution of the raw `family` column, for the report."""
    sizes = df.groupby("family").size().sort_values(ascending=False)
    return {
        "n_families": int(len(sizes)),
        "max_size": int(sizes.iloc[0]),
        "top20": {str(k): int(v) for k, v in sizes.head(20).items()},
    }


def data_version(dvc_file: str | Path | Sequence[str | Path] = ("data/features.dvc", "features.dvc"),
                 csv_path: str | Path | None = None) -> str:
    """md5 of the tracked feature folder from its .dvc pointer, else a hash of the CSV.

    `dvc add data/features` writes data/features.dvc. Several candidate paths are tried.
    """
    candidates = [dvc_file] if isinstance(dvc_file, (str, Path)) else list(dvc_file)
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            m = re.search(r"md5:\s*([0-9a-f]{32}(?:\.dir)?)", p.read_text())
            if m:
                return m.group(1)
    if csv_path and Path(csv_path).exists():
        h = hashlib.md5()
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "csv-md5:" + h.hexdigest()
    return "unknown"
