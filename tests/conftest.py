"""Synthetic feature tables that look like data/features/features.csv (no real packages)."""
import numpy as np
import pandas as pd
import pytest

FEATURES = (
    "sp_present sp_cmdclass sp_subprocess sp_network sp_exec_eval sp_base64 "
    "c_exec_eval c_subprocess c_socket c_http c_b64_decode c_decompress c_ctypes c_environ "
    "c_sensitive_paths c_webhook_urls c_ip_literals max_blob_entropy longest_blob_len "
    "obfuscation_score parse_fail_ratio meta_present meta_has_description meta_has_homepage "
    "meta_has_author meta_n_deps name_len name_digits name_hyphens name_edit_dist name_edit_ratio is_typosquat"
).split()
INT_COLS = [c for c in FEATURES if c not in ("max_blob_entropy", "obfuscation_score", "parse_fail_ratio", "name_edit_ratio")]


def make_features(n_per_source: int = 150, seed: int = 0) -> pd.DataFrame:
    """Four sources, versioned families, and campaigns of identical vectors under different names."""
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for source, label in [("datadog", 1), ("pypi_malregistry", 1), ("top_pypi", 0), ("pypi_random", 0)]:
        i = 0
        while i < n_per_source:
            fam = f"{source[:3]}-fam{i}"
            n_versions = int(rng.choice([1, 1, 1, 3, 8]))
            base = rng.poisson(3 if label else 0.6, len(FEATURES)).astype(float)
            base[FEATURES.index("sp_present")] = float(rng.random() < (0.9 if label else 0.5))
            base[FEATURES.index("meta_has_homepage")] = float(rng.random() < (0.15 if label else 0.8))
            for v in range(n_versions):
                feats = base + (rng.random(len(FEATURES)) < 0.1) * rng.integers(0, 2, len(FEATURES))
                rows.append({"row_id": rid, "name": f"{fam}{v}" if source != "top_pypi" else fam,
                             "version": str(v), "source": source, "label": label, "family": fam, "ok": 1,
                             "error": "", **dict(zip(FEATURES, feats)), "x_py_files": 3.0,
                             "x_py_kb": float(rng.integers(1, 200)), "x_nearest_popular": "requests"})
                rid += 1
            i += n_versions
    df = pd.DataFrame(rows)
    # a campaign: 30 malicious rows with different names but one identical vector
    camp = df[df.label == 1].iloc[0].copy()
    extra = []
    for k in range(30):
        r = camp.copy()
        r["row_id"], r["name"], r["family"], r["source"] = rid, f"camp{k}", f"camp{k}", "pypi_malregistry"
        extra.append(r)
        rid += 1
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    fail = df.iloc[:5].copy()
    fail["ok"], fail["error"] = 0, "ValueError: wheel archive, skipped"
    fail["row_id"] = range(rid, rid + 5)
    return pd.concat([df, fail], ignore_index=True)


@pytest.fixture(scope="session")
def raw_features():
    return make_features()


@pytest.fixture(scope="session")
def usable(raw_features):
    from suspackage.train import data as D
    return D.usable_rows(raw_features)
