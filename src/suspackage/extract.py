"""Build features.csv from the label table. Resumable; reads archives in memory only.

    python -m suspackage.extract --limit 200     # quick test on 200 packages
    python -m suspackage.extract                 # full run
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from .archive import read_package
from .config import load_params
from .features import code, metadata, setup_py
from .features.name import NameFeatures, norm

_CFG = _NAMER = None


def featurize(name, path, cfg, namer):
    """Features for one package archive. The CLI reuses this at runtime."""
    pkg = read_package(path, cfg)
    if pkg.is_wheel:
        raise ValueError("wheel archive, skipped")
    f = {}
    f.update(setup_py.features(pkg.setup_py))
    f.update(code.features(pkg.py))
    f.update(metadata.features(pkg.pkg_info))
    f.update(namer(name))
    return f


def _init(cfg, popular):
    global _CFG, _NAMER
    _CFG, _NAMER = cfg, NameFeatures(popular)


def _work(task):
    rid, name, path = task
    try:
        return rid, {"ok": 1, **featurize(name, path, _CFG, _NAMER)}
    except Exception as e:  # corrupt archive, bad password, etc.
        return rid, {"ok": 0, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="test run size (half per class)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()

    P = load_params(a.params)
    cfg, paths = P["extract"], P["paths"]
    labels = pd.read_csv(paths["labels"], keep_default_na=False)
    labels["row_id"] = labels.index
    labels = labels[~labels["path"].str.endswith(".whl")]  # benign data has no wheels
    out = a.out or (paths["features"].replace(".csv", "_test.csv") if a.limit else paths["features"])
    if a.limit:
        half = a.limit // 2
        labels = pd.concat([labels[labels.label == l].sample(min(half, int((labels.label == l).sum())), random_state=0)
                            for l in (0, 1)])

    pop = pd.read_csv(paths["popular"]).sort_values("download_count", ascending=False)
    popular = pop["project"].map(norm).head(cfg["popular_top_n"]).tolist()

    meta_cols = ["row_id", "name", "version", "source", "label", "family", "ok", "error"]
    feat_cols = (list(setup_py.features("")) + list(code.features({})) + list(metadata.features(""))
                 + list(NameFeatures(popular)("example")))
    cols = meta_cols + feat_cols

    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set(pd.read_csv(out, usecols=["row_id"])["row_id"]) if os.path.exists(out) else set()
    todo = labels[~labels.row_id.isin(done)]
    print(f"{len(labels)} rows | already done {len(done)} | to do {len(todo)} -> {out}")
    if todo.empty:
        return

    meta = todo.set_index("row_id")
    tasks = [(r, meta.at[r, "name"], meta.at[r, "path"]) for r in todo.row_id]
    with ProcessPoolExecutor(cfg["workers"], initializer=_init, initargs=(cfg, popular)) as ex:
        for i in range(0, len(tasks), cfg["chunk"]):
            batch = tasks[i:i + cfg["chunk"]]
            rows = []
            for rid, feats in tqdm(ex.map(_work, batch, chunksize=8), total=len(batch), leave=False):
                m = meta.loc[rid]
                rows.append({"row_id": rid, "name": m["name"], "version": m["version"], "source": m["source"],
                             "label": m["label"], "family": m["family"], **feats})
            pd.DataFrame(rows).reindex(columns=cols).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"checkpoint: {min(i + cfg['chunk'], len(tasks))}/{len(tasks)}")
    df = pd.read_csv(out)
    print(df.groupby(["label", "ok"]).size())
    print(df.loc[df.ok == 0, "error"].value_counts().head(5))


if __name__ == "__main__":
    main()
