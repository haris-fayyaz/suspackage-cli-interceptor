"""Add randomly chosen PyPI packages as extra benign rows (unverified).

Why: the top-downloaded packages are all polished, so a model could learn "polished = safe".
Random small packages teach it that small or plain packages are not automatically bad.

    python -m suspackage.sample_benign            # uses params.yaml -> random_benign
"""
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from glob import glob

import pandas as pd
import requests
from tqdm import tqdm

from .config import load_params
from .features.name import norm

H = {"User-Agent": "suspackage-coursework (FAST-NUCES student project)"}


def all_names():
    r = requests.get("https://pypi.org/simple/", timeout=120,
                     headers={**H, "Accept": "application/vnd.pypi.simple.v1+json"})
    r.raise_for_status()
    return [p["name"] for p in r.json()["projects"]]


def excluded_names(P, labels):
    ex = set(labels["name"].map(norm))
    rb = P["random_benign"]
    if os.path.isdir(rb.get("datadog_compromised", "")):
        ex |= {norm(os.path.basename(d)) for d in glob(rb["datadog_compromised"] + "/*")}
    if os.path.exists(rb.get("datadog_manifest", "")):
        m = json.load(open(rb["datadog_manifest"]))
        ex |= {norm(k) for k in (m if isinstance(m, dict) else [x[0] for x in m])}
    return ex


def fetch(name, rb):
    d = f"{rb['out_dir']}/{name}"
    os.makedirs(d, exist_ok=True)
    have = [f for f in os.listdir(d) if f.endswith((".tar.gz", ".zip"))]
    if have:
        return name, f"{d}/{have[0]}", "ok"
    last = ""
    for attempt in range(3):
        try:
            r = requests.get(f"https://pypi.org/pypi/{name}/json", timeout=20, headers=H)
            if r.status_code == 404:
                return name, "", "not_found"
            r.raise_for_status()
            j = r.json()
            times = [f["upload_time_iso_8601"] for fs in j["releases"].values() for f in fs]
            if not times:
                return name, "", "no_files"
            first = datetime.fromisoformat(min(times).replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - first).days < rb["min_age_days"]:
                return name, "", "too_new"     # a package that survived a while is less likely to be unreported malware
            sd = [u for u in j["urls"] if u["packagetype"] == "sdist" and u["filename"].endswith((".tar.gz", ".zip"))]
            if not sd:
                return name, "", "no_sdist"
            u = sd[0]
            if u["size"] > rb["max_mb"] * 1e6:
                return name, "", "too_big"
            data = requests.get(u["url"], timeout=60, headers=H).content
            if hashlib.sha256(data).hexdigest() != u["digests"]["sha256"]:
                raise ValueError("hash mismatch")
            path = f"{d}/{u['filename']}"
            with open(path, "wb") as f:
                f.write(data)
            return name, path, "ok"
        except Exception as e:
            last = str(e)[:50]
            time.sleep(2 ** attempt)
    return name, "", "error: " + last


def main():
    P = load_params()
    rb, paths = P["random_benign"], P["paths"]
    labels = pd.read_csv(paths["labels"], keep_default_na=False)
    if (labels["source"] == "pypi_random").any():
        print("pypi_random rows already exist. Remove them first if you want to redo this.")
        return
    ex = excluded_names(P, labels)
    names = [n for n in dict.fromkeys(norm(n) for n in all_names()) if n not in ex]
    random.Random(rb["seed"]).shuffle(names)
    names = names[:rb["n_request"]]
    print(f"{len(names)} random names requested (excluded {len(ex)} known names)")

    os.makedirs(rb["out_dir"], exist_ok=True)
    res = {}
    with ThreadPoolExecutor(rb["workers"]) as ex_:
        futs = [ex_.submit(fetch, n, rb) for n in names]
        for f in tqdm(as_completed(futs), total=len(futs)):
            n, p, s = f.result()
            res[n] = (p, s)

    status = pd.Series({n: s for n, (p, s) in res.items()})
    print(status.str.split(":").str[0].value_counts())
    status[status != "ok"].rename("reason").to_csv(paths["labels"].replace("labels_final", "skipped_random"))
    new = pd.DataFrame([dict(name=n, version="", source="pypi_random", label=0, path=p, family=n, rank="", download_count="")
                        for n, (p, s) in res.items() if s == "ok"])
    out = pd.concat([labels, new], ignore_index=True)
    out.to_csv(paths["labels"], index=False)
    print(out.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
