# Limitations

## How far to trust the numbers
- Scores are very high (holdout PR-AUC 0.994). Part of the reason is the labels. Most malware samples were found by one detector (GuardDog), and our features look for similar signals (exec, subprocess, base64, obfuscation). The model may partly copy that detector.
- Stealthy malware is under-represented. Expect lower recall on packages that hide their payload.
- Duplicates inflate the raw results. 8,768 rows have an identical feature vector, mostly malicious packages with different names but identical features. On unique vectors, random forest recall on the holdout falls from 0.944 to 0.909.
- The cross-dataset test showed no large drop (DataDog recall 0.920). Both malware sets are public lists with overlapping campaigns, so this does not prove the model works on new malware in the wild.

## Base rate
- Training data is 46% malicious. Real installs are almost all clean.
- At the balanced block threshold, 0.57% of clean packages are blocked in out-of-fold scoring. That is about 6 false blocks per 1,000 clean installs.
- Precision at runtime will be far lower than the precision in our tables. Warn is safer than block for a first release.

## Data limits
- Random benign packages were not checked by hand. They are flagged 2% to 3% of the time, against 0.3% to 0.7% for popular packages. Some may be bad, or they may just look plain.
- Popular benign packages are polished. A new, plain, honest package can look suspicious to the model.
- Packaging style carries signal. `meta_has_homepage` and `meta_n_deps` rank in the top four features. Dropping all metadata costs about 0.003 PR-AUC.
- Package size carries signal through the density features. Dropping densities costs 0.001 PR-AUC and 0.013 recall.
- Name features add nothing. Typosquats are caught by the edit-distance rule at runtime, not by the model.

## Coverage
- Wheel-only packages cannot be scored. 903 wheels failed to read. The CLI should warn on them.
- Names missing from PyPI never reach the model. A 404 is flagged as slopsquatting by a rule.
- Only `.py` files, `setup.py` and `PKG-INFO` are read. Payloads in other file types, in dependencies, or fetched after install are invisible.
- Static counts are easy to evade by splitting strings or encoding payloads.
- PyPI only. npm is not covered.

## Engineering
- The model file is a joblib pickle. Load only files built by this project.
- The training environment was Python 3.14.7. Exact versions are in `requirements.lock.txt`.
- Training ran before the code was committed, so the `git_commit` in `metrics.json` and in MLflow is `7491e3e`, the commit before the training code. The tag `v0.1-model` holds the code that ran.
- The final `.joblib` is tracked by DVC. Run `dvc pull` after cloning.

## What we did not do
- CodeBERT embeddings (raw archives no longer available).
- SHAP values. Permutation importance only.
- Testing on packages published after the datasets were collected.
