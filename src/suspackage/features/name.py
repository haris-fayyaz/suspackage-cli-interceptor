"""Name features. The package is never compared with itself."""
import re

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


def norm(s):
    return re.sub(r"[-_.]+", "-", str(s).strip().lower())


class NameFeatures:
    def __init__(self, popular):
        self.pop = list(dict.fromkeys(norm(p) for p in popular))

    def __call__(self, name):
        n = norm(name)
        hits = process.extract(n, self.pop, scorer=Levenshtein.distance, limit=2)
        near, dist = next(((c, d) for c, d, _ in hits if c != n), ("", len(n)))
        return {
            "name_len": len(n),
            "name_digits": sum(ch.isdigit() for ch in n),
            "name_hyphens": n.count("-"),
            "name_edit_dist": dist,
            "name_edit_ratio": round(dist / max(len(n), 1), 3),
            "is_typosquat": int(1 <= dist <= 2 and len(n) >= 4),
            "x_nearest_popular": near,
        }
