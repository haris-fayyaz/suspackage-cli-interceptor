"""Fields read from the archive's own PKG-INFO."""


def _parse(text):
    head, _, body = text.partition("\n\n")
    fields, deps = {}, 0
    for line in head.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            if k in ("Requires-Dist", "Requires"):
                deps += 1
            else:
                fields.setdefault(k, v.strip())
    return fields, deps, body.strip()


def _has(v):
    return int(bool(v) and v.upper() != "UNKNOWN")


def features(text):
    if not text:
        return {"meta_present": 0, "meta_has_description": 0, "meta_has_homepage": 0,
                "meta_has_author": 0, "meta_n_deps": 0}
    f, deps, body = _parse(text)
    return {
        "meta_present": 1,
        "meta_has_description": int(_has(f.get("Summary")) or bool(body and body.upper() != "UNKNOWN")),
        "meta_has_homepage": int(_has(f.get("Home-page")) or _has(f.get("Project-URL"))),
        "meta_has_author": int(_has(f.get("Author")) or _has(f.get("Author-email"))),
        "meta_n_deps": deps,
    }
