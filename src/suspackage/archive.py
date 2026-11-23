"""Read a package archive fully in memory.

Nothing is extracted to disk and no package code is ever executed.
"""
import tarfile
import zipfile
from dataclasses import dataclass, field

SKIP_DIRS = {"test", "tests", "docs", "examples", "example"}


@dataclass
class Package:
    py: dict = field(default_factory=dict)  # path -> source text
    setup_py: str = ""
    pkg_info: str = ""
    n_entries: int = 0
    is_wheel: bool = False  # wheels have no setup.py or PKG-INFO, so they are skipped


def _depth(p):
    return p.strip("/").count("/")


def _text(b):
    return b.decode("utf-8", "ignore")


def _is_zip(path):
    """Detect the format from the first bytes, not the file extension."""
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


def _iter_zip(path, pwd, cap):
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.is_dir():
                yield info.filename, info.file_size, (lambda i=info: zf.open(i, pwd=pwd).read(cap + 1))


def _iter_tar(path, cap):
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if m.isfile():
                yield m.name, m.size, (lambda m=m: tf.extractfile(m).read(cap + 1))


def read_package(path, cfg):
    cap = cfg["max_py_bytes"]
    if _is_zip(path):
        items = _iter_zip(path, cfg["zip_password"].encode(), cap)
    else:
        items = _iter_tar(path, cap)
    pkg, total, info_path, sp_path = Package(), 0, None, None
    for name, size, read in items:
        pkg.n_entries += 1
        if pkg.n_entries > cfg["max_files"]:
            break
        base = name.rsplit("/", 1)[-1]
        if ".dist-info/" in name:
            pkg.is_wheel = True
        if base == "PKG-INFO" and size <= 200_000:
            if info_path is None or _depth(name) < _depth(info_path):
                pkg.pkg_info, info_path = _text(read()), name
        elif name.endswith(".py") and size <= cap and total + size <= cfg["max_total_py_bytes"]:
            dirs = set(name.lower().split("/")[:-1])
            if dirs & SKIP_DIRS:  # tests, docs and examples are not install-time code
                continue
            text = _text(read())
            pkg.py[name] = text
            total += len(text)
            if base == "setup.py":
                if sp_path is None or _depth(name) < _depth(sp_path):
                    pkg.setup_py, sp_path = text, name
    return pkg
