"""Whole-package code features (Python ast, with a regex fallback for files that fail to parse)."""
import ast
import math
import re
import warnings
from collections import Counter

SENS = re.compile(r"\.ssh|\.aws|\.kube|\.docker|id_rsa|\.npmrc|\.pypirc|/etc/passwd|/etc/shadow|Login Data|Local State|Cookies|wallet|keychain|\.bash_history|\.git-credentials", re.I)
HOOK = re.compile(r"discord(?:app)?\.com/api/webhooks|api\.telegram\.org|hooks\.slack\.com|pastebin\.com|ngrok|webhook\.site|requestbin|interact\.sh|oast\.|transfer\.sh|anonfiles|ghostbin", re.I)
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
BLOB = re.compile(r"[\"']([A-Za-z0-9+/=_\-]{40,})[\"']")
HEX = re.compile(r"\\x[0-9a-fA-F]{2}")
DYN = re.compile(r"__import__|getattr\(|globals\(\)|bytes\.fromhex|\[::-1\]|chr\(")
ENVIRON = re.compile(r"os\.environ|os\.getenv|\bgetenv\(")
CALL_RE = re.compile(r"([A-Za-z_][\w\.]*)\s*\(")
LOCAL_IP = ("127.", "0.", "255.")


def _dotted(n):
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


def call_names(text):
    """Return (list of called names, parse_failed)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
        return [_dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)], 0
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return CALL_RE.findall(text), 1


def _entropy(s):
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def features(py):
    c = Counter()
    fails = 0
    parts = []
    for text in py.values():
        names, failed = call_names(text)
        fails += failed
        for n in names:
            c["exec_eval"] += n in ("exec", "eval")
            c["subprocess"] += n.startswith(("subprocess.", "os.system", "os.popen", "os.exec", "os.spawn")) or n == "Popen"
            c["socket"] += n.startswith("socket.")
            c["http"] += n.startswith(("requests.", "urllib", "http.client", "httpx.", "aiohttp.")) or n.endswith("urlopen")
            c["b64"] += n.endswith(("b64decode", "decodebytes", "a2b_base64", "b85decode", "b32decode"))
            c["decomp"] += n.endswith(".decompress") or n in ("marshal.loads", "codecs.decode")
            c["ctypes"] += n.startswith("ctypes.")
        parts.append(text)
    all_text = "\n".join(parts)
    n_files = len(py)

    blobs = BLOB.findall(all_text)[:200]
    max_ent = max((_entropy(b) for b in blobs), default=0.0)
    longest = max((len(b) for b in blobs), default=0)

    lines = all_text.split("\n")
    long_ratio = sum(len(l) > 300 for l in lines) / max(len(lines), 1)
    hex_density = len(HEX.findall(all_text)) / max(len(all_text) / 1000, 1)
    dyn = len(DYN.findall(all_text))
    obf = min(1.0, 2 * long_ratio + 0.4 * min(hex_density / 50, 1) + 0.3 * min(dyn / 10, 1))

    return {
        "c_exec_eval": c["exec_eval"],
        "c_subprocess": c["subprocess"],
        "c_socket": c["socket"],
        "c_http": c["http"],
        "c_b64_decode": c["b64"],
        "c_decompress": c["decomp"],
        "c_ctypes": c["ctypes"] + len(re.findall(r"import ctypes", all_text)),
        "c_environ": len(ENVIRON.findall(all_text)),
        "c_sensitive_paths": len(SENS.findall(all_text)),
        "c_webhook_urls": len(HOOK.findall(all_text)),
        "c_ip_literals": sum(not ip.startswith(LOCAL_IP) for ip in IPV4.findall(all_text)),
        "max_blob_entropy": round(max_ent, 3),
        "longest_blob_len": longest,
        "obfuscation_score": round(obf, 3),
        "parse_fail_ratio": round(fails / n_files, 3) if n_files else 0.0,
        "x_py_files": n_files,
        "x_py_kb": round(len(all_text) / 1024, 1),
    }
