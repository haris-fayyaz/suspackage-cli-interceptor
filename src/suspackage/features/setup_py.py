"""Install-time behavior: setup.py runs automatically on install."""
import re

CMD = re.compile(r"cmdclass|class\s+\w+\s*\(\s*(?:\w+\.)*(?:install|develop|egg_info|build_py|bdist_wheel)\s*\)")
SUBP = re.compile(r"subprocess|os\.system|os\.popen|\bPopen\b|os\.exec|os\.spawn")
NET = re.compile(r"urllib|urlopen|requests\.|http\.client|httpx|socket|wget|curl|urlretrieve|ftplib")
EXE = re.compile(r"\b(?:exec|eval)\s*\(")
B64 = re.compile(r"b64decode|base64|a2b_base64|b85decode")


def features(text):
    return {
        "sp_present": int(bool(text)),
        "sp_cmdclass": len(CMD.findall(text)),
        "sp_subprocess": len(SUBP.findall(text)),
        "sp_network": len(NET.findall(text)),
        "sp_exec_eval": len(EXE.findall(text)),
        "sp_base64": len(B64.findall(text)),
    }
