"""Smoke tests for the configuration file, scripts and version numbers."""
import py_compile
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_has_expected_keys_and_types():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert set(config) == {"root", "NFM_features_to_include", "pop_size"}
    assert isinstance(config["root"], str) and config["root"]
    available = {"Extra_Pond_Capacity_RP100_10", "WfW", "Losses", "ImpGrass"}
    assert set(config["NFM_features_to_include"]) <= available
    assert isinstance(config["pop_size"], int) and config["pop_size"] > 2


def test_scripts_compile():
    for name in ["rains_main.py", "rains_all_functions.py",
                 "SHETRAN_Post_Simulation_Functions.py", "version.py"]:
        py_compile.compile(str(ROOT / name), doraise=True)


def test_version_numbers_agree():
    namespace = {}
    exec((ROOT / "version.py").read_text(encoding="utf-8"), namespace)
    version = namespace["__version__"]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf'^version = "{re.escape(version)}"$', pyproject, re.M)
    assert re.search(rf'^version: "?{re.escape(version)}"?$', citation, re.M)
