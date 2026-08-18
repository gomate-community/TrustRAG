"""Regression tests for the Python distribution configuration."""

from pathlib import Path
import runpy

import setuptools


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _setup_kwargs(monkeypatch):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(PROJECT_ROOT / "setup.py"), run_name="__main__")
    return captured


def test_distribution_includes_every_python_package(monkeypatch):
    """Every source module must have its containing package in the wheel."""
    setup_kwargs = _setup_kwargs(monkeypatch)
    distributed_packages = set(setup_kwargs["packages"])
    source_packages = {
        ".".join(path.relative_to(PROJECT_ROOT).parent.parts)
        for path in (PROJECT_ROOT / "trustrag").rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert source_packages <= distributed_packages


def test_distribution_includes_tokenizer_resources(monkeypatch):
    """The document tokenizer's runtime data files must remain packaged."""
    setup_kwargs = _setup_kwargs(monkeypatch)

    assert set(setup_kwargs["package_data"]["trustrag.modules.document"]) == {
        "huqie.txt",
        "huqie.txt.trie",
    }
