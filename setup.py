from pathlib import Path

from setuptools import find_namespace_packages, setup

__version__ = "0.15.0"
ROOT = Path(__file__).resolve().parent

required = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

setup(
    name="trustrag",
    version=__version__,
    author="gomate-community",
    packages=find_namespace_packages(include=["trustrag", "trustrag.*"]),
    package_data={
        "trustrag.modules.document": ["huqie.txt", "huqie.txt.trie"],
    },
    install_requires=required,
    author_email="yanqiang@ict.ac.cn",
    description="RAG Framework within Reliable input,Trusted output",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/gomate-community/TrustRAG",
    python_requires=">=3.11.0",
)
