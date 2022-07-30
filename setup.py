import os
import sys
from distutils.util import convert_path
from pathlib import Path


sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Load version
main_ = {}
ver_path = convert_path("liionpack/__init__.py")
with open(ver_path) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, main_)

# Load readme for description
with open("README.md", encoding="utf-8") as f:
    readme = f.read()


# https://stackoverflow.com/a/62724213/14746647
def current_branch():
    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()
    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


setup(
    name="liionpack",
    description="A battery pack simulator for PyBaMM",
    version=main_["__version__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    packages=["liionpack"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        # see https://github.com/pypa/setuptools/issues/2568
        "PyBaMM @ git+https://github.com/pybamm-team/PyBaMM@develop#egg=PyBaMM"
        if current_branch() == "develop"
        else "pybamm>=22.6",
        "pandas",
        "plotly",
        "openpyxl",
        "tqdm",
        "lcapy",
        "Ipython",
        "scikit-spatial",
        "networkx",
        "textwrapper",
        "ray",
        "redis",
    ],
    author="Tom Tranter",
    author_email="t.g.tranter@gmail.com",
    url="https://liionpack.readthedocs.io/en/latest/",
    project_urls={
        "Documentation": "https://liionpack.readthedocs.io/en/latest/",
        "Source": "https://github.com/pybamm-team/liionpack",
        "Tracker": "https://github.com/pybamm-team/liionpack/issues",
    },
)
