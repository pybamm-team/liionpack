import os
import sys
from distutils.util import convert_path

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
        "pybamm",
        "pandas",
        "plotly",
        "openpyxl",
        "tqdm",
        "lcapy",
        "Ipython",
        "scikit-spatial",
        "networkx",
        "textwrapper",
        "dask[complete]",
        "ray",
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
