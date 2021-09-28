import os
import sys
from distutils.util import convert_path

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

main_ = {}
ver_path = convert_path('liionpack/__init__.py')
with open(ver_path) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, main_)

setup(
    name='liionpack',
    description='A battery pack simulator for PyBaMM',
    version=main_['__version__'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics'],
    packages=['liionpack'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'pybamm'],
    author='Tom Tranter',
    author_email='t.g.tranter@gmail.com',
    url='https://liionpack.readthedocs.io/en/latest/',
    project_urls={
        'Documentation': 'https://liionpack.readthedocs.io/en/latest/',
        'Source': 'https://github.com/pybamm-team/liionpack',
        'Tracker': 'https://github.com/pybamm-team/liionpack/issues',
    },
)
