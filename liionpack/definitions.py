#
# Definitions
#
import liionpack as lp
import os
import pathlib


ROOT_DIR = str(pathlib.Path(lp.__path__[0]).parent)
MODULE_DIR = os.path.dirname(os.path.abspath(lp.__file__))
CIRCUIT_DIR = os.path.join(MODULE_DIR, "circuits")
