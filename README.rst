
.. image:: https://badge.fury.io/py/pytrax.svg
   :target: https://pypi.python.org/pypi/pytrax

.. image:: https://travis-ci.org/PMEAL/pytrax.svg?branch=master
   :target: https://travis-ci.org/PMEAL/pytrax

.. image:: https://codecov.io/gh/PMEAL/pytrax/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/PMEAL/pytrax

.. image:: https://readthedocs.org/projects/pytrax/badge/?version=latest
   :target: http://pytrax.readthedocs.org/

###############################################################################
Overview of liionpack
###############################################################################

*liionpack* takes a 1D PyBaMM model and makes it into a pack. You can either specify
the configuration e.g. 16 cells in parallel and 2 in series (16p2s) or load a
netlist

===============================================================================
Example Usage
===============================================================================

The following code block illustrates how to use liionpack to perform a simulation:

.. code-block:: python

  >>> import liionpack as lp
  >>> protocol = lp.generate_protocol()
  >>> output = lp.solve(protocol=protocol)
