.. silmaril documentation master file, created by
   sphinx-quickstart on Thu Nov  2 14:50:21 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

`silmaril <https://github.com/syp2001/silmaril>`_ is a python package for simulating gravitationally lensed observations of high-redshift galaxies.


Installation
------------
Clone the repository from github

.. code-block:: bash

   git clone https://github.com/syp2001/silmaril.git

Install using pip from within the cloned repo

.. code-block:: bash

   cd silmaril


.. code-block:: bash

   pip install .

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/sunrise

.. _API Reference:

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :caption: API Reference
   :template: custom-module-template.rst
   :recursive:

   ~silmaril.galaxy.Galaxy
   ~silmaril.lens.Lens
   ~silmaril.imaging.Detector
   ~silmaril.imaging.Observation
   ~silmaril.galaxy
   ~silmaril.imaging
   ~silmaril.lens
   ~silmaril.utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
