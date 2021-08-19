.. local2global_embedding documentation master file, created by
   sphinx-quickstart on Mon Aug 16 13:20:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
==================================================

.. toctree::
   :hidden:
   :maxdepth: 1

   self


Local2Global Embedding
------------------------

This package implements the embedding methods used in [#l2g]_. This package uses `pytorch <https://pytorch.org>`_ and `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`_ and it is a good idea to install these packages first following their respective installation instructions. If these packages are not already available during setup, an attempt is made to install them via ``pip`` which may not always work as expected. Afterwards use

.. code-block:: bash

   pip install git+https://github.com/LJeub/Local2Global_embedding@master

to install the package and other dependencies. The patch alignment algorithm used in [#l2g]_ is implemented in the separate `local2global <https://github.com/LJeub/Local2Global>` package. Installing this package will also install the latest version of ``local2global``.


Command-line interface
+++++++++++++++++++++++

This package exposes a command-line interface. Use

.. code-block:: bash

   python -m local2global_embedding.run --help

to see the available options. For example, to reproduce Figure 1(a) of [#l2g]_, use

.. code-block:: bash

   mkdir Cora
   cd Cora
   python -m local2global_embedding.run --dims=2,4,8,16,32,64,128 --plot

Note however that this will take a while to run.

For details of the python api see the :doc:`module reference <reference>`.

References
+++++++++++

.. [#l2g] L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu (2021).
          Local2Global: Scaling global representation learning on graphs via local training.
          DLG-KDD'21. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   reference


Index
-------

* :ref:`genindex`
