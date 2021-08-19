Local2Global Embedding
------------------------

This package implements the embedding methods used in [#l2g]_. This package uses `pytorch <https://pytorch.org>`_ and `pytorch-geometric <https://github.com/rusty1s/pytorch_geometric>`_ and it is a good idea to install these packages first following their respective installation instructions. If these packages are not already available during setup, an attempt is made to install them via ``pip`` which may not always work as expected. Afterwards use

.. code-block:: bash

   pip install git+https://github.com/LJeub/Local2Global_embedding@master

to install the package and other dependencies. The patch alignment algorithm used in [#l2g]_ is implemented in the separate `local2global <https://github.com/LJeub/Local2Global>`_ package. Installing this package will also install the latest version of ``local2global``.

For more information see the `Documentation <https://ljeub.github.io/Local2Global_embedding/>`_.

References
+++++++++++

.. [#l2g] L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu (2021).
          Local2Global: Scaling global representation learning on graphs via local training.
          DLG-KDD'21. `arXiv:2107.12224 [cs.LG] <https://arxiv.org/abs/2107.12224>`_
