=============================================================================
hdhp : Simulation and inference for the hierarchical Dirichlet-Hawkes process
=============================================================================

This is a Python implementation of the **hierarchical Dirichlet-Hawkes**
process, that includes both the generation and the inference algorithm.
For more details about this process and an application on
Stack Overflow data, please see the pre-print publication on arXiv_.

.. _arXiv: https://arxiv.org/abs/1610.05775

To cite this work, please use

::

   Mavroforakis, C., Valera, I. and Rodriguez, M.G., 2016.
   Modeling the Dynamics of Online Learning Activity.
   arXiv preprint arXiv:1610.05775.



Main Features
-------------

* Generative model for the hierarchical Dirichlet-Hawkes process

* Inference algorithm based on sequential Monte-Carlo

* Multi-threaded

* Arbitrary choice of vocabulary

* Plotting capabilities


Installation
------------

You can install the *hdhp* package by executing the following command in a terminal.

::

   pip install hdhp


Documentation
-------------

For instructions on how to use the package, consult `its documentation`__.

__ https://hdhp.readthedocs.org/

Examples
--------
You can find an example of how to use this package in the Jupyter notebooks
under the directory *examples*.



Note that the code is distributed under the Open Source Initiative (ISC) license.
For the exact terms of distribution, see the LICENSE_.

.. _LICENSE: ./LICENSE

::

   Copyright (c) 2016, hdhp contributors
   Charalampos Mavroforakis <cmav@bu.edu>,
   Isabel Valera <ivalera@mpi-sws.org>,
   Manuel Gomez-Rodriguez <manuelgr@mpi-sws.org>
