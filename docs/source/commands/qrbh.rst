cspy-qrbh command
=================

The ``cspy-qrbh`` command can be used to perform quasi-random basin-hopping crystal structure prediction.

.. sphinx_argparse_cli::
  :module: cspy.apps.qr_bh
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-qrbh


The ``cspy-qrbh`` is very similar to the ``cspy-csp`` command but instead, performs crystal structure prediction 
in which quasi-random structure generation is used to seed multiple Monte Carlo basin hopping simulations as 
described in the linked `publication <https://doi.org/10.1021/acs.jctc.0c01101>`_.

For more information on how to use this command, plese see the worked exmaple - :ref:`qrbh`.
