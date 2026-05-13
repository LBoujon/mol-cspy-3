.. _threshold:

cspy-threshold command
======================

The ``cspy-threshold`` command is used to perform threshold monte carlo simulations.

.. sphinx_argparse_cli::
  :module: cspy.apps.threshold
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-threshold

Background
----------

The threshold algorithm is a Monte Carlo simulation with an energy
threshold, also called the energy lid. This threshold is set relative to
the energy of the initial configuration. Monte Carlo moves are then
accepted if and only if the energy of the resulting perturbed
configuration is below the energy threshold. Effectively, this
constrains the MC trajectory to explore only regions accessible below
the energy threshold. The threshold can be iteratively increased and
barriers between sampled energy minima and the initial configuration can
then be estimated as the energy of the threshold when the minima were
first sampled. These connections can be presented in a disconnectivity
graph, which clusters the minima sampled based on the energy barrier
from the initial configuration. By running trajectories from multiple
minima can estimate the energy barrier between them.

Note that the configurational space typically increases exponentially
with energy. Hence, converging high energy barriers requires extensive
sampling. From testing, can reasonably converge energy lids up to 20-30
kJ/mol above the initial structure.


Running Threshold Simulations
-----------------------------

For more information on running a threshold simulation, 
please have a look at the following worked example - :ref:`threshold-example`.

