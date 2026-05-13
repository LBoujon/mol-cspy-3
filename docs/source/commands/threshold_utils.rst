.. _threshold_utils:

cspy-threshold-utils command
============================

The ``cspy-threshold-utils`` provides three subcommands to interact with 
threshold databases, generate disconnectivity graphs and prepare the data 
to run threshold jobs with the :ref:`threshold`.

.. sphinx_argparse_cli::
  :module: cspy.apps.threshold_utils
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-threshold-utils

Preparing the data
------------------

.. sphinx_argparse_cli::
  :module: cspy.threshold.threshold_setup
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-threshold-utils setup

In order to run threshold jobs, the starting crystals must be properly setup. This
app will accept crystal files or CSPy databases and create a new file for 
each crystal in the P1 space group and in the SHELXL file format. The appeal of this
app is that it generates the output crystals ensuring they all have the same number 
of molecules in the cell. This is done by creating supercells where necessary. If all
the input crystals can't be made to have the same number of molecules in the cell, the
app will refuse to run.

The reason to generate the crystals with the same number of molecules in the cell is
so that, when the threshold databases are merged and clustered, matches can be potentially
found between different trajectories. If your set of initial crystals can't all have the same
number of molecules in the cell, you should make sure that the different cells have 
number of molecules multiples of each other, for example: 4, 8, 16. 

Interacting with databases
--------------------------

.. sphinx_argparse_cli::
  :module: cspy.threshold.threshold_db
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-threshold-utils db

This utility provides the following actions on threshold databases:

* `--minimize` minimizes the valid points along the trajectory. There are flags that
  can be set in order to skip holding points between minimizations or to minimize
  structures from a specific trial of the database.
* `--combine` combines multiple threshold databases into a single database. This is
  useful when the trajectories for different starting points have been run in 
  individual jobs. The databases must be merged in the end so that they can be 
  clustered to find connections between the trajectories, and so that the disconnectivity
  graph can be generated.
* `--dump` saves the valid minimization structures of a threshold database to a zip file
  in CIF or SHELXL file.
* `--cluster` allows running further clustering using the COMPACK algorithm on
  threshold databases.
* `--split` allows splitting a single database into one database for each trial.

Generating disconnectivity graphs
---------------------------------

.. sphinx_argparse_cli::
  :module: cspy.threshold.disconnectivity_graph
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-threshold-utils disconn

Once the all the trajectories have been run and minimized, disconnectivity graphs
can be generated. To do so pass the clustered trajectory database to this utility:

.. code:: bash

    cspy-threshold-utils disconn database-1.db

This will generate a file with the default name `disconnectivity_graph.png`. Many
options are provided to change what will be plotted and how:

* `--plot-branch` will only plot the disconnectivity of the structures from which 
  trajectories were started. It will remove any structures that were found 
  during the course of the trajectories.

* `--plot-nodes` adds the node at the end of the branches of the graph. 
  Nodes for trajectory starts are dhown differently than those of structures 
  found during the trajectories.

Generation of the disconnectivity graph can be a costly operation for very large
databases. In order to avoid having to recalculate the graph every time, this utility
provides the `--save-pickle` option:

.. code:: bash

    cspy-threshold-utils disconn --save-pickle database-1.db

This generates a pickle file that can be used to replot the disconnectivity graph:

.. code:: bash

    cspy-threshold-utils disconn disconnectivity_graph.pickle

This utility is limited in terms of the types of modifying the looks of the disconnectivity
graphs. In order to modify the looks of the plot or how the disconnectivity graph
is generated, the user can write their own scripts. To do so, use the 
`cspy.threshold.disconnectivity_graph.DisconnectivityGraph` class.  

.. code-block:: python
    :linenos:

    from cspy.db import CspDataStore
    from cspy.threshold.disconnectivity_graph import DisconnectivityGraph, DrawFilters

    ds = CspDataStore("database-1.db")
    disconn = DisconnectivityGraph(ds=ds)

    # Obtain data from the disconnectivity graph. Check the available methods
    # that DisconnectivityGraph provides.
    lid_id = disconn.barrier_between_ids("STRUCT_ID1", "STRUCT_ID2")

    # DrawFilters provides some default utilities to change the looks of the 
    # disconnectivity graph. The user can also write their own filters.
    disconn.draw("my_disconn_graph", DrawFilters.initial_structures())

    # DisconnectivityGraph can also provide the figure and axes of the graph
    fig, ax = disconn.create_plot()
