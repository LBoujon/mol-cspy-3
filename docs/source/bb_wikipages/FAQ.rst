Common Errors
=============

No module named 'ccdc'
^^^^^^^^^^^^^^^^^^^^^^

This error is analogous to ``ccdc module does not exist - compack method can not be used.``

This is a Python ImportError which typically appears when the user tries to perform clustering in 
the ``cspy-db`` command using COMPACK. While CSPy does have the capabilities to use the CSD Python
API, this requires a Python environment which integrates the dependencies for CSPy and the CSD Python
API. Please see :ref:`csd-python-api-installation` for more information.


Error parsing dmacrys_summary could not convert string to float: '************'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error occurs because atoms get too close together and are able to overcome the buckingham repulsive wall. The buckingham potential becomes
attractive as the interatomic distance becomes very small which results in atoms becoming strongly bound together. This is unphysical and DMACRYS
no longer returns an energy from the geometry optimisation.

In a cspy-csp simulation, the optimisation is recorded as a failure and the siumulation moves on to the next job.

.. note::
  In a future release, we plan to tidy up error handling


Command XXXXX timed out after XXXX.XXXX seconds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error usually occurs when DMACRYS or PMIN fail to optimise a crystal structure within the specified timeout parameter. This can be 
adjusted in the TOML configuration file as described in :ref:`config`.


I am providing an axis file to Neighcrys, but it can’t find it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your axis file's name may be too long – Neighcrys cannot read filenames beyond a certain length. Try shortening the filename.


A hydrogen cannot be connected to more than one atom. Exiting.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contrary to the error message, this often happens when a hydrogen has
//no// bonds, i.e. at least one of your X-H distances in your cutoff
file is too short. Visualise your structure and measure any unusual X-H
distances, ensuring your cutoff file has values large enough to
accommodate them.

If the structure seems fine the MAXSCH value might be too small. Setting
the MAXSCH value to 7 or greater seems to often fix this error.


Too many potentials in input file Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This happens when the potential file you are supplying to NEIGHCRYS has
too many entries for types of interactions (the current limit is 200).
One way around this is to simply remove interactions not present in your
system. A check has been added to the energy_minimise function in
crystal.py. This will obtain the neighcrys labels from the fort.21, read
the potential file and write only the terms needed for that particular
system to a new file “r.pots”. This should remove the issue, unless the
system involves more that 200 unique interactions in which case the
neighcrys source code will need to be modified.


NEIGHCRYS (or DMACRYS) fails with halogen atoms in one of the molecules of a multi component system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NEIGHCRYS (or DMACRYS) requires molecules with halogen atoms to be last
in the res file. If halogen atoms are present before any non-halogenated
molecules neighcrys/dmacrys will exhibit non-standard behaviour and will
most likely fail.
