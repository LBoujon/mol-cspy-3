.. _efficient_optimisations:

Efficient Geometry Optimisations
================================

mol-CSPy offers a couple of opt-in features to make geometry optimisations with DMACRYS more efficient.
These features should not be used in combination with other optimisers like pmin, GULP, or ASE.


Automatic Timeouts
-------------------------------

Timeouts exist in mol-CSPy to end a geometry optimisations if it has been running too long. 
This is necessary because GOs can sometimes get "stuck" and take far longer than usual to converge; hogging compute cores and slowing down the overal run.
The default timeout for DMACRYS is 1200 seconds: in most cases, this is very excessive and can lead to stuck processes running longer than needed.
The optimal timeout depends on the space group, molecules, and hardware. We have therefore implemented automatic timeouts to reduce the burden on the user of optimising timeouts.

When ``timeout_auto`` is set to ``True`` in the ``cspy.toml`` file, mol-CSPy will perform an initial sample of geometry optimisations with a long timeout (1800 seconds by default but can be overidden with ``timeout_auto_cap``).
At the end of the sample, it will calculate what the optimal timeout would have been for that sample, and then use that timeout going forward.
The automatic timeout will be written to the ``cspy.toml`` file so that it can be used after restarts.
The sample size is set to 1000 by default but can be overrided with ``timeout_auto_sample``.

.. note::
   The value of the automatic timeout is a helpful guide for how long geometry optimisations take but two automatic timeout values should not be compares as a quantitative measure of the speed difference between two space groups, molecules, or machines.
   It may be used qualitatively at the users own judgement.


Optimisation Schemes
-------------------------------

Optimisation schemes have been implemented to make multistep geometry optimisations more efficient when using DMACRYS.
The user may specific DMACRYS settings directly in ``cspy.toml`` files, but it is not recommended unless the user has a good understanding of DMACRYS.
Optimisation schemes are preset "bundles" of settings that have been configured to allow the user to balance speed against precision in a coarse-grained fashion.
The four settings that are controlled are:

- ``LIMI`` : Maximum molecular displacement allowed to satisfy convergence criterion
- ``LIMG`` : Maximum predicted change in energy after molecular displacement to satisfy convergence criterion
- ``MAXD`` : Maximum allowed step size
- ``UDTE`` : Controls how often the Hessian is updated. Larger values = more frequent.


Six schemes are offered at the time of writing but a full list can be found in ``cspy/configuration.py``:

- ``fast&loosest`` : Recommend only for development work
- ``fast&very_loose`` : Recommend only for development work
- ``fast&loose`` : Useful for an early optimisation with a cheaper energy model
- ``loose`` : Use this scheme in place of above if the frequency of Buckingham catastrophes is a problem.
- ``precise`` : Good for a final optimisation step
- ``very_precise`` : Useful if you need very well converged geometries. Energies will not change much between this and above.


Optimisation schemes can be configured in ``cspy.toml`` as ``opt_scheme = "fast&loose"``.

If not specified, mol-CSPy will use ``very_precise``.
For a two step optimisation with point charges and then multipoles, I recommennd ``fast&loose`` and then ``precise`` e.g.:

.. code:: toml

    [[csp_minimization_step]]
    kind = "dmacrys"
    electrostatics = "charges"
    PRES = "0.1 GPa"
    opt_scheme = "fast&loose"

    [[csp_minimization_step]]
    kind = "dmacrys"
    electrostatics = "multipoles"
    opt_scheme = "precise"

    [dmacrys]
    timeout_auto = true
    timeout_auto_cap = 600