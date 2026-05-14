.. _config:

Advanced Configuration
======================

mol-CSPy uses `toml <https://github.com/toml-lang/toml>`__ as a
configuration language, which allows modification of many settings
utilized throughout the library.

By default, mol-CSPy will locate configuration changes in the following
order, where lower numbers/earlier in this list will override settings
from other sources:

1. Runtime changes (e.g. command line flags).
2. ``cspy.toml`` located in the current directory.
3. ``cspy/cspy_defaults.toml`` source file. This file is initially blank upon installation but can be altered if you wish.
4. The ``DEFAULTS`` variable in the ``cspy/configuration.py`` source file.

If you're changing settings and they don't seem to be having an effect,
check that one of the higher priority sources doesn't override that
variable.

Modifying Minimization Settings
-------------------------------

When using the ``cspy-csp`` app (see :ref:`csp-app`), settings can be
modified in order to change the minimization steps in addition to the
available command line flags.

The following is an example which sets some of the current defaults
steps involved in minimization:

.. code:: toml

   dmacrys.timeout = 1200.0

   [pmin]
   timeout = 1200.0 

   [[csp_minimization_step]]
   kind = "pmin"
   electrostatics = "charges"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "charges"
   CONP = true
   PRES = "0.1 GPa"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "multipoles"

Symmetry Reduction Stage
------------------------

You can add a symmetry reduction as a final stage of your ``cspy-csp`` search
using specific settings in the toml for example here is a three stage
process with a symmetry reduction to a P1 supercell that has Z'=16 in
the final optimisation. When using this option CSP will drop crystal
structures to P1 and then expand to a supercell with Z'>=16, ensure that
you choose to sample space groups with the number of molecules in the
unit cell that is a factor of the Z' target if you want to sample only
P1 Z'=16. If you just want to drop all structures to P1 with any Z' you
can set reduction = 0.

.. code:: toml

   dmacrys.timeout = 600.0

   [pmin]
   timeout = 600.0 

   [[csp_minimization_step]]
   kind = "pmin"
   electrostatics = "charges"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "charges"
   CONP = true
   PRES = "0.1 GPa"

   [[csp_minimization_step]]
   kind = "dmacrys"
   electrostatics = "multipoles"
   reduction = 16
   SEIG = "1"
   NOPR = false
