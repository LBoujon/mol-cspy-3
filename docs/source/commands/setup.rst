cspy-setup command
------------------

The ``cspy-setup`` command is a useful tool for performing calculation setups.

.. sphinx_argparse_cli::
  :module: cspy.apps.setup
  :func: main
  :hook:
  :title:
  :description:
  :epilog:
  :group_title_prefix:
  :prog: cspy-setup

When you have a large number of conformers/molecules, writing
optimization and DMA inputs is likely to be error prone and very
tedious. As such, there is a convenience script for generating these
inputs.

The naming of the conformer files is very important for this script, as
it relies on the ``_1.xyz`` or ``_2.xyz`` etc. suffix for each file to
identify the conformers. So, even if you have only one conformer, choose
the name of the file like conformer_1.xyz.

Geometry optimisations
^^^^^^^^^^^^^^^^^^^^^^

For a typical set of geometry optimizations, running the script would
look something like this:

With the contents of the current directory being two conformations,
numbered as follows:

.. code:: bash

   acetic_1.xyz
   acetic_2.xyz

We can run the optimization setup script:

.. code:: bash

   cspy-setup -j opt --walltime=2:00:00 --procs=4 --memory=10GB -- *.xyz

Which will result in the directory containing:

.. code:: bash

   acetic_1.com
   acetic_1.xyz
   acetic_2.com
   acetic_2.xyz
   acetic.opt.sh

The contents of the ``acetic.opt.sh`` file will be:

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=acetic.opt
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --mail-type=NONE
   #SBATCH --array=1-2
   #SBATCH --time=2:00:00
   #SBATCH --mem=10GB
   #SBATCH --output="opt-acetic_%a-%A.out"

   export GAUSS_MDEF=8GB
   export GAUSS_PDEF=4
   DIR=/mainfs/scratch/prs1m18/acetic/
   com_file=TSPMCl_${SLURM_ARRAY_TASK_ID}.com
   NAME=${com_file%.com}
   echo "Geometry optimisation for $NAME using ${GAUSS_MDEF} memory and ${GAUSS_PDEF} processors"
   workdir="${DIR}/${NAME}"
   mkdir -p ${workdir}
   cp  ${com_file} ${workdir}
   cd ${workdir}
   g09 ${com_file}
   pexit=$?
   echo "exiting with status ${pexit}"
   exit $pexit

And submission of this job will create a job array, where each array id
corresponds to a conformer optimization.

DMA
^^^

Much like the case of geometry optimizations, running the script would
look something like this:

.. code:: bash

   cspy-setup -j dma --walltime=2:00:00 --procs=4 --memory=10GB -- *.xyz

This will create directories for each conformation, so the listing would
look something like this:

.. code:: bash

   acetic.dma.sh

   acetic_1:
   acetic_1.xyz

   acetic_2:
   acetic_2.xyz

And the contents of ``acetic.dma.sh`` will be:

.. code:: bash

   #!/bin/bash
   #SBATCH --job-name=acetic.dma
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --mail-type=NONE
   #SBATCH --array=1-2
   #SBATCH --time=2:00:00
   #SBATCH --workdir=/mainfs/scratch/prs1m18/TSPMCl/2.dma
   #SBATCH --mem=10GB
   #SBATCH --output="dma-acetic_%a-%A.out"

   conda activate cspy
   DIR=/mainfs/scratch/prs1m18/TSPMCl/2.dma
   cd ${DIR}/TSPMCl_${SLURM_ARRAY_TASK_ID}
   cspy-dma TSPMCl_${SLURM_ARRAY_TASK_ID}.xyz -p F -j 4 -m 8GB
   pexit=$?
   echo "exiting with status ${pexit}"
   exit $pexit

Once again, the submission of this job will create a job array, where
each array id corresponds to a conformer for DMA. **Note**: It’s not
recommended to use many processors for DMA jobs, as GDMA is not
parallellized, and will often be the bottleneck especially if you
utilise many cores for the single point energy calculation.
