#!/bin/bash
#SBATCH --mincpus=40
#SBATCH --nodes=2-2
#SBATCH --ntasks=9
#SBATCH --time=24:00:00

cd $SLURM_SUBMIT_DIR

export GAUSS_SCRDIR=/scratch/$USER
export g09root=/local/software/gaussian
source $g09root/g09/bsd/g09.profile

module load conda/py3-latest
source activate molbuilder

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpiexec --map-by socket cspy-mol molbuilder.ini
