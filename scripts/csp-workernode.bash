#!/bin/bash
#SBATCH --job-name=worker-pyrene-csp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=NONE
#SBATCH --mem=60GB
#SBATCH --time=0:59:00

#SBATCH --workdir=/scratch/prs1m18/pyrene
. ~/.bashrc
conda activate cspy
dask-worker --interface ib0 --nprocs=10 --nthreads=4 --scheduler-file='scheduler.json' 2> "worker-${SLURM_JOB_NODELIST}.stderr" > "worker-${SLURM_JOB_NODELIST}.stdout" --local-directory=/dev/shm
