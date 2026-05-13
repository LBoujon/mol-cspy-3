#!/bin/bash
#SBATCH --job-name=pyrene-csp-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=NONE
#SBATCH --mem=60GB
#SBATCH --time=7:59:00

#SBATCH --workdir=/scratch/prs1m18/pyrene
. ~/.bashrc
conda activate cspy
dask-scheduler --interface ib0 --scheduler-file scheduler.json 2> 'scheduler.stderr' > 'scheduler.stdout' &
dask-worker --nprocs=4 --interface ib0 --nthreads=10 --scheduler-file='scheduler.json' 2> 'worker.stderr' > 'worker.stdout' --local-directory=/dev/shm &
csp pyrene.xyz -n 10000 -a pyrene.mols -m pyrene_mp.smult -c pyrene_rank0.smult
