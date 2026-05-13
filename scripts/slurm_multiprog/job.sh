#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=mptest2
#SBATCH --partition=scavenger
. ~/.bashrc
conda activate cspy
export SCHEDULER_FILE=${SLURM_SUBMIT_DIR}/scheduler.json
export MOLS=acetic.xyz
export MULTS=acetic.dma
export CHARGES=acetic_rank0.dma
export AXIS=acetic.mols
export NUMBER_STRUCTURES=10
export SPACEGROUPS=coarse10
export CSPY_SCRIPTS_DIR=$HOME/cspy/scripts/slurm_multiprog
cp -r $CSPY_SCRIPTS_DIR job_steps
srun --multi-prog job_steps/csp_slurm.config
