#!/bin/bash
#PBS -l nodes=2:ppn=16
#PBS -l walltime=45:00:00

CSPY_DIR=${HOME}/cspy

cd ${PBS_O_WORKDIR}

. ~/.bashrc
conda activate cspy

# CSP-specific variables
export MOLS=acetic.xyz
export MULTS=acetic.dma
export CHARGES=acetic_rank0.dma
export AXIS=acetic.mols
export NUMBER_STRUCTURES=10
export SPACEGROUPS=coarse10
export MEMORY_LIMIT=3750MB
export SCHEDULER_FILE=${PBS_O_WORKDIR}/scheduler.json
export CSPY_SCRIPTS_DIR=${CSPY_DIR}/scripts/pbs_multiprog

rm -f ${PBS_O_WORKDIR}/csp_variables.sh
echo "export MOLS=${MOLS}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export MULTS=${MULTS}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export CHARGES=${CHARGES}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export AXIS=${AXIS}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export NUMBER_STRUCTURES=${NUMBER_STRUCTURES}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export SPACEGROUPS=${SPACEGROUPS}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export MEMORY_LIMIT=${MEMORY_LIMIT}" >> ${PBS_O_WORKDIR}/csp_variables.sh
echo "export SCHEDULER_FILE=${SCHEDULER_FILE}" >> ${PBS_O_WORKDIR}/csp_variables.sh

cp ${CSPY_SCRIPTS_DIR}/csp_pbs.bash ${PBS_O_WORKDIR}

pbsdsh -v bash -l -c ${PBS_O_WORKDIR}/csp_pbs.bash
