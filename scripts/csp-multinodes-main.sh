#!/bin/bash
#SBATCH --job-name=testmul
#SBATCH --array=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem=60GB
#SBATCH --time=00:30:00
##SBATCH --mail-type=NONE

#SBTACH --workdir=/scratch/sy1u17/
node_list=(`scontrol show hostnames ${SLURM_JOB_NODELIST}`)
main_node=${node_list[0]}
DIR=`pwd`

crystal=tcyety
sg=${SLURM_ARRAY_TASK_ID}

for node in ${node_list[@]:1}
do
    ssh ${node} "cd ${DIR}; bash job_work.sh ${crystal} ${sg} ${node}"
done
cd ${DIR}
bash job_sche.sh ${crystal} ${sg}
