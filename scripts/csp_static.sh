#!/bin/bash
#SBATCH --job-name NAME
#SBATCH --nodes=NODES
#SBATCH --array=CONFORMER_NUMBRES
#SBATCH --ntasks-per-node=40
#SBATCH --time=TIME
#SBATCH --workdir=WORKDIR
#SBATCH --mem=64GB
#SBATCH --output="CSP_NAME_%A_%a.out"

. ~/.bashrc
conda activate cspy
MOL_NAME=PREFIX_${SLURM_ARRAY_TASK_ID}
SCHED_CSP_SCRIPT=$(realpath scheduler_csp.sh)
cd ${MOL_NAME}
LOG_DIRECTORY=logs
SCHED=$(realpath scheduler.json)
((NUM_WORKER_TASKS=SLURM_NTASKS - 2))
MOLS=${MOL_NAME}.xyz
MULTS=${MOL_NAME}.dma
CHARGES=${MOL_NAME}_rank0.dma
AXIS=${MOL_NAME}.mols
HEAD_NODE=${SLURM_JOB_NODELIST} # get the head node
NUM_NODES=${SLURM_JOB_NUM_NODES}
((HEAD_NODE_TASKS=SLURM_NTASKS_PER_NODE - 2))

echo "CSP script for molecule ${MOL_NAME}"
echo "${NUM_NODES} nodes, total number of dask-worker instances: ${NUM_WORKER_TASKS}"
echo "Head node: ${HEAD_NODE}, tasks: ${HEAD_NODE_TASKS}"

WORKER_CMD="dask-worker --interface=ib0 --scheduler-file=${SCHED} --nthreads=1 --nprocs=1 --local-directory=/dev/shm --death-timeout=60"
export SCHEDULER_CMD="dask-scheduler --scheduler-file=${SCHED} --interface=ib0"
export CSP_CMD="csp ${MOLS} -c ${CHARGES} -m ${MULTS} -a ${AXIS} -g coarse10 -n 100 -p w99rev_6311_s --scheduler=file --scheduler-file=${SCHED}"

echo "Creating directory for logs: ${LOG_DIRECTORY}"
mkdir -p ${LOG_DIRECTORY}

echo "Starting workers in 10s: "
echo "NOTE: workers allocated using plane=2 so scheduler + csp are on the same node"
# the & at the end of this line is very important, don't remove it
srun --exclusive --output="logs/worker_%t.log" --begin=now+10 -m plane=2 --ntasks=${NUM_WORKER_TASKS} ${WORKER_CMD} &

rm -f ${SCHED}
echo "Starting scheduler process:\n${SCHEDULER_CMD}"
echo "Scheduler file location: ${SCHED}"
echo "Starting csp process:\n${CSP_CMD}"
echo "Scheduler+csp script: ${SCHED_CSP_SCRIPT}"
srun --exclusive --cpus-per-task=2 --ntasks=1 --nodes=1 "${SCHED_CSP_SCRIPT}"
