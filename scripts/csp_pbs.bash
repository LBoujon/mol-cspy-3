# This should start the scheduler and csp script on task 0/1

cd ${PBS_O_WORKDIR}
source ~/.bashrc
conda activate cspy

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ${PBS_O_WORKDIR}/csp_variables.sh
LOGS=${PBS_O_WORKDIR}/logs

mkdir -p ${LOGS}
echo "Starting job with ${PBS_VNODENUM}"
if [ ${PBS_VNODENUM} == 0 ]; then
    # Run the CSP
    echo "Starting CSP script task_id=${PBS_VNODENUM} on $(hostname) after 5s sleep in ${PWD}"
    sleep 5
    python -m cspy.apps.csp ${MOLS} -c ${CHARGES} -m ${MULTS} -a ${AXIS} \
        -g "${SPACEGROUPS}" -n ${NUMBER_STRUCTURES} --scheduler=file \
        --scheduler-file=${SCHEDULER_FILE} 2> ${LOGS}/csp.err > ${LOGS}/csp.log
elif [ ${PBS_VNODENUM} == 1 ]; then
    # Run the scheduler
    echo "Starting scheduler task_id=${PBS_VNODENUM} on $(hostname) in ${PWD}"
    python -m distributed.cli.dask_scheduler --scheduler-file=${SCHEDULER_FILE} \
        --interface=ib0 2> ${LOGS}/scheduler.err > ${LOGS}/scheduler.log
else
    # All other cores run workers
    echo "Starting worker task_id=${PBS_VNODENUM} on $(hostname) after 10s sleep in ${PWD}"
    sleep 10
    python -m distributed.cli.dask_worker --interface=ib0 --scheduler-file=${SCHEDULER_FILE} \
        --nthreads=1 --nprocs=1 --memory-limit=${MEMORY_LIMIT} --local-directory=/dev/shm/ \
        --death-timeout=60 > ${LOGS}/worker_${PBS_VNODENUM}.log 2> logs/worker_${PBS_VNODENUM}.err
fi
