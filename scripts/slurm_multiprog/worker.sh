#!/bin/bash
TASK_ID=$1
echo "Starting worker on $(hostname) after 2s sleep in ${PWD}"
sleep 2
mkdir -p logs
dask-worker --interface=ib0 --scheduler-file=${SCHEDULER_FILE} \
    --nthreads=1 --nprocs=1 --local-directory=/dev/shm/ \
    --death-timeout=60 > logs/worker_${TASK_ID}.log 2> logs/worker_${TASK_ID}.err
