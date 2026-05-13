#!/bin/bash
echo "Starting scheduler on $(hostname) in ${PWD}"
mkdir -p logs
dask-scheduler --scheduler-file=${SCHEDULER_FILE} \
    --interface=ib0 2> logs/scheduler.err > logs/scheduler.log
