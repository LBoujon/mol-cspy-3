#!/bin/bash
. "/home/prs1m18/miniconda3/etc/profile.d/conda.sh"
PATH=/home/prs1m18/miniconda3/bin:$PATH
conda init bash
conda activate cspy
TMPDIR=$(mktemp -d cspy_scheduler.XXX)
cd ${TMPDIR}
dask-scheduler
