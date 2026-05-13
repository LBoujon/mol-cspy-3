crystal=$1
sg=$2
node=$3
. ~/.bashrc
source activate cspy
dask-worker --interface ib0 --nprocs=40 --death-timeout=60 --scheduler-file=scheduler_${sg}.json 2> worker_${sg}-${node}.stderr > worker_${sg}-${node}.stdout --local-directory=/dev/shm &
