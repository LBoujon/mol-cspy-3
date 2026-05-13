crystal=$1
sg=$2
. ~/.bashrc
source activate cspy
dask-scheduler --interface ib0 --scheduler-file scheduler_${sg}.json 2> scheduler_${sg}.stderr > scheduler_${sg}.stdout &
dask-worker --nprocs=38 --interface ib0 --scheduler-file=scheduler_${sg}.json 2> worker_${sg}.stderr > worker_${sg}.stdout --local-directory=/dev/shm &
csp ${crystal}.xyz --scheduler-file scheduler_${sg}.json -a ${crystal}.mols -c ${crystal}_rank0.dma -m ${crystal}.dma -g "${sg}" -n 10000 --log-level DEBUG
