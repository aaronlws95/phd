# /usr/bin/sh
export PATH=/mnt/lustre/share/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64:/mnt/lustre/share/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

N_GPU=$1
JOB=$2
CFG=$3
MODE=$4
EPOCH=$5
srun -p NTU --mpi=pmi2 --gres=gpu:$N_GPU -n1 --ntasks-per-node=$N_GPU \
--job-name=$JOB --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-40 \
python -m torch.distributed.launch \
--nproc_per_node=$N_GPU --master_port=2345 \
run.py --mode $MODE --cfg $CFG --epoch $EPOCH


