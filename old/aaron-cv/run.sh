# /usr/bin/sh
export PATH=/mnt/lustre/share/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64:/mnt/lustre/share/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

N_GPU=1
NODE=1
JOB=$1
CFG=$2
MODE=$3
EPOCH=$4
srun -p NTU --mpi=pmi2 --gres=gpu:$N_GPU -n1 --ntasks-per-node=$NODE \
--job-name=$JOB --kill-on-bad-exit=1 -w SG-IDC1-10-51-1-40 \
python run.py --mode $MODE --cfg $CFG --epoch $EPOCH
