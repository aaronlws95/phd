# /usr/bin/sh
export PATH=/mnt/lustre/share/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64:/mnt/lustre/share/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

N_GPU=$1
NODE=$2
PORT=$3
EXP_NAME=$4
CONF=$5
EPOCH=$6
echo N_GPU=$N_GPU
echo NODE=$NODE
echo PORT=$PORT
echo EXP_NAME=$EXP_NAME
echo CONF=$CONF
echo EPOCH=$EPOCH
srun -p Platform --mpi=pmi2 --gres=gpu:$N_GPU -n1 --ntasks-per-node=$N_GPU --job-name=$EXP_NAME --kill-on-bad-exit=1 -w SH-IDC1-10-5-34-${NODE} python train.py --conf $CONF --epoch $EPOCH