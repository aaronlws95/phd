# NTU

## Workstations

* **NTU workstation**: `ssh aaron@172.21.146.106`
* **NTU micl-loyccws1**: `ssh aaron@172.21.148.106`
    * password: micl-loyccws
* **NTU micl-loyccws2**: `ssh aaron@172.21.145.2`
* **SenseTime workstation**: `ssh -p 10087 awslow@203.126.234.118`
    * password: UW@TY9FTD!UIF9$leizLuQ

## SenseTime

### Running GPU

PC Num ranges from 33 to 42

```
# /usr/bin/sh
export PATH=/mnt/lustre/share/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64:/mnt/lustre/share/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

srun -p NTU --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=8 --job-name=4200 \
--kill-on-bad-exit=1 -w SG-IDC1-10-51-1-{Insert PC Num} \
python main.py --cfg /mnt/lustre/awslow/mlcv-exp/data/cfg/yolo_hand_crop/fpha_yolo_hand_crop_exp3.cfg --mode train
```

### `nvidia-smi`

```
srun -p NTU -w SG-IDC1-10-51-1-36  nvidia-smi
```

### Syncing

* Local -> SenseTime

```
scp -r /mnt/4TB/aaron/mlcv-exp/data st-ws:/mnt/lustre/awslow/mlcv-exp
```

or

```
rsync -v --stats --progress -a /mnt/4TB/aaron/mlcv-exp/data/ st-ws:/mnt/lustre/awslow/mlcv-exp/data/ -avz --exclude 'root.txt' --exclude 'exp/' --exclude 'saved/'
```

* SenseTime -> Local

```
rsync -v --stats --progress -a st-ws:/mnt/lustre/awslow/mlcv-exp/data/ /mnt/4TB/aaron/mlcv-exp/data/ -avz --exclude 'root.txt'
```