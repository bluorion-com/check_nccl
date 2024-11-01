pip install torch

python3 -m torch.distributed.launch \
--nproc-per-node=8 \
check_nccl.py # (--arg1 --arg2 --arg3)

