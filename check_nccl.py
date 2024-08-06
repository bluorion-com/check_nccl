import argparse

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int)
args = parser.parse_args()

print("Pytorch version", torch.__version__)

local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

torch.distributed.init_process_group(backend="nccl", init_method="env://")

# Allocate a tensor on the gpu
shape = 2**17
dtype = torch.float32
x = torch.randn(shape, dtype=dtype).to(device)

print("local result:", x.sum())

# Do a broadcast from rank 0
dist.broadcast(x, 0)
print("broadcast result:", x.sum())

# Do an all-reduce
dist.all_reduce(x)
print("allreduce result:", x.sum())

dist.destroy_process_group()
print("great success!")
