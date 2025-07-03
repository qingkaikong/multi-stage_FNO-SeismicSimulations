import torch.distributed as dist
import torch

import os

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    
def setup_Tioga():
    # LC HACK: to inspect CPU and GPU binding (optional)
    import psutil
    import socket
    p = psutil.Process()
    rank = int(os.environ['FLUX_TASK_RANK'])
    size = int(os.environ['FLUX_JOB_SIZE'])
    local_rank = int(os.environ['FLUX_TASK_LOCAL_ID'])
    gpus = os.environ['ROCR_VISIBLE_DEVICES']

    print("rank", rank, "size", size, socket.gethostname(), "local_rank", local_rank, "gpus", gpus, "cpus", p.cpu_affinity(), flush=True)

    dist.init_process_group(backend="nccl", init_method="env://",
        world_size=int(os.environ['FLUX_JOB_SIZE']),
        rank=int(os.environ['FLUX_TASK_RANK']))

    dist.barrier()
    # lookup number of ranks in the job, and our rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print("rank", rank, "size", world_size, flush=True)
    
    
    # get number of GPUs the calling process can access
    # NOTE: this prints a warning: "UserWarning: Can't initialize NVML"
    # see https://rzlc.llnl.gov/jira/browse/ELCAP-387
    ngpus_per_node = torch.cuda.device_count()

    # compute our local rank on the node and select a corresponding gpu,
    # this assumes we run one rank per gpu on each compute node
    local_rank = rank % ngpus_per_node
    torch.cuda.set_device(local_rank)
    print("gpus", ngpus_per_node, "local_rank", local_rank, flush=True)
    
    return local_rank, rank, world_size

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0