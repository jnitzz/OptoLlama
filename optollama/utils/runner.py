import os
import random

import numpy as np
import torch
import torch.distributed as dist


def init_distributed() -> tuple[str, int, int, int]:
    """
    Init torch.distributed if plausible.

    Args
    ----
    backend: str
        The communication backend to use, defaults to: "nccl"

    Returns
    -------
    tuple
        A tuple with (device, local_rank, rank, world_size)
    """
    rank = int(os.getenv("SLURM_PROCID", 0))  # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS", 1))  # Total number of processes.
    local_rank = int(os.getenv("SLURM_LOCALID", rank))  # Local GPU ID.
    
    ddp = False
    device = "cpu"
    backend = "gloo"
    
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    device = torch.device(device)

    if dist.is_available() and not dist.is_initialized() and world_size > 1:
        ddp = True
        slurm_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
        os.environ["MASTER_ADDR"] = slurm_addr
        os.environ["MASTER_PORT"] = "29500"
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device,
            init_method="env://",
        )

    if rank == 0:
        print(
            f"[DDP={ddp}] backend={backend} world={world_size} rank={rank} "
            f"local_rank={local_rank} "
            f"device={device}"
        )

    return device, local_rank, rank, world_size


def set_all_seeds(seed: int = 42) -> None:
    """
    Set numpy, random, torch and cuda seed.
    
    Args
    ----
    seed: int
        The random number generator seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_torch_options() -> None:
    """Set basic torch options."""
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_printoptions(threshold=int(1e10))


def setup_run(cfg: dict, make_dirs: bool = False) -> tuple[str, int, int, int]:
    """
    Run init_distributed, set the seeds and create dirs.
    
    Args
    ----
    cfg: dict
        The configuration.
    make_dirs: bool
        Whether to create the directories if not existing, defaults to False.
    
    Returns
    -------
    tuple[str, int, int, int]
        The device string, node local rank, global rank and world size
    """
    device, local_rank, rank, world_size = init_distributed()
    seed = cfg["SEED"] or random.randint(1, int(1e6))
    set_all_seeds(seed)
    set_torch_options()
    if make_dirs:
        os.makedirs(cfg["OUTPUT_PATH"], exist_ok=True)
        
    return device, local_rank, rank, world_size
    

def is_ddp() -> bool:
    """
    Determines whether ddp is running.
    
    Returns
    -------
    bool
        Whether ddp is running.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    
    
def stop_ddp() -> None:
    """Destroys the DDP group."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
