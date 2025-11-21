import os, datetime, random
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional, Tuple

def init_distributed(
    *,
    force_backend: Optional[str] = None,   # "nccl" | "gloo" | "mpi" | "auto" | None
    prefer_slurm_addr: bool = True,        # use SLURM_LAUNCH_NODE_IPADDR if present
    prefer_tcp: bool = False,              # set NCCL_IB_DISABLE=1 (+ optional P2P off)
    timeout_minutes: float = 10.0,
    log_env: bool = False,
) -> Tuple[str, int, int, int]:
    """
    Initialize (or skip) torch.distributed with sensible defaults for:
      - SLURM multi-node (srun)
      - torchrun / launch
      - single-process local (e.g., Spyder)

    Returns: (device_str, local_rank, rank, world_size)
    """
    # Fast path: already initialized (e.g., inside a library re-entry)
    if dist.is_available() and dist.is_initialized():
        rank      = dist.get_rank()
        world_sz  = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        # print(rank,world_sz,local_rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device_str = f"cuda:{local_rank}"
        else:
            device_str = "cpu"
        return device_str, local_rank, rank, world_sz

    # Infer basic topology from env
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank       = int(os.environ.get("RANK",       os.environ.get("SLURM_PROCID", "0")))
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else int(os.environ.get("SLURM_LOCALID", "0"))

    # Single-process: no process group; just pick device and go (great for Spyder)
    if world_size <= 1 or not dist.is_available():
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        return device_str, 0, 0, 1

    # Rendezvous: prefer SLURM-provided node IP, else leave defaults or set loopback
    if prefer_slurm_addr:
        slurm_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
        if slurm_addr:
            os.environ.setdefault("MASTER_ADDR", slurm_addr)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    # Stable MASTER_PORT across all ranks (no per-rank free port!)
    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id and job_id.isdigit():
            os.environ["MASTER_PORT"] = str(29500 + (int(job_id) % 1000))  # 29500–30499
        else:
            os.environ["MASTER_PORT"] = "29500"

    # Optional: prefer TCP over IB when your fabric is unstable
    if prefer_tcp:
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        # Uncomment if you want to be extra strict on some clusters:
        # os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # Backend selection
    if force_backend in {"nccl", "gloo", "mpi"}:
        backend = force_backend
    else:
        # auto: NCCL if CUDA+NCCL available, else Gloo
        try:
            nccl_ok = dist.is_nccl_available()
        except Exception:
            nccl_ok = False
        backend = "nccl" if (torch.cuda.is_available() and nccl_ok) else "gloo"
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(minutes=float(timeout_minutes)),
        world_size=world_size,
        rank=rank,
    )

    # Map each process to a local GPU (if available)
    if torch.cuda.is_available():
        if local_rank_env is None:
            # Fallback mapping by global rank modulo GPUs/node
            num_gpus = max(1, torch.cuda.device_count())
            local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cpu"
    
    if log_env and rank == 0:
        print(
            f"[DDP] backend={backend} world={world_size} rank={rank} "
            f"local_rank={local_rank} master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
            f"cuda={torch.cuda.is_available()} ib_disabled={os.environ.get('NCCL_IB_DISABLE','0')}"
        )

    return device_str, local_rank, rank, world_size


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def set_all_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_torch_options() -> None:
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_printoptions(threshold=int(1e10))


def setup_run(cfg, make_dirs=False):
    device, local_rank, rank, world_size = init_distributed()
    seed = getattr(cfg, "SEED", random.randint(1, int(1e6)))
    set_all_seeds(seed)
    set_torch_options()
    if make_dirs:
        os.makedirs(cfg.PATH_RUN, exist_ok=True)
        os.makedirs(cfg.PATH_SAVED, exist_ok=True)
    return device, local_rank, rank, world_size
