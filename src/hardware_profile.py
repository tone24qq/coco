from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class HardwareProfile:
    cpu_logical: int
    cpu_physical: int
    total_ram_gb: float
    available_ram_gb: float
    cuda_available: bool
    mps_available: bool
    platform: str


@dataclass
class TrainingPlan:
    backend: str
    n_jobs: int
    num_workers: int
    batch_size: int
    shard_size: int
    prefetch: int


def detect_hardware_profile() -> HardwareProfile:
    cpu_logical = os.cpu_count() or 1
    cpu_physical = max(1, cpu_logical // 2)
    total_ram = 8.0
    available_ram = 4.0
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)
        available_ram = vm.available / (1024**3)
        cpu_physical = psutil.cpu_count(logical=False) or cpu_physical
    except Exception:
        pass

    cuda_available = False
    mps_available = False
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass

    return HardwareProfile(
        cpu_logical=int(cpu_logical),
        cpu_physical=int(cpu_physical),
        total_ram_gb=float(total_ram),
        available_ram_gb=float(available_ram),
        cuda_available=cuda_available,
        mps_available=mps_available,
        platform=f"{platform.system()}-{platform.machine()}",
    )


def choose_training_plan(
    profile: HardwareProfile,
    requested_device: str = "auto",
    max_workers: str = "auto",
) -> TrainingPlan:
    backend = "lightgbm" if profile.available_ram_gb >= 8 else "sklearn"
    if requested_device != "auto" and requested_device != "cpu":
        backend = "lightgbm"

    if max_workers == "auto":
        workers = max(1, min(profile.cpu_logical - 1, 8))
    else:
        workers = max(1, int(max_workers))

    ram_factor = max(1, int(profile.available_ram_gb // 2))
    batch_size = min(65536, 4096 * ram_factor)
    shard_size = min(2_000_000, 250_000 * ram_factor)
    prefetch = min(8, workers)

    return TrainingPlan(
        backend=backend,
        n_jobs=workers,
        num_workers=workers,
        batch_size=batch_size,
        shard_size=shard_size,
        prefetch=prefetch,
    )


def to_dict(obj: object) -> Dict[str, object]:
    return asdict(obj)  # type: ignore[arg-type]
