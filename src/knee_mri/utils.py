"""Seeding, logging, device/AMP selection and small filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOGGER_NAME = "knee_mri"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_logging(path: str | Path, name: str = LOGGER_NAME) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(name)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s | %(message)s"))
    logger.addHandler(handler)


def remove_file_logging(directory: str | Path, name: str = LOGGER_NAME) -> None:
    """Close and detach every file handler writing below `directory` (Windows keeps them locked)."""
    root = Path(directory).resolve()
    logger = get_logger(name)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and root in Path(handler.baseFilename).resolve().parents:
            handler.close()
            logger.removeHandler(handler)


LOG = get_logger()


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch. This does not promise bitwise reproducibility across
    hardware, driver or library versions - only that the same machine repeats a run."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is required for training only
        pass


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible seed."""
    import torch

    base_seed = torch.initial_seed() % (2**31)
    seed = (base_seed + worker_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed % (2**32))


@dataclass
class DeviceSpec:
    device: Any
    amp_dtype: Any | None
    use_grad_scaler: bool
    name: str

    def describe(self) -> str:
        amp = "fp32" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")
        return f"device={self.device} ({self.name}) amp={amp} grad_scaler={self.use_grad_scaler}"


def select_device(amp_mode: str = "auto", device_str: str | None = None) -> DeviceSpec:
    """Pick the compute device and a hardware-appropriate AMP policy.

    bf16 is preferred when the GPU supports it (no GradScaler needed); otherwise
    fp16 + GradScaler; CPU always runs in float32.
    """
    import torch

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        if amp_mode not in ("auto", "fp32"):
            LOG.warning("AMP mode %s requested but device is %s; falling back to fp32", amp_mode, device.type)
        return DeviceSpec(device=device, amp_dtype=None, use_grad_scaler=False, name=platform.processor() or "cpu")

    name = torch.cuda.get_device_name(device.index or 0)
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    if amp_mode == "fp32":
        return DeviceSpec(device, None, False, name)
    if amp_mode == "bf16":
        if not bf16_ok:
            raise RuntimeError(f"train.amp=bf16 requested but {name} does not support bfloat16")
        return DeviceSpec(device, torch.bfloat16, False, name)
    if amp_mode == "fp16":
        return DeviceSpec(device, torch.float16, True, name)
    # auto
    if bf16_ok:
        return DeviceSpec(device, torch.bfloat16, False, name)
    return DeviceSpec(device, torch.float16, True, name)


@contextmanager
def autocast_ctx(spec: DeviceSpec) -> Iterator[None]:
    import torch

    if spec.amp_dtype is None:
        yield
    else:
        with torch.autocast(device_type=spec.device.type, dtype=spec.amp_dtype):
            yield


def environment_report() -> dict:
    """Collect the versions and GPU information worth storing next to every run."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for module_name in ("numpy", "pandas", "torch", "torchvision", "pydicom", "sklearn", "yaml", "matplotlib"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment dependent
            report[module_name] = f"MISSING ({exc.__class__.__name__})"
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                    "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - environment dependent
        report["torch_runtime_error"] = str(exc)
    return report


def log_environment() -> dict:
    report = environment_report()
    LOG.info("Environment: %s", json.dumps(report, default=str))
    return report


def stable_hash(payload: Any, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(data)
    os.replace(tmp, path)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def atomic_save_npz(path: str | Path, arrays: dict, meta: dict) -> Path:
    """Write a compressed npz with a JSON metadata entry, atomically."""
    import io

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    payload = dict(arrays)
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, default=str).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(buffer, **payload)
    return atomic_write_bytes(path, buffer.getvalue())


def load_npz_meta(npz: Any) -> dict:
    raw = npz["meta_json"]
    return json.loads(bytes(raw.tobytes()).decode("utf-8"))


def atomic_write_dataframe(df: Any, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)
    return path


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def peak_gpu_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:  # pragma: no cover - environment dependent
        pass
    return 0.0


def reset_peak_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover - environment dependent
        pass
