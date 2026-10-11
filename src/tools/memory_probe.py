"""Peak GPU memory of one full training step, worst case: every slot with all its windows valid.

The real model of the config (encoder, side pooling / target attention head), its AdamW groups,
the EMA copy and bf16/fp16 autocast as in training; the input is random, the size is the
config's (data.image_size, data.centers_per_series, train.microbatch_studies). Use it before a
first run of a new encoder on a given GPU, e.g. to decide model.grad_checkpointing.

    cd src
    python tools/memory_probe.py --set model.backbone=coatnet_0_rw --set model.weights=sw_in1k \
        --set data.encoder_normalization=inception --set data.image_size=320 \
        --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true \
        --set train.ema_decay=0.999 --set model.grad_checkpointing=false

Pretrained weights are not needed for the measurement: model.weights is replaced by none.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri.config import load_config  # noqa: E402
from knee_mri.ema import ModelEMA  # noqa: E402
from knee_mri.model import build_model  # noqa: E402
from knee_mri.utils import autocast_ctx, select_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: src/config.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Dotted config override")
    parser.add_argument("--steps", type=int, default=3, help="Training steps; the peak is taken over all of them")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.model.weights = "none"  # the memory does not depend on the pretrained values
    spec = select_device(cfg.train.amp)
    if spec.device.type != "cuda":
        sys.exit("memory_probe needs a CUDA device")
    torch.manual_seed(0)
    model = build_model(cfg).to(spec.device).train()
    ema_decay = float(cfg.train.get("ema_decay", 0.0) or 0.0)
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    optimizer = torch.optim.AdamW(
        model.parameter_groups(float(cfg.train.encoder_lr), float(cfg.train.head_lr), float(cfg.train.weight_decay))
    )

    b = int(cfg.train.microbatch_studies)
    p, s, size = len(cfg.data.series_slots), int(cfg.data.centers_per_series), int(cfg.data.image_size)
    images = torch.randn((b, p, s, 3, size, size), device=spec.device)
    valid = torch.ones((b, p, s), dtype=torch.bool, device=spec.device)
    present = torch.ones((b, p), dtype=torch.bool, device=spec.device)
    targets = torch.rand((b, 12), device=spec.device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(max(1, args.steps)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with autocast_ctx(spec):
            logits = model(images, valid, present)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(spec.device).total_memory / 2**30
    print(
        f"{cfg.model.backbone} | {b} study x {p} slots x {s} windows at {size} px | "
        f"grad_checkpointing={bool(cfg.model.get('grad_checkpointing', False))} | amp {spec.amp_dtype} | "
        f"peak allocated {peak:.2f} GiB, reserved {reserved:.2f} GiB of {total:.1f} GiB "
        f"({torch.cuda.get_device_name(spec.device)}) | step {min(times[1:] or times) * 1000:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
