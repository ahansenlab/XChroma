"""Single-GPU DNA→ATAC training with resumable optimizer, schedule and data order."""
import argparse
from contextlib import nullcontext
import csv
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .data import TrainingBatches, example, read_dataset
from .io import fingerprint, write_json
from .loss import atac_loss, pool128, predictions_scaling, targets_scaling
from .model import DNA2ATAC

DEFAULTS = dict(batch_size=8, max_steps=50000, lr=2e-4, min_lr=2e-5, warmup=250,
                weight_decay=0.01, grad_clip=1.0, shift_bp=1000,
                reverse_complement_probability=0.5, seed=42, val_seed=42, val_windows=20,
                validate_every=500, save_every=100, loader_workers=4, threads=4, precision="bf16")
MODEL_DEFAULTS = dict(seq_len=1048576, width=512, n_dilated=13, dilation_cycles=1,
                      pool_factors=[5, 5, 5, 1, 1, 1], up_factors=[5, 5, 5],
                      up_channels=[128, 64, 32], grad_checkpoint=True)


def configure(user, metadata):
    if set(user) - (set(DEFAULTS) | {"model"}):
        raise ValueError(f"Unknown training keys: {set(user) - (set(DEFAULTS) | {'model'})}")
    config = {**DEFAULTS, **user}
    config["model"] = {**MODEL_DEFAULTS, **user.get("model", {})}
    mean = metadata["track_mean"]
    if "track_mean" in config["model"] and config["model"]["track_mean"] != mean:
        raise ValueError("Model track_mean differs from the dataset")
    config["model"]["track_mean"] = mean
    for key in ("batch_size", "max_steps", "val_windows", "validate_every", "save_every", "threads"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["loader_workers"] < 0 or config["shift_bp"] < 0 or not 0 <= config["warmup"] < config["max_steps"]:
        raise ValueError("Invalid worker count, shift range or warmup")
    if not 0 <= config["reverse_complement_probability"] <= 1 or not 0 <= config["min_lr"] <= config["lr"]:
        raise ValueError("Invalid augmentation probability or learning-rate range")
    if config["lr"] <= 0 or config["grad_clip"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer parameters")
    if config["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    return config


def lr_for(step, config):
    if step <= config["warmup"]:
        return config["lr"] * step / max(1, config["warmup"])
    phase = min(1., (step - config["warmup"]) / max(1, config["max_steps"] - config["warmup"]))
    return config["min_lr"] + (config["lr"] - config["min_lr"]) * 0.5 * (1 + math.cos(math.pi * phase))


def autocast(config, device):
    return torch.autocast(device.type, dtype=torch.bfloat16) if config["precision"] == "bf16" else nullcontext()


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def validate(model, root, rows, config, device):
    model.eval()
    records = []
    mean = model.track_mean
    for row in rows:
        dna, counts, observed = example(root, row, model.seq_len)
        truth = torch.from_numpy(counts[None]).to(device)
        with autocast(config, device):
            predictions = model(torch.from_numpy(dna[None]).to(device))
        loss, _ = atac_loss(predictions, truth, mean, observed)
        target = (targets_scaling(truth, mean), targets_scaling(pool128(truth), mean * 128))
        reference, _ = atac_loss(target, truth, mean)
        left, right = observed
        pred = predictions_scaling(predictions[0][0, left:right, 0], mean).cpu().numpy()
        real = counts[:, 0]
        n = len(real) // 1000 * 1000
        offset = (len(real) - n) // 2
        pk = pred[offset:offset + n].reshape(-1, 1000).sum(1)
        tk = real[offset:offset + n].reshape(-1, 1000).sum(1)
        records.append(dict(path=row["path"], chrom=row["chrom"], observed_slice=list(observed),
                            loss=float(loss), excess_loss=float(loss - reference),
                            r_1bp=correlation(pred, real), r_1kb=correlation(pk, tk),
                            r_log1p_1kb=correlation(np.log1p(pk), np.log1p(tk))))
    model.train()
    return records


def atomic_save(value, path):
    tmp = path.with_name(path.name + ".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def trim_uncommitted_log(path, step):
    if not path.exists():
        return
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    keep = [row for row in rows if int(row["step"]) <= step]
    if len(keep) != len(rows):
        path.replace(path.with_name(f"{path.name}.before_resume_{time.time_ns()}"))
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader(); writer.writerows(keep)


def train(root, run, config_path, device_name="cuda", resume=False, stop_after=None):
    root, run = Path(root).resolve(), Path(run).resolve()
    metadata, rows, dataset_id = read_dataset(root)
    config = configure(json.loads(Path(config_path).read_text()), metadata)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; --device cpu is available for explicit small tests")
    if device.type == "cuda" and config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; explicitly select fp32")
    train_rows = [r for r in rows if r["subset"] == "TRAIN"]
    valid_rows = [r for r in rows if r["subset"] == "VALID"]
    if not train_rows or not valid_rows:
        raise ValueError("TRAIN and VALID records are required; TEST is never used in training")
    vrng = np.random.default_rng(config["val_seed"])
    valid_rows = [valid_rows[i] for i in vrng.choice(len(valid_rows), min(len(valid_rows), config["val_windows"]), replace=False)]
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("models", "logs"):
            (run / name).mkdir(exist_ok=True)
        latest = run / "models/latest.pt"
        if latest.exists() != resume:
            raise ValueError("Use --resume with an existing latest.pt; otherwise use a fresh run directory")
        if (run / "config.json").exists() and json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Training configuration changed across resume")
        write_json(run / "config.json", config)
        write_json(run / "validation_records.json", valid_rows)
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = DNA2ATAC(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999),
                                     weight_decay=config["weight_decay"], foreach=False)
        completed, samples = 0, 0
        if resume:
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state.get("format") != "xatac-v1" or state["dataset_id"] != dataset_id or state["config"] != config:
                raise ValueError("Checkpoint format, dataset or training recipe changed")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            completed, samples = state["step"], state["samples"]
            del state
        target = config["max_steps"] if stop_after is None else min(stop_after, config["max_steps"])
        if target < completed:
            raise ValueError("Stop step precedes the restored checkpoint")
        trim_uncommitted_log(run / "train.tsv", completed)
        write_json(run / "runtime.json", dict(dataset_id=dataset_id, config_sha256=fingerprint(config),
                   device=str(device), torch=torch.__version__, numpy=np.__version__,
                   cuda=torch.version.cuda, precision=config["precision"], resumed_at_step=completed,
                   tf32=False, parameters=sum(p.numel() for p in model.parameters())))
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            batches = TrainingBatches(root, train_rows, config, completed + 1, target)
            kwargs = dict(num_workers=config["loader_workers"], batch_size=None, pin_memory=device.type == "cuda",
                          generator=torch.Generator().manual_seed(config["seed"]))
            if config["loader_workers"]:
                kwargs.update(prefetch_factor=1, persistent_workers=True, multiprocessing_context="spawn")
            loader = torch.utils.data.DataLoader(batches, **kwargs)
            last = time.monotonic()
            print(f"[{'resume' if resume else 'start'}] step={completed} target={target} device={device}", flush=True)
            for batch in loader:
                step = int(batch["step"])
                ready = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                lr = lr_for(step, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, raw = batch["x"].to(device, non_blocking=True), batch["raw"].to(device, non_blocking=True)
                with autocast(config, device):
                    predictions = model(x)
                loss, parts = atac_loss(predictions, raw, model.track_mean)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                completed, samples = step, samples + config["batch_size"]
                stats = dict(step=step, samples=samples, loss=float(loss.detach()), lr=lr, grad_norm=float(grad),
                             data_seconds=ready - last, compute_seconds=time.monotonic() - ready,
                             **{k: float(v.detach()) for k, v in parts.items()})
                with (run / "train.tsv").open("a") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(stats), delimiter="\t")
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(stats)
                del x, raw, predictions, loss, parts
                validate_now = step % config["validate_every"] == 0 or step == target
                if step % config["save_every"] == 0 or validate_now or stop:
                    state = dict(format="xatac-v1", model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 config=config, dataset_id=dataset_id, step=step, samples=samples,
                                 torch_rng=torch.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
                    atomic_save(state, latest)
                    if validate_now:
                        atomic_save({k: state[k] for k in ("format", "model", "config", "dataset_id", "step", "samples")},
                                    run / "models" / f"step{step}.pt")
                    del state
                if validate_now and not stop:
                    results = validate(model, root, valid_rows, config, device)
                    write_json(run / "logs" / f"val_step{step}.json", dict(step=step, windows=results))
                if step <= 3 or step % 10 == 0 or stop or step == target:
                    print(f"[train] step={step} loss={stats['loss']:.6g} lr={lr:.6g} data={stats['data_seconds']:.3f}s compute={stats['compute_seconds']:.3f}s", flush=True)
                if stop:
                    break
                last = time.monotonic()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        status = "COMPLETE" if completed == config["max_steps"] else "STOPPED_RESUMABLE"
        write_json(run / "status.json", dict(status=status, step=completed, target_steps=config["max_steps"], samples=samples))
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Stop safely at this absolute step without changing the LR schedule")
    args = parser.parse_args()
    train(args.data, args.run, args.config, args.device, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
