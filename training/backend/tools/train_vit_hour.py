"""One declared DirectML selective ViT hour from parent7750. No downloads or automatic retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"backend"))

from race_localization import vit_hour_finetune as vf
from race_localization import vit_laptop_runtime as rt


def initialize(args):
    base = rt.strict((ROOT/"docs/vit_hour.example.json").read_bytes())
    dataset = Path(args.dataset_dir).resolve()
    # Presence of a recording is not a prepared dataset.
    if not (dataset/"dataset.json").is_file():
        raise ValueError("prepared dataset.json required; see docs/VIT_FINETUNE.md preparation inputs")
    if rt.sha_file(dataset/"dataset.json", 1024**2) != args.dataset_sha256:
        raise ValueError("prepared dataset identity mismatch")
    base["dataset"] = {"directory": str(dataset), "sha256": args.dataset_sha256,
                       "cache_root": str(Path(args.cache_root).resolve())}
    for raw, maximum in ((args.warm_start, 512*1024**2), (args.warm_start_card, 16384)):
        path = Path(raw)
        if not path.is_file() or not 0 < path.stat().st_size <= maximum:
            raise ValueError("bounded hour parent artifact required")
    base["warm_start"], base["warm_start_card"] = rt.pin(args.warm_start), rt.pin(args.warm_start_card)
    base["output"] = str(Path(args.output).resolve())
    card = rt.strict(Path(args.warm_start_card).read_bytes())
    base["parent"]["training_identity"] = vf.sha(card["training_identity"])
    base["backend"] = args.backend
    if getattr(args, "five_lap", False):
        # Explicit v5 five-lap policy. Old example-derived configs are unchanged when the flag is absent.
        base.pop("baseline_import", None)
        soft, hard, reserve = vf.RAM_PROFILES[vf.SCHEMA_V5]
        base.update(schema=vf.SCHEMA_V5, lap_policy=vf.FIVE_POLICY)
        base["recipe"] = {**base["recipe"], "learning_rate": 1e-5, "max_updates": 1000}
        base["limits"] = {**base["limits"], "ram_soft_bytes": soft, "ram_hard_bytes": hard, "system_reserve_bytes": reserve}
        if rt.strict((dataset/"dataset.json").read_bytes()).get("schema") != "schubki-place-dataset-v2":
            raise ValueError("--five-lap requires a prepared schubki-place-dataset-v2 dataset")
    config = Path(args.config).resolve()
    if config.exists() or Path(base["output"]).exists():
        raise FileExistsError("init requires a new private config and output directory")
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("xb") as stream:
        stream.write(rt.canonical(base))
        stream.flush()
        os.fsync(stream.fileno())
    vf.read_config(config)
    return {"state": "configured", "config": str(config)}


def require_previous(config, mode, resume):
    if mode == "probe":
        if resume:
            raise ValueError("hour probe cannot resume or retry")
        return
    snapshot = rt.inspect_status(config["output"], vf.digest(config))
    final, receipt = snapshot["finalization"], snapshot.get("final_receipt", {})
    expected = "probe_complete" if mode == "pilot" else "pilot_complete"
    if final.get("success") is True and final.get("terminal_state") == expected:
        return
    if (resume and final.get("terminal_state") == "paused_memory" and final.get("sealed") is True
            and final.get("host_exited") is True and final.get("exit_observed") is True
            and type(final.get("exit_code")) is int and final["exit_code"] == 0
            and receipt.get("process_tree_empty") is True and receipt.get("wall_exceeded") is False
            and not (Path(config["output"])/"host.lock").exists() and (Path(config["output"])/"latest.pt").is_file()):
        return
    raise ValueError("matching safely observed previous phase required; preserve failure/partial evidence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("init")
    command.add_argument("--backend", choices=("directml",), default="directml")
    command.add_argument("--five-lap", action="store_true",
                         help="Write a schubki-vit-hour-config-v5 train1-5/validation6-7 config (no baseline import)")
    for name in ("dataset-dir", "dataset-sha256", "cache-root", "warm-start", "warm-start-card", "output", "config"):
        command.add_argument("--"+name, required=True)
    for name in ("probe", "pilot", "train", "status", "launch-info", "host", "_child"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name in ("pilot", "train", "host", "_child"):
            command.add_argument("--resume", action="store_true")
        if name in ("host", "_child"):
            command.add_argument("--mode", choices=("probe", "pilot", "train"), required=True)
        if name in ("host", "status"):
            command.add_argument("--attempt-id")
        if name == "host":
            command.add_argument("--expected-config-sha256")
        if name == "_child":
            command.add_argument("--deadline", required=True, type=float)
            command.add_argument("--job-name", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(args)
        else:
            config = vf.read_config(args.config)
            config_sha = hashlib.sha256(rt.canonical(config)).hexdigest()
            if args.command == "status":
                result = rt.inspect_status(config["output"], config_sha, args.attempt_id)
            elif args.command == "launch-info":
                result = {"config_sha256": config_sha, "output": config["output"], "disk_bytes": config["limits"]["disk_bytes"]}
            elif args.command == "_child":
                rt.require_containment(config["limits"]["ram_hard_bytes"], args.job_name)
                output = Path(config["output"])
                # Child tracebacks are local and bounded; no data-bearing console output.
                result = vf.probe(config, args.deadline) if args.mode == "probe" else vf.run(
                    config, args.deadline, pilot=args.mode == "pilot", resume=args.resume)
                rt.Disk(output, config["limits"]["disk_bytes"]).json("child-result.json", result)
            else:
                mode = args.mode if args.command == "host" else args.command
                require_previous(config, mode, getattr(args, "resume", False))
                result = rt.supervise(args.config, mode, getattr(args, "resume", False),
                    attempt_id=getattr(args, "attempt_id", None),
                    expected_config_sha256=getattr(args, "expected_config_sha256", None),
                    config_loader=vf.read_config, child_entry=ROOT/"backend/tools/train_vit_hour.py", admission=vf.admission,
                    required_free_memory=vf.admission_requirement)
        if args.command != "_child":
            print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        result = {"state": "failed", "type": type(error).__name__}
        if args.command == "_child":
            try:
                config = vf.read_config(args.config)
                rt.Disk(config["output"], config["limits"]["disk_bytes"]).json("child-error.json", {
                    **result, "local_message": str(error)[:500]})
            except Exception:
                pass
        else:
            print(json.dumps(result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
