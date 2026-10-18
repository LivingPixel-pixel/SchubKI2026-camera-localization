"""Laptop host dispatch over unchanged reviewed Windows Job/Disk/exit primitives."""
from __future__ import annotations
import ctypes
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from . import vit_finetune_runtime as base

GIB = base.GIB
canonical, strict, sha_file, pin, verify_pin = base.canonical, base.strict, base.sha_file, base.pin, base.verify_pin
Disk, WindowsJob = base.Disk, base.WindowsJob
private_commit_bytes, require_containment = base.private_commit_bytes, base.require_containment
inspect_status, process_alive = base.inspect_status, base.process_alive


def notify(state):
    """Best-effort local alert, with the same protected accounting as the donor."""
    if os.name == "nt":
        try:
            subprocess.Popen(["msg", "*", "SchubKI laptop: " + state], creationflags=0x08000000,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


def available_memory():
    if os.name != "nt":
        raise RuntimeError("Windows system RAM accounting required")
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise RuntimeError("Windows system RAM accounting unavailable")
    return int(value.available)


def supervise(config_path, mode, resume=False, child_command=None, attempt_id=None, expected_config_sha256=None,
              *, config_loader=None, child_entry=None, admission=None, required_free_memory=None):
    """Exclusive detached host, one finite budget across probe/pilot/resumes."""
    if config_loader is None:
        from .vit_laptop_finetune import read_config
        config_loader = read_config
    config = config_loader(config_path)
    config_sha = hashlib.sha256(canonical(config)).hexdigest()
    if expected_config_sha256 is not None and expected_config_sha256 != config_sha:
        raise ValueError("launch config changed after observer identity capture")
    attempt_id = str(uuid.uuid4()) if attempt_id is None else attempt_id
    if str(uuid.UUID(attempt_id)) != attempt_id:
        raise ValueError("canonical unique launch attempt required")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output/(prefix+attempt_id+".json")).exists() for prefix in ("host-final-", "host-exit-")):
        raise ValueError("launch attempt identity already used")
    disk = Disk(output, config["limits"]["disk_bytes"])
    lock = output / "host.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.monotonic()
    host_started_unix = time.time()
    budget_path = output / "budget.json"
    budget = None
    job = None
    hard_deadline = None
    charged = 0
    primary_error = None
    secondary_errors = []
    cleanup_confirmed = True
    state = "failed"
    receipt_name = "host-final-"+attempt_id+".json"
    attempt = {"attempt_id": attempt_id, "host_pid": os.getpid(), "final_receipt": receipt_name,
               "exit_receipt": "host-exit-"+attempt_id+".json", "config_sha256": config_sha,
               "host_started_unix": host_started_unix}

    def remember(error):
        nonlocal primary_error
        if primary_error is None:
            primary_error = error
        else:
            secondary_errors.append(type(error).__name__)

    def fence():
        """Account completed work, including cleanup and durable-write latency."""
        if budget is None:
            return False
        now = time.monotonic()
        budget["charged_seconds"] = max(budget["charged_seconds"], charged+now-started,
                                        time.time()-budget["started_unix"])
        late = (hard_deadline is not None and now >= hard_deadline) or budget["charged_seconds"] >= budget["wall_seconds"]
        if late and primary_error is None:
            remember(TimeoutError("original cumulative wall exceeded during finalization"))
        return late

    def final_state():
        final_state = state if primary_error is None else "failed_"+type(primary_error).__name__
        if not cleanup_confirmed:
            final_state = "cleanup_failed"
        return final_state

    def own_receipt(sealed):
        late = fence()
        return {**attempt, "schema": "schubki-vit-host-final-v1", "sealed": sealed, "state": final_state(),
            "process_tree_empty": cleanup_confirmed,
            "primary_error": None if primary_error is None else type(primary_error).__name__,
            "secondary_errors": secondary_errors.copy(), "wall_exceeded": late,
            "wall_deadline_unix": None if budget is None else budget["started_unix"]+budget["wall_seconds"],
            "charged_seconds": None if budget is None else budget["charged_seconds"]}

    try:
        if budget_path.exists():
            budget = strict(budget_path.read_bytes())
            if budget["wall_seconds"] != config["limits"]["wall_seconds"]:
                raise ValueError("cumulative wall budget cannot change")
            charged = max(budget["charged_seconds"], time.time()-budget["started_unix"])
        else:
            budget = {"started_unix": time.time(), "charged_seconds": 0,
                      "wall_seconds": config["limits"]["wall_seconds"]}
            charged = 0
        remaining = budget["wall_seconds"]-charged
        if remaining <= 10:
            raise TimeoutError("cumulative wall exhausted")
        hard_deadline = started + remaining
        deadline = hard_deadline - 10  # reserve cleanup, then fence its actual completion
        disk.json("budget.json", budget)
        cli = Path(__file__).parents[1] / "tools/train_vit_laptop.py" if child_entry is None else Path(child_entry)
        command = child_command or [sys.executable, "-B", str(cli), "_child", "--config", str(Path(config_path).resolve()),
                                    "--mode", mode, "--deadline", str(deadline)] + (["--resume"] if resume else [])
        limits = config["limits"]
        required = limits["ram_hard_bytes"] + limits["system_reserve_bytes"]
        if required_free_memory is not None:
            required = required_free_memory(config)
            if (type(required) is not int or required < limits["ram_soft_bytes"] + limits["system_reserve_bytes"]
                    or required > limits["ram_hard_bytes"] + limits["system_reserve_bytes"]):
                raise ValueError("trusted fresh-RAM requirement must be a native integer within soft/hard plus reserve")
        if available_memory() < required:
            raise MemoryError("fresh laptop RAM admission failed; frozen limits cannot change")
        if admission is not None:
            admission(config)
        job = WindowsJob(config["limits"]["ram_hard_bytes"])
        if child_command is None:
            command += ["--job-name", job.name]
        disk.json("child-result.json", {"state": "starting"})
        pid = job.launch(command)
        disk.json("host.json", {**attempt, "state": "running", "provisional": True, "pid": pid, "mode": mode,
                                "contained": True, "remaining_seconds_at_start": remaining})
        while job.poll() is None:
            budget["charged_seconds"] = charged + time.monotonic()-started
            disk.json("budget.json", budget)
            if time.monotonic() >= deadline:
                raise TimeoutError("hard cumulative wall reached")
            time.sleep(.5)
        code = job.poll()
        if time.monotonic() >= deadline:
            raise TimeoutError("child exit observed after reserved execution deadline")
        if code:
            raise RuntimeError("contained child failed; inspect local status")
        state = strict((output/"child-result.json").read_bytes()).get("state")
        if state not in ("probe_complete", "pilot_complete", "complete", "paused_memory"):
            raise RuntimeError("missing coherent child completion record")
    except BaseException as error:
        remember(error)
    finally:
        cleanup_confirmed = job is None
        try:
            if job:
                job.close()
                cleanup_confirmed = True
        except BaseException as error:
            remember(error)
        fence()
        # Notification belongs to the outgoing owner's protected finalization.
        try:
            notify("cleanup_failed" if not cleanup_confirmed else "finished; inspect durable status")
        except BaseException as error:
            remember(error)
        fence()
        shared_finalized = False
        try:
            if budget is not None:
                disk.json("budget.json", budget)
            fence()
            disk.json(receipt_name, own_receipt(False))
            disk.json("host.json", {**attempt, "state": "finalizing", "provisional": True,
                "candidate_state": final_state(), "process_tree_empty": cleanup_confirmed})
            fence()
            shared_finalized = True
        except BaseException as error:
            remember(error)
        # Unlink is the ownership handoff. No shared-record writes follow it on ANY path.
        try:
            if cleanup_confirmed and shared_finalized:
                lock.unlink(missing_ok=True)
        except BaseException as error:
            remember(error)
        fence()
        try:
            disk.json(receipt_name, own_receipt(True))
        except BaseException as error:
            remember(error)
        fence()
        if primary_error is not None:
            # Late sealing may change only this unique receipt, never a successor's shared records.
            try:
                disk.json(receipt_name, own_receipt(True))
            except BaseException as error:
                secondary_errors.append(type(error).__name__)
    if primary_error is not None:
        raise primary_error
    return {"state": state, **attempt, "sealed": True, "host_exited": False}
