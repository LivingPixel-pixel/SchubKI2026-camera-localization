"""Portable local CUDA job containment. Standard library only, no asset downloads.

Windows Job structures follow the project's reviewed runtime-envelope implementation.
This is a separately named eight-hour policy, not a change to that helper.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

GIB = 1024**3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def sha_file(path, cap=2*GIB):
    path = Path(path)
    size = path.stat().st_size
    if not 0 < size <= cap:
        raise ValueError("file size outside declared cap")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            count += len(block)
            if count > size:
                raise ValueError("file grew while hashing")
            digest.update(block)
    if count != size or path.stat().st_size != size:
        raise ValueError("file changed while hashing")
    return digest.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def verify_pin(value, cap=2*GIB):
    if set(value) != {"path", "sha256", "bytes"} or not Path(value["path"]).is_absolute():
        raise ValueError("absolute pinned input required")
    if Path(value["path"]).stat().st_size != value["bytes"] or sha_file(value["path"], cap) != value["sha256"]:
        raise ValueError("input identity changed")
    return Path(value["path"])


class Disk:
    """Checks existing output plus bounded transient writes before allocation."""
    def __init__(self, root, maximum, checkpoint_maximum=2*GIB):
        self.root = Path(root).resolve()
        self.maximum, self.checkpoint_maximum = maximum, checkpoint_maximum

    def total(self):
        result = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("output symlink forbidden")
            if path.is_file():
                result += path.stat().st_size
        return result

    def atomic(self, name, writer, cap):
        destination = (self.root / name).resolve()
        if destination.parent != self.root or cap > self.maximum:
            raise ValueError("invalid bounded output route")
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise ValueError("preserve stale partial output for operator inspection")
        if self.total() + cap > self.maximum or shutil.disk_usage(self.root).free < cap + 128*1024**2:
            raise OSError("insufficient declared output or free disk reserve")
        class Bounded:
            def __init__(self, stream):
                self.stream, self.count = stream, 0
            def write(self, raw):
                if self.count + len(raw) > cap:
                    raise OSError("single output size cap")
                count = self.stream.write(raw)
                self.count += count
                return count
            def flush(self):
                return self.stream.flush()
            def tell(self):
                return self.stream.tell()
        try:
            with temporary.open("xb") as stream:
                writer(Bounded(stream))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return pin(destination)

    def json(self, name, value):
        raw = canonical(value)
        return self.atomic(name, lambda stream: stream.write(raw), max(len(raw), 4096))


def private_commit_bytes():
    if os.name != "nt":
        raise RuntimeError("Windows process memory accounting required")
    class Counters(C.Structure):
        _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [(name, C.c_size_t) for name in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    k = C.WinDLL("kernel32", use_last_error=True)
    p = C.WinDLL("psapi", use_last_error=True)
    k.GetCurrentProcess.restype = W.HANDLE
    p.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
    p.GetProcessMemoryInfo.restype = W.BOOL
    counters = Counters()
    counters.cb = C.sizeof(counters)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(), C.byref(counters), C.sizeof(counters)):
        raise C.WinError(C.get_last_error())
    return int(counters.PrivateUsage)


class IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class Basic(C.Structure):
    _fields_ = [("PerProcessUserTimeLimit", C.c_longlong), ("PerJobUserTimeLimit", C.c_longlong),
        ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", C.c_size_t), ("MaximumWorkingSetSize", C.c_size_t),
        ("ActiveProcessLimit", W.DWORD), ("Affinity", C.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class Extended(C.Structure):
    _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), ("ProcessMemoryLimit", C.c_size_t),
        ("JobMemoryLimit", C.c_size_t), ("PeakProcessMemoryUsed", C.c_size_t), ("PeakJobMemoryUsed", C.c_size_t)]


class Startup(C.Structure):
    _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR), ("lpTitle", W.LPWSTR)] + [
        (n, W.DWORD) for n in ("dwX", "dwY", "dwXSize", "dwYSize", "dwXCountChars", "dwYCountChars", "dwFillAttribute", "dwFlags")
        ] + [("wShowWindow", W.WORD), ("cbReserved2", W.WORD), ("lpReserved2", C.c_void_p),
             ("hStdInput", W.HANDLE), ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]


class Process(C.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE), ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]


class Accounting(C.Structure):
    _fields_ = [(n, C.c_longlong) for n in ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")
        ] + [(n, W.DWORD) for n in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class WindowsJob:
    FLAGS = 0x2000 | 0x100 | 0x200

    def __init__(self, memory_bytes):
        if os.name != "nt":
            raise RuntimeError("Windows Job containment required")
        self.k = C.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            "CreateProcessW": ([W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL, W.DWORD, C.c_void_p, W.LPCWSTR, C.POINTER(Startup), C.POINTER(Process)], W.BOOL),
            "AssignProcessToJobObject": ([W.HANDLE, W.HANDLE], W.BOOL),
            "IsProcessInJob": ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
            "ResumeThread": ([W.HANDLE], W.DWORD), "WaitForSingleObject": ([W.HANDLE, W.DWORD], W.DWORD),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL), "TerminateProcess": ([W.HANDLE, W.UINT], W.BOOL),
            "CloseHandle": ([W.HANDLE], W.BOOL), "GetExitCodeProcess": ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL)}
        for name, (args, result) in signatures.items():
            getattr(self.k, name).argtypes, getattr(self.k, name).restype = args, result
        self.name = "SchubKI-ViT-" + str(uuid.uuid4())
        self.handle, self.process = self.k.CreateJobObjectW(None, self.name), None
        self.ok(self.handle)
        try:
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = self.FLAGS
            limits.ProcessMemoryLimit = limits.JobMemoryLimit = memory_bytes
            self.ok(self.k.SetInformationJobObject(self.handle, 9, C.byref(limits), C.sizeof(limits)))
            actual = Extended()
            self.ok(self.k.QueryInformationJobObject(self.handle, 9, C.byref(actual), C.sizeof(actual), None))
            if (actual.BasicLimitInformation.LimitFlags != self.FLAGS or actual.ProcessMemoryLimit != memory_bytes
                    or actual.JobMemoryLimit != memory_bytes):
                raise RuntimeError("Job limit readback mismatch")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def ok(value):
        if not value:
            raise C.WinError(C.get_last_error())

    def launch(self, command):
        startup, process = Startup(), Process()
        startup.cb = C.sizeof(startup)
        self.ok(self.k.CreateProcessW(None, C.create_unicode_buffer(subprocess.list2cmdline(command)), None, None,
            False, 0x4 | 0x08000000, None, None, C.byref(startup), C.byref(process)))
        self.process = process
        try:
            self.ok(self.k.AssignProcessToJobObject(self.handle, process.hProcess))
            contained = W.BOOL()
            self.ok(self.k.IsProcessInJob(process.hProcess, self.handle, C.byref(contained)))
            if not contained.value or self.k.ResumeThread(process.hThread) != 1:
                raise RuntimeError("suspended Job admission failed")
        except BaseException:
            self.k.TerminateProcess(process.hProcess, 1)
            raise
        finally:
            self.k.CloseHandle(process.hThread)
        return int(process.dwProcessId)

    def poll(self):
        result = self.k.WaitForSingleObject(self.process.hProcess, 0)
        if result == 258:
            return None
        if result != 0:
            raise RuntimeError("process wait failed")
        code = W.DWORD()
        self.ok(self.k.GetExitCodeProcess(self.process.hProcess, C.byref(code)))
        return int(code.value)

    def close(self):
        if not self.handle:
            return
        self.k.TerminateJobObject(self.handle, 0xE2)
        deadline = time.monotonic() + 10
        empty = False
        while time.monotonic() < deadline:
            accounting = Accounting()
            self.ok(self.k.QueryInformationJobObject(self.handle, 1, C.byref(accounting), C.sizeof(accounting), None))
            root_done = self.process is None or self.k.WaitForSingleObject(self.process.hProcess, 0) == 0
            if accounting.ActiveProcesses == 0 and root_done:
                empty = True
                break
            time.sleep(.05)
        if self.process:
            self.k.CloseHandle(self.process.hProcess)
        self.k.CloseHandle(self.handle)
        self.handle = None
        if not empty:
            raise RuntimeError("Job process tree cleanup not confirmed")


def notify(state):
    if os.name == "nt":
        try:
            subprocess.Popen(["msg", "*", "SchubKI ViT CUDA: " + state], creationflags=0x08000000,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


def require_containment(memory_bytes, job_name):
    if os.name != "nt":
        raise RuntimeError("Windows Job required")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
    kernel.QueryInformationJobObject.restype = W.BOOL
    kernel.OpenJobObjectW.argtypes, kernel.OpenJobObjectW.restype = [W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.IsProcessInJob.argtypes, kernel.IsProcessInJob.restype = [W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [W.HANDLE], W.BOOL
    handle = kernel.OpenJobObjectW(0x4, False, job_name)  # JOB_OBJECT_QUERY
    if not handle:
        raise RuntimeError("cannot open the declared containment Job")
    try:
        actual, member = Extended(), W.BOOL()
        if (not kernel.QueryInformationJobObject(handle, 9, C.byref(actual), C.sizeof(actual), None)
                or not kernel.IsProcessInJob(kernel.GetCurrentProcess(), handle, C.byref(member)) or not member.value):
            raise RuntimeError("child is not in the declared Windows Job")
        if (actual.BasicLimitInformation.LimitFlags & WindowsJob.FLAGS != WindowsJob.FLAGS
                or actual.ProcessMemoryLimit != memory_bytes or actual.JobMemoryLimit != memory_bytes):
            raise RuntimeError("child Job limits differ from declared allocation")
    finally:
        kernel.CloseHandle(handle)


def process_alive(pid):
    """Conservative process-exit evidence. Unknown access never proves exit."""
    if os.name != "nt":
        raise RuntimeError("Windows host exit accounting required")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [W.DWORD, W.BOOL, W.DWORD], W.HANDLE
    kernel.WaitForSingleObject.argtypes, kernel.WaitForSingleObject.restype = [W.HANDLE, W.DWORD], W.DWORD
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [W.HANDLE], W.BOOL
    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False if C.get_last_error() == 87 else None  # nonexistent PID vs unknown/access denied
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        return False if result == 0 else True if result == 258 else None
    finally:
        kernel.CloseHandle(handle)


def inspect_status(output, expected_config_sha256, expected_attempt_id=None):
    """A provisional shared record alone is never successful final evidence."""
    output = Path(output)
    result = {name: strict((output/name).read_bytes()) if (output/name).exists() else None
              for name in ("host.json", "status.json", "budget.json")}
    host = result["host.json"]
    final = {"phase": "provisional", "sealed": False, "host_exited": False, "success": False}
    result["finalization"] = final
    if not host:
        return result
    if expected_attempt_id is not None and host.get("attempt_id") != expected_attempt_id:
        final["phase"] = "attempt_mismatch"
        return result
    if host.get("config_sha256") != expected_config_sha256:
        final["phase"] = "config_mismatch"
        return result
    final["phase"] = "running" if host.get("state") == "running" else "provisional"
    attempt = host.get("attempt_id")
    try:
        if not isinstance(attempt, str) or str(uuid.UUID(attempt)) != attempt:
            return result
    except ValueError:
        return result
    name = "host-final-"+attempt+".json"
    if host.get("final_receipt") != name or not (output/name).is_file():
        return result
    receipt = strict((output/name).read_bytes())
    if (receipt.get("schema") != "schubki-vit-host-final-v1" or receipt.get("attempt_id") != attempt
            or receipt.get("host_pid") != host.get("host_pid")
            or receipt.get("config_sha256") != expected_config_sha256):
        return result
    result["final_receipt"] = receipt
    if receipt.get("sealed") is not True:
        return result
    final["sealed"] = True
    pid = receipt.get("host_pid")
    alive = process_alive(pid) if type(pid) is int and pid > 0 else None
    final["host_exited"] = alive is False
    final["phase"] = "host_exited" if alive is False else "sealed" if alive is True else "sealed_exit_unknown"
    final["terminal_state"] = receipt.get("state")
    final["exit_observed"] = False
    exit_name = "host-exit-"+attempt+".json"
    if host.get("exit_receipt") == exit_name and (output/exit_name).is_file():
        observed = strict((output/exit_name).read_bytes())
        if (observed.get("schema") == "schubki-vit-host-exit-v1" and observed.get("attempt_id") == attempt
                and observed.get("host_pid") == pid and observed.get("config_sha256") == expected_config_sha256
                and type(observed.get("observer_pid")) is int and observed["observer_pid"] > 0
                and observed["observer_pid"] != pid and type(observed.get("exit_code")) is int):
            result["observed_exit_receipt"] = observed
            final["exit_observed"] = True
            final["exit_code"] = observed["exit_code"]
    observed = result.get("observed_exit_receipt", {})
    exit_time, observed_time, wall_end, host_start = (observed.get("host_exit_unix_seconds"),
        observed.get("observed_exit_unix_seconds"), receipt.get("wall_deadline_unix"), receipt.get("host_started_unix"))
    exit_within_wall = (all(type(v) in (int, float) and math.isfinite(v) for v in (exit_time, observed_time, wall_end, host_start))
                        and host_start <= exit_time <= wall_end and observed_time >= exit_time)
    final["success"] = (alive is False and final["exit_observed"] and observed.get("exit_code") == 0
        and exit_within_wall and not (output/"host.lock").exists()
        and receipt.get("process_tree_empty") is True and receipt.get("wall_exceeded") is False
        and receipt.get("state") in ("probe_complete", "pilot_complete", "complete"))
    if strict((output/"host.json").read_bytes()) != host:
        final.update(phase="status_changed", success=False)
    return result


def supervise(config_path, mode, resume=False, child_command=None, attempt_id=None, expected_config_sha256=None):
    """Exclusive detached host, one finite budget across probe/pilot/resumes."""
    from .vit_finetune import read_config
    config = read_config(config_path)
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
        cli = Path(__file__).parents[1] / "tools/train_vit_finetune.py"
        command = child_command or [sys.executable, "-B", str(cli), "_child", "--config", str(Path(config_path).resolve()),
                                    "--mode", mode, "--deadline", str(deadline)] + (["--resume"] if resume else [])
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
