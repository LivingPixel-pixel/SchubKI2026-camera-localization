"""Process telemetry with explicit measurement boundaries for recorded replay."""
import platform
import sys


def memory_report() -> dict:
    value = None
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                              "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                value = counters.PeakWorkingSetSize / (1024 * 1024)
        else:
            import resource
            raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value = raw / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    except (ImportError, OSError, AttributeError, ValueError):
        pass
    return {"peak_process_ram_mb": value,
            "memory_scope": "Peak process working set since server launch; includes all jobs and map construction, not exclusive per-run allocation." if value is not None else "Process memory telemetry unavailable on this host.",
            "gpu_memory_mb": None, "gpu_memory_reason": "Not applicable: this candidate uses CPU-only inference.",
            "output_age_ms": None, "output_age_reason": "Recorded replay has no live capture-to-publication clock. Frame processing latency is measured separately."}
