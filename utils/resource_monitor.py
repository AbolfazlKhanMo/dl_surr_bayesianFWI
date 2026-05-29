# resmon.py
import os, time, threading, collections
from typing import List, Tuple, Optional
try:
    import psutil
except ImportError as e:
    raise SystemExit("Please install psutil: pip install psutil") from e

# NVML is optional; we'll handle gracefully if unavailable
_nvml_ok = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

Sample = collections.namedtuple("Sample", "t rss_bytes gpu_bytes")

def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

class ResourceMonitor:
    """
    Monitors total RAM/GPU usage of the current process + all descendants.
    - RAM via psutil RSS
    - GPU via NVML (only processes that match our PID set)
    Use:
        mon = ResourceMonitor(interval=0.5)
        mon.start()
        mon.mark("phase name")  # optional, can call multiple times
        ...
        mon.stop()
        mon.report()
    Or use as a context manager.
    """
    def __init__(self, interval: float = 0.5, include_self: bool = True):
        self.interval = interval
        self.include_self = include_self
        self.root_pid = os.getpid()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Sample] = []
        # marks: list of (label, t_start, t_end or None)
        self._marks: List[Tuple[str, float, Optional[float]]] = []
        self._lock = threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.report()

    def start(self):
        if self._thread is not None:
            return
        self._t0 = time.time()
        # Start with a default mark that covers the whole run
        self._marks.append(("total", self._t0, None))
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # close any open marks
        with self._lock:
            now = time.time()
            for i, (label, t0, t1) in enumerate(self._marks):
                if t1 is None:
                    self._marks[i] = (label, t0, now)

    def mark(self, label: str):
        """Close previous open mark (if any), start a new mark window."""
        with self._lock:
            now = time.time()
            # close currently-open mark (the last one with None end)
            for i in range(len(self._marks)-1, -1, -1):
                lab, t0, t1 = self._marks[i]
                if t1 is None:
                    self._marks[i] = (lab, t0, now)
                    break
            # start new
            self._marks.append((label, now, None))

    def _descendant_pids(self) -> List[int]:
        # Build set of PIDs for root + children recursively
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return []
        procs = root.children(recursive=True)
        if self.include_self:
            procs.append(root)
        pids = []
        for p in procs:
            try:
                if p.is_running():
                    pids.append(p.pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        return pids

    def _total_rss(self, pids: List[int]) -> int:
        total = 0
        for pid in pids:
            try:
                p = psutil.Process(pid)
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
        return total

    def _total_gpu(self, pids: List[int]) -> int:
        if not _nvml_ok:
            return 0
        pidset = set(pids)
        total = 0
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Try v3, fallback to older API if needed
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)
                except AttributeError:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                for pr in procs:
                    # pr.pid, pr.usedGpuMemory (bytes, may be NVML_VALUE_NOT_AVAILABLE)
                    if getattr(pr, "pid", None) in pidset:
                        used = getattr(pr, "usedGpuMemory", 0) or 0
                        if used > 0:
                            total += used
        except Exception:
            # If NVML errors, just return what we have
            pass
        return total

    def _run(self):
        while not self._stop.is_set():
            pids = self._descendant_pids()
            rss = self._total_rss(pids)
            gpu = self._total_gpu(pids)
            with self._lock:
                self.samples.append(Sample(time.time(), rss, gpu))
            time.sleep(self.interval)

    def _window_stats(self, t0: float, t1: float) -> Tuple[int, int]:
        """Return (peak_rss_bytes, peak_gpu_bytes) within [t0, t1]."""
        peak_rss = 0
        peak_gpu = 0
        for s in self.samples:
            if t0 <= s.t <= t1:
                if s.rss_bytes > peak_rss:
                    peak_rss = s.rss_bytes
                if s.gpu_bytes > peak_gpu:
                    peak_gpu = s.gpu_bytes
        return peak_rss, peak_gpu

    def report(self):
        with self._lock:
            if not self.samples:
                print("[ResourceMonitor] No samples collected.")
                return
            # Overall peaks
            all_rss = max(s.rss_bytes for s in self.samples)
            all_gpu = max(s.gpu_bytes for s in self.samples)
            print("\n===== Resource Usage Summary =====")
            print(f"Peak Total RAM: { _bytes_to_mb(all_rss):.1f} MB")
            if _nvml_ok:
                print(f"Peak Total GPU Memory (tracked PIDs): { _bytes_to_mb(all_gpu):.1f} MB")
            else:
                print("Peak Total GPU Memory: NVML not available (install nvidia-ml-py3 / ensure NVIDIA driver).")

            # Per-mark peaks
            print("\n-- Peaks by Phase --")
            for label, t0, t1 in self._marks:
                if t1 is None:  # should be closed by stop()
                    continue
                prss, pgpu = self._window_stats(t0, t1)
                dur = t1 - t0
                print(f"[{label}] duration {dur:.2f}s | RAM peak { _bytes_to_mb(prss):.1f} MB"
                      + (f" | GPU memory peak { _bytes_to_mb(pgpu):.1f} MB" if _nvml_ok else ""))

            # Top contributors by process name (peak snapshot)
            try:
                # Find the sample with max RSS, then show its breakdown
                peak_sample = max(self.samples, key=lambda s: s.rss_bytes)
                # Approximate breakdown at that instant
                root = psutil.Process(self.root_pid)
                procs = root.children(recursive=True)
                if self.include_self:
                    procs.append(root)
                by_name = []
                for p in procs:
                    try:
                        mi = p.memory_info().rss
                        name = p.name()
                        by_name.append((name, mi))
                    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                        pass
                by_name.sort(key=lambda x: x[1], reverse=True)
                print("\n-- Largest Processes by RSS (approx at overall peak) --")
                for name, bytes_ in by_name[:10]:
                    print(f"{name:30s} { _bytes_to_mb(bytes_):6.1f} MB")
            except Exception:
                pass
            print("===== End Summary =====\n")
