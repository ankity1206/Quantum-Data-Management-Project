# gpu_utils.py

from pynvml import *
import time


def init_gpu():
    try:
        nvmlInit()
        return True
    except:
        return False


def shutdown_gpu():
    try:
        nvmlShutdown()
    except:
        pass


def get_gpu_used_mb():
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except:
        return None


def track_peak_gpu_memory_during(func, *args, **kwargs):
    """
    Runs function and tracks peak GPU memory usage.
    """
    peak = 0
    result = None

    def wrapper():
        nonlocal peak
        result_local = func(*args, **kwargs)
        return result_local

    start_time = time.time()

    # Poll GPU memory during execution
    while True:
        current = get_gpu_used_mb()
        if current is not None:
            peak = max(peak, current)

        if time.time() - start_time > 0.01:
            break

    result = func(*args, **kwargs)

    # Final read
    current = get_gpu_used_mb()
    if current is not None:
        peak = max(peak, current)

    return result, peak
