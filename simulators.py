"""
Simulation backends for G7 benchmark (FINAL UNIFIED VERSION)
Compatible with:
- Qiskit CPU Statevector
- Qiskit Aer GPU/CPU
- RDBMS simulator (same schema alignment)
"""

import numpy as np
import time
import psutil
import threading
from contextlib import contextmanager

from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# ==============================
# GPU MEMORY HELPER
# ==============================
def get_gpu_memory_mb():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except:
        return None


# ==============================
# RESOURCE TRACKER
# ==============================
class ResourceTracker:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.running = False
        self.cpu_peak = 0
        self.gpu_peak = 0
        self.process = psutil.Process()

    def _track(self):
        while self.running:
            self.cpu_peak = max(
                self.cpu_peak,
                self.process.memory_info().rss / (1024 ** 2)
            )

            gpu = get_gpu_memory_mb()
            if gpu is not None:
                self.gpu_peak = max(self.gpu_peak, gpu)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

        return self.cpu_peak, self.gpu_peak


# ==============================
# CONTEXT MANAGER
# ==============================
@contextmanager
def measure_resources(label=""):
    start = time.perf_counter()

    tracker = ResourceTracker()
    tracker.start()

    metrics = {}

    try:
        yield metrics
    finally:
        end = time.perf_counter()
        cpu_peak, gpu_peak = tracker.stop()

        metrics.update({
            "label": label,
            "time_s": end - start,
            "cpu_memory_mb": cpu_peak,
            "gpu_memory_mb": gpu_peak,
        })


# ============================================================
# CPU STATEVECTOR SIMULATOR
# ============================================================
class StatevectorSimulator:
    def __init__(self):
        self.name = "Statevector (CPU)"

    def simulate(self, circuit, shots=1024):

        with measure_resources(self.name) as metrics:

            circuit = circuit.remove_final_measurements(inplace=False)

            sv = Statevector.from_instruction(circuit)

            probs = sv.probabilities()

            probs = np.array(probs, dtype=float)

            n = circuit.num_qubits

            samples = np.random.choice(len(probs), size=shots, p=probs)

            bitstrings = [format(s, f'0{n}b') for s in samples]

            counts = {b: bitstrings.count(b) for b in set(bitstrings)}

            metrics.update({
                "state_size": len(probs)
            })

            return {
                "counts": counts,
                "metrics": metrics
            }


# ============================================================
# AER GPU / CPU SIMULATOR
# ============================================================
class AerSimulatorBackend:
    def __init__(self, method="statevector", device="GPU"):
        self.name = f"Aer ({method}, {device})"

        self.backend = AerSimulator(
            method=method,
            device=device,
            max_parallel_threads=0,
            max_parallel_experiments=1,
        )

    def simulate(self, circuit, shots=1024):

        with measure_resources(self.name) as metrics:

            job = self.backend.run(circuit, shots=shots)
            result = job.result()

            counts = result.get_counts()

            metrics.update({
                "state_size": len(counts)
            })

            return {
                "counts": counts,
                "metrics": metrics
            }


# ============================================================
# FACTORY (USED BY unified_benchmark.py)
# ============================================================
def get_simulators():
    simulators = {}

    simulators["Statevector (CPU)"] = StatevectorSimulator()

    try:
        simulators["Aer (statevector, GPU)"] = AerSimulatorBackend(
            method="statevector",
            device="GPU"
        )
    except Exception as e:
        print(f"⚠️ GPU not available: {e}")

    return simulators
