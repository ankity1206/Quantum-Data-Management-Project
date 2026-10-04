"""
Simulation backends for G7 benchmark (FINAL CLEAN VERSION)
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import time
import psutil
import threading
from contextlib import contextmanager

# ==============================
# GPU MEMORY HELPER (NVML)
# ==============================
def get_gpu_memory_mb():
    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except:
        return None


# ==============================
# CONTINUOUS RESOURCE TRACKER
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
            cpu_mem = self.process.memory_info().rss / (1024 ** 2)
            self.cpu_peak = max(self.cpu_peak, cpu_mem)

            gpu_mem = get_gpu_memory_mb()
            if gpu_mem is not None:
                self.gpu_peak = max(self.gpu_peak, gpu_mem)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return self.cpu_peak, self.gpu_peak


# ==============================
# MEASURE RESOURCES
# ==============================
@contextmanager
def measure_resources(label=""):
    start_time = time.perf_counter()

    tracker = ResourceTracker(interval=0.01)
    tracker.start()

    metrics = {}

    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        cpu_peak, gpu_peak = tracker.stop()

        metrics.update({
            "label": label,
            "time_s": end_time - start_time,
            "cpu_memory_mb": cpu_peak,
            "gpu_memory_mb": gpu_peak,
        })


# ==============================
# CPU STATEVECTOR SIMULATOR
# ==============================
class StatevectorSimulator:
    def __init__(self):
        self.name = "Statevector (CPU)"

    def simulate(self, circuit, shots=1024):
        with measure_resources(self.name) as metrics:
            sv = Statevector.from_instruction(
                circuit.remove_final_measurements(inplace=False)
            )

            probs = sv.probabilities()
            samples = np.random.choice(len(probs), size=shots, p=probs)

            n = circuit.num_qubits
            bitstrings = [format(s, f'0{n}b') for s in samples]

            return {
                "counts": {bs: bitstrings.count(bs) for bs in set(bitstrings)},
                "statevector": sv.data,
                "metrics": metrics,
            }


# ==============================
# AER SIMULATOR (GPU / CPU)
# ==============================
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
            transpiled = transpile(circuit, self.backend)
            job = self.backend.run(transpiled, shots=shots)
            result = job.result()

            return {
                "counts": result.get_counts(),
                "metrics": metrics,
            }


# ==============================
# SIMULATOR FACTORY
# ==============================
def get_simulators():
    simulators = {}

    cpu_sim = StatevectorSimulator()
    simulators[cpu_sim.name] = cpu_sim

    try:
        gpu_sim = AerSimulatorBackend(method="statevector", device="GPU")
        simulators[gpu_sim.name] = gpu_sim
    except Exception as e:
        print(f"⚠️ GPU simulator not available: {e}")

    return simulators
