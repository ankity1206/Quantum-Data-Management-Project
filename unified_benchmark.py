"""
UNIFIED QISKIT (CPU vs GPU) vs RDBMS BENCHMARK
FIXED + CLEAN + CSV CONSISTENT + CIRCUIT REUSE
"""

import time
import csv
import os

from circuits import get_circuits
from simulators import get_simulators
from rdbms_simulator import RDBMSQuantumSimulator


# ============================================================
# CSV CONFIG
# ============================================================

CSV_FILE = "results/unified_benchmark.csv"

CSV_FIELDS = [
    "family",
    "n_qubits",
    "backend",
    "time_s",
    "memory_mb",
    "state_size",
    "compression_vs_rdbms",
    "gpu_speedup_vs_cpu"
]


def init_csv():
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def log_csv(row: dict):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# BACKEND HELPERS
# ============================================================

def get_backend(simulators, keyword):
    for name, backend in simulators.items():
        if keyword.lower() in name.lower():
            return backend
    return None


# ============================================================
# UNIFIED QISKIT RUNNER
# ============================================================

def run_qiskit(backend, circuit):
    res = backend.simulate(circuit, shots=2048)
    m = res["metrics"]

    return {
        "time": m["time_s"],
        "memory": m.get("cpu_memory_mb", 0),
        "state_size": m.get("state_size", len(res["counts"])),
        "counts": res["counts"]
    }


# ============================================================
# RDBMS RUNNER
# ============================================================

def run_rdbms(circuit):
    sim = RDBMSQuantumSimulator()
    res = sim.simulate(circuit, shots=2048)
    sim.close()
    return res


# ============================================================
# MAIN BENCHMARK
# ============================================================

def benchmark():
    print("=" * 110)
    print("UNIFIED QISKIT (CPU vs GPU) vs RDBMS + CSV LOGGING")
    print("=" * 110)

    init_csv()

    # ---------------- LOAD CIRCUITS ----------------
    circuits = get_circuits(save=False)

    print("\nLoaded circuits:")
    for k, v in circuits.items():
        print(f"  {k}: {len(v)} circuits")

    # ---------------- LOAD BACKENDS ----------------
    simulators = get_simulators()

    print("\nDetected Backends:")
    for name in simulators:
        print(f"  {name}")

    cpu_backend = get_backend(simulators, "statevector")
    gpu_backend = get_backend(simulators, "gpu")

    # ---------------- BENCHMARK FAMILIES ----------------
    families = ["ghz", "bv", "random"]

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for family, circuit_list in circuits.items():

        if family not in families:
            continue

        print("\n" + "=" * 110)
        print(f"CIRCUIT FAMILY: {family}")
        print("=" * 110)

        for circuit in circuit_list:

            n = circuit.num_qubits
            print(f"\n▶ Qubits: {n}")

            # ====================================================
            # CPU
            # ====================================================
            cpu = run_qiskit(cpu_backend, circuit)
            print(f"CPU   | time={cpu['time']:.4f}s | states={cpu['state_size']}")

            # ====================================================
            # GPU
            # ====================================================
            gpu = None
            gpu_speedup = None

            if gpu_backend:
                gpu = run_qiskit(gpu_backend, circuit)
                gpu_speedup = cpu["time"] / gpu["time"] if gpu["time"] > 0 else None
                print(f"GPU   | time={gpu['time']:.4f}s | states={gpu['state_size']}")
            else:
                print("GPU   | NOT AVAILABLE")

            # ====================================================
            # RDBMS
            # ====================================================
            rdbms = run_rdbms(circuit)
            r_time = rdbms["metrics"]["time_s"]
            r_states = rdbms["metrics"]["state_size"]

            print(f"RDBMS | time={r_time:.4f}s | states={r_states}")

            # ====================================================
            # METRICS
            # ====================================================
            compression = cpu["state_size"] / r_states if r_states > 0 else float("inf")

            print(f"Compression (CPU/RDBMS): {compression:.2f}x")
            if gpu_speedup:
                print(f"GPU Speedup vs CPU: {gpu_speedup:.2f}x")

            # ====================================================
            # CSV: CPU
            # ====================================================
            log_csv({
                "family": family,
                "n_qubits": n,
                "backend": "CPU",
                "time_s": cpu["time"],
                "memory_mb": cpu["memory"],
                "state_size": cpu["state_size"],
                "compression_vs_rdbms": compression,
                "gpu_speedup_vs_cpu": gpu_speedup if gpu_speedup else ""
            })

            # ====================================================
            # CSV: GPU
            # ====================================================
            if gpu:
                log_csv({
                    "family": family,
                    "n_qubits": n,
                    "backend": "GPU",
                    "time_s": gpu["time"],
                    "memory_mb": gpu["memory"],
                    "state_size": gpu["state_size"],
                    "compression_vs_rdbms": compression,
                    "gpu_speedup_vs_cpu": gpu_speedup if gpu_speedup else ""
                })

            # ====================================================
            # CSV: RDBMS
            # ====================================================
            log_csv({
                "family": family,
                "n_qubits": n,
                "backend": "RDBMS",
                "time_s": r_time,
                "memory_mb": rdbms["metrics"].get("cpu_memory_mb", 0),
                "state_size": r_states,
                "compression_vs_rdbms": compression,
                "gpu_speedup_vs_cpu": ""
            })


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    benchmark()
