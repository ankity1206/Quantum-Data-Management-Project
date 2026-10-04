# benchmark.py

import csv


def run_simulation(simulator, circuit):
    """
    Runs simulation using simulator abstraction.
    Measurement is handled inside simulator (CPU + GPU via tracker).
    """

    try:
        result = simulator.simulate(circuit)
        metrics = result["metrics"]

        return {
            "time_s": metrics.get("time_s"),
            "cpu_memory_mb": metrics.get("cpu_memory_mb"),
            "gpu_memory_mb": metrics.get("gpu_memory_mb"),
            "success": True
        }

    except Exception as e:
        print(f"Simulation failed: {e}")
        return {
            "time_s": None,
            "cpu_memory_mb": None,
            "gpu_memory_mb": None,
            "success": False
        }


# =========================================================
# MAIN BENCHMARK FUNCTION
# =========================================================
def run_benchmark(circuits, simulators, output_csv="results/benchmark.csv"):
    """
    Runs full benchmark and saves results to CSV
    """

    fieldnames = [
        "circuit",
        "n_qubits",
        "simulator",
        "time_s",
        "cpu_memory_mb",
        "gpu_memory_mb",              # NVML (baseline / pooled)
        "gpu_memory_estimated_mb",    # THEORETICAL (correct scaling)
        "log2_memory",                # NEW: linearized memory scale
        "num_gates",
        "depth",
        "success"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for circuit_name, circuit_list in circuits.items():
            for circuit in circuit_list:

                # Actual qubits used by circuit
                actual_qubits = circuit.num_qubits

                # Gate stats
                num_gates = circuit.size()
                depth = circuit.depth()

                # 🔥 THEORETICAL GPU MEMORY (STATEVECTOR)
                # complex128 = 16 bytes
                gpu_estimated_mb = (2 ** actual_qubits) * 16 / (1024 ** 2)

                # 🔥 LOG SCALE REPRESENTATION (linearizes exponential growth)
                log2_memory = actual_qubits

                for sim_name, simulator in simulators.items():

                    print(f"\nRunning: {circuit_name} | {actual_qubits} qubits | {sim_name}")

                    result = run_simulation(simulator, circuit)

                    row = {
                        "circuit": circuit_name,
                        "n_qubits": actual_qubits,
                        "simulator": sim_name,
                        "time_s": result["time_s"],
                        "cpu_memory_mb": result["cpu_memory_mb"],
                        "gpu_memory_mb": result["gpu_memory_mb"],
                        "gpu_memory_estimated_mb": gpu_estimated_mb,
                        "log2_memory": log2_memory,
                        "num_gates": num_gates,
                        "depth": depth,
                        "success": result["success"]
                    }

                    writer.writerow(row)

    print("\n✅ Results saved to:", output_csv)
