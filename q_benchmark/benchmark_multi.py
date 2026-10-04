"""
Multi-iteration benchmark with statistical analysis
Runs each configuration 10 times and reports mean, std, min, max
"""

import csv
import time
import numpy as np
from circuits import get_circuit
from simulator import QuantumSimulator
from db_model import RDBMS, reconstruct, fidelity

# Configuration
NUM_ITERATIONS = 10  # Number of runs per configuration
WARMUP_RUNS = 2      # Warmup runs before measurement (for GPU)

print("=" * 80)
print("MULTI-ITERATION QUANTUM BENCHMARK (10 runs per config)")
print("=" * 80)
print(f"Config: {NUM_ITERATIONS} iterations, {WARMUP_RUNS} warmup runs")
print("=" * 80)

sim = QuantumSimulator()
db = RDBMS()

circuits = ["GHZ", "BV", "QFT", "GROVER", "QAOA"]
qubits_list = [10, 12, 14, 16, 18, 20, 22]

# Store results with all iterations for later analysis
all_results = []

for circuit in circuits:
    for n_qubits in qubits_list:
        print(f"\n[{circuit}] QUBITS: {n_qubits}")
        
        qc = get_circuit(circuit, n_qubits)
        
        # --- Warmup runs (not measured) ---
        print(f"  Warming up ({WARMUP_RUNS} runs)...", end=" ", flush=True)
        for _ in range(WARMUP_RUNS):
            sim.cpu_run(qc)
            sim.gpu_run(qc)
            # Also warmup RDBMS
            cpu_state, _ = sim.cpu_run(qc)
            db.store(cpu_state)
        print("done")
        
        # --- Actual measurements ---
        cpu_times = []
        gpu_times = []
        rdbms_times = []
        db_sizes = []
        n_states_list = []
        fidelities = []
        prob_masses = []
        
        print(f"  Running {NUM_ITERATIONS} iterations...", end=" ", flush=True)
        
        for iteration in range(NUM_ITERATIONS):
            # CPU
            cpu_state, cpu_t = sim.cpu_run(qc)
            cpu_times.append(cpu_t)
            
            # GPU
            gpu_state, gpu_t = sim.gpu_run(qc)
            gpu_times.append(gpu_t)
            
            # RDBMS (store the CPU state)
            stored = db.store(cpu_state)
            rdbms_times.append(stored['time'])
            db_sizes.append(stored['size'])
            n_states_list.append(stored['n_stored'])
            prob_masses.append(stored['probability_mass'])
            
            # Fidelity (reconstruct from stored state)
            recon = reconstruct(cpu_state)  # Adaptive k
            fid = fidelity(cpu_state, recon)
            fidelities.append(fid)
            
            # Progress indicator
            if (iteration + 1) % 5 == 0:
                print(f"{iteration+1}", end=" ", flush=True)
        
        print("done")
        
        # --- Calculate statistics ---
        def stats(data):
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'p95': np.percentile(data, 95),
                'p99': np.percentile(data, 99)
            }
        
        cpu_stats = stats(cpu_times)
        gpu_stats = stats(gpu_times)
        rdbms_stats = stats(rdbms_times)
        
        # Store for this configuration
        config_result = {
            'circuit': circuit,
            'qubits': n_qubits,
            'iterations': NUM_ITERATIONS,
            'cpu_mean': cpu_stats['mean'],
            'cpu_std': cpu_stats['std'],
            'cpu_min': cpu_stats['min'],
            'cpu_max': cpu_stats['max'],
            'cpu_p95': cpu_stats['p95'],
            'gpu_mean': gpu_stats['mean'],
            'gpu_std': gpu_stats['std'],
            'gpu_min': gpu_stats['min'],
            'gpu_max': gpu_stats['max'],
            'gpu_p95': gpu_stats['p95'],
            'rdbms_mean': rdbms_stats['mean'],
            'rdbms_std': rdbms_stats['std'],
            'rdbms_min': rdbms_stats['min'],
            'rdbms_max': rdbms_stats['max'],
            'rdbms_p95': rdbms_stats['p95'],
            'db_size_bytes': np.mean(db_sizes),
            'db_size_std': np.std(db_sizes),
            'n_states_mean': np.mean(n_states_list),
            'n_states_std': np.std(n_states_list),
            'fidelity_mean': np.mean(fidelities),
            'fidelity_std': np.std(fidelities),
            'prob_mass_mean': np.mean(prob_masses),
            'all_cpu_times': cpu_times,
            'all_gpu_times': gpu_times,
            'all_rdbms_times': rdbms_times
        }
        
        all_results.append(config_result)
        
        # Print summary
        print(f"\n  📊 RESULTS (mean ± std, n={NUM_ITERATIONS}):")
        print(f"     CPU:   {cpu_stats['mean']:.6f}s ± {cpu_stats['std']:.6f}s  [min:{cpu_stats['min']:.6f}, max:{cpu_stats['max']:.6f}]")
        print(f"     GPU:   {gpu_stats['mean']:.6f}s ± {gpu_stats['std']:.6f}s  [min:{gpu_stats['min']:.6f}, max:{gpu_stats['max']:.6f}]")
        print(f"     RDBMS: {rdbms_stats['mean']:.6f}s ± {rdbms_stats['std']:.6f}s  [min:{rdbms_stats['min']:.6f}, max:{rdbms_stats['max']:.6f}]")
        print(f"     DB size: {np.mean(db_sizes):.0f} ± {np.std(db_sizes):.0f} bytes")
        print(f"     Fidelity: {np.mean(fidelities):.6f} ± {np.std(fidelities):.6f}")

# --- Save summary CSV ---
summary_rows = []
for r in all_results:
    summary_rows.append([
        r['circuit'], r['qubits'],
        r['cpu_mean'], r['cpu_std'], r['cpu_min'], r['cpu_max'], r['cpu_p95'],
        r['gpu_mean'], r['gpu_std'], r['gpu_min'], r['gpu_max'], r['gpu_p95'],
        r['rdbms_mean'], r['rdbms_std'], r['rdbms_min'], r['rdbms_max'], r['rdbms_p95'],
        r['db_size_bytes'], r['db_size_std'],
        r['n_states_mean'], r['n_states_std'],
        r['fidelity_mean'], r['fidelity_std'],
        r['prob_mass_mean']
    ])

with open("benchmark_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "circuit", "qubits",
        "cpu_mean", "cpu_std", "cpu_min", "cpu_max", "cpu_p95",
        "gpu_mean", "gpu_std", "gpu_min", "gpu_max", "gpu_p95",
        "rdbms_mean", "rdbms_std", "rdbms_min", "rdbms_max", "rdbms_p95",
        "db_size_bytes_mean", "db_size_bytes_std",
        "n_states_mean", "n_states_std",
        "fidelity_mean", "fidelity_std",
        "prob_mass_mean"
    ])
    writer.writerows(summary_rows)

# --- Save detailed iteration data ---
detailed_rows = []
for r in all_results:
    for i in range(NUM_ITERATIONS):
        detailed_rows.append([
            r['circuit'], r['qubits'], i,
            r['all_cpu_times'][i],
            r['all_gpu_times'][i],
            r['all_rdbms_times'][i]
        ])

with open("benchmark_detailed.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["circuit", "qubits", "iteration", "cpu_time", "gpu_time", "rdbms_time"])
    writer.writerows(detailed_rows)

print("\n" + "=" * 80)
print("✅ BENCHMARK COMPLETE!")
print(f"   Summary saved: benchmark_summary.csv")
print(f"   Detailed saved: benchmark_detailed.csv")
print("=" * 80)
