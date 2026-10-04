"""
Updated benchmark with fixed RDBMS model
"""

import csv
from circuits import get_circuit
from simulator import QuantumSimulator
from db_model import RDBMS, reconstruct, fidelity  # Use fixed version

print("=" * 80)
print("MULTI-CIRCUIT QISKIT BENCHMARK (FIXED RDBMS)")
print("=" * 80)

sim = QuantumSimulator()
db = RDBMS()

circuits = ["GHZ", "BV", "QFT", "GROVER", "QAOA"]
qubits_list = [10, 12, 14, 16, 18, 20, 22]

rows = []

for c in circuits:
    for n in qubits_list:
        print(f"\n[{c}] QUBITS: {n}")
        
        qc = get_circuit(c, n)

        cpu_state, cpu_t = sim.cpu_run(qc)
        gpu_state, gpu_t = sim.gpu_run(qc)

        stored = db.store(cpu_state)
        recon = reconstruct(cpu_state)  # Adaptive k
        fid = fidelity(cpu_state, recon)

        print(f"  CPU: {cpu_t:.6f}s | GPU: {gpu_t:.6f}s | RDBMS: {stored['time']:.6f}s")
        print(f"  DB size: {stored['size']} bytes | Stored states: {stored['n_stored']} | Fidelity: {fid:.6f}")
        print(f"  Probability mass kept: {stored['probability_mass']:.4f}")

        rows.append([
            c, n, cpu_t, gpu_t, stored["time"], 
            stored["size"], stored["n_stored"], stored["probability_mass"], fid
        ])

# Save CSV
with open("benchmark_data_fixed.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "circuit", "qubits", "cpu_time", "gpu_time", "rdbms_time",
        "db_size_bytes", "n_states_stored", "prob_mass_kept", "fidelity"
    ])
    writer.writerows(rows)

print("\n✔ Fixed CSV saved: benchmark_data_fixed.csv")
