import csv
from circuits import get_circuit
from simulator import QuantumSimulator
from db_model import RDBMS, reconstruct, fidelity

print("=" * 80)
print("MULTI-CIRCUIT QISKIT BENCHMARK (CLEAN RESEARCH SYSTEM)")
print("=" * 80)

sim = QuantumSimulator()
db = RDBMS()

circuits = ["GHZ", "BV", "QFT", "GROVER", "QAOA"]
qubits_list = [10, 12, 14, 16, 18, 20, 22]

rows = []

for c in circuits:
    for n in qubits_list:

        qc = get_circuit(c, n)

        cpu_state, cpu_t = sim.cpu_run(qc)
        gpu_state, gpu_t = sim.gpu_run(qc)

        stored = db.store(cpu_state)
        recon = reconstruct(cpu_state, k=4)
        fid = fidelity(cpu_state, recon)

        print(f"\n[{c}] QUBITS: {n}")
        print(f"CPU: {cpu_t:.6f}s | GPU: {gpu_t:.6f}s | RDBMS: {stored['time']:.6f}s")
        print(f"DB size: {stored['size']} bytes | Fidelity: {fid:.6f}")

        rows.append([
            c, n, cpu_t, gpu_t, stored["time"], stored["size"], fid
        ])

# -------------------------
# SAVE CSV (COLAB READY)
# -------------------------
with open("benchmark_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "circuit", "qubits",
        "cpu_time", "gpu_time",
        "rdbms_time", "db_size_bytes",
        "fidelity"
    ])
    writer.writerows(rows)

print("\n✔ CSV saved: benchmark_data.csv")
