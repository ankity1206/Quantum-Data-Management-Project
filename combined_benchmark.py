"""
QISKIT vs RDBMS BENCHMARK (USING run_all.py CIRCUITS)
"""

import time
from qiskit.quantum_info import Statevector

from circuits import get_circuits
from rdbms_simulator import RDBMSQuantumSimulator


# ---------------- QISKIT ----------------
def run_qiskit(circuit):
    start = time.perf_counter()

    sv = Statevector.from_instruction(
        circuit.remove_final_measurements(inplace=False)
    )

    probs = sv.probabilities_dict()

    end = time.perf_counter()

    return {
        "time": end - start,
        "probs": probs,
        "states": len(probs)
    }


# ---------------- FIDELITY ----------------
def fidelity(p, q):
    f = 0.0
    for k in p:
        f += (p.get(k, 0) * q.get(k, 0)) ** 0.5
    return f


# ---------------- RDBMS ----------------
def run_rdbms(circuit):
    sim = RDBMSQuantumSimulator()
    res = sim.simulate(circuit, shots=2048)
    sim.close()
    return res


# ---------------- BENCHMARK ----------------
def benchmark():
    print("=" * 90)
    print("QISKIT vs RDBMS BENCHMARK (USING get_circuits)")
    print("=" * 90)

    # 🔥 LOAD FROM run_all.py PIPELINE
    circuits_dict = get_circuits(save=False)

    # choose subset for comparison
    selected = ["ghz", "bv"]

    for name in selected:
        print(f"\n📌 CIRCUIT TYPE: {name}")
        print("-" * 60)

        for circuit in circuits_dict[name]:
            print(f"\n▶ QUBITS: {circuit.num_qubits}")

            # Qiskit
            q = run_qiskit(circuit)
            print(f"QISKIT  | time={q['time']:.4f}s | states={q['states']}")

            # RDBMS
            r = run_rdbms(circuit)
            print(f"RDBMS   | time={r['time']:.4f}s | states={r['states']}")

            # compression
            print(f"Compression: {q['states'] / r['states']:.2f}x")

            # fidelity
            f = fidelity(q["probs"], r["probs"])
            print(f"Fidelity: {f:.6f}")


if __name__ == "__main__":
    benchmark()
