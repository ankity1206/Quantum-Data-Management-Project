"""
G7 Tier 1 Benchmark Suite - Circuit Generators (With Saving Support)
"""

import numpy as np
import random
import networkx as nx
import os

from qiskit import QuantumCircuit
from qiskit.qpy import dump as qpy_dump


# ============================================================
# SAVE UTILITY
# ============================================================

BASE_DIR = "saved_circuits"


def save_circuit(circuit: QuantumCircuit, name: str, folder: str):
    """
    Save circuit using QPY format (recommended for Qiskit)
    """
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{name}.qpy")

    with open(file_path, "wb") as f:
        qpy_dump(circuit, f)


# ============================================================
# GHZ CIRCUIT
# ============================================================

def create_ghz_circuit(n_qubits: int, save: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)

    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)

    qc.measure_all()

    if save:
        save_circuit(qc, f"ghz_{n_qubits}", os.path.join(BASE_DIR, "ghz"))

    return qc


# ============================================================
# QFT CIRCUIT
# ============================================================

def create_qft_circuit(n_qubits: int, save: bool = True) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            qc.cp(np.pi / (2 ** (j - i)), j, i)

    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - i - 1)

    qc.measure_all()

    if save:
        save_circuit(qc, f"qft_{n_qubits}", os.path.join(BASE_DIR, "qft"))

    return qc


# ============================================================
# RANDOM CIRCUIT
# ============================================================

def create_random_circuit(n_qubits: int, depth: int = 20, seed: int = 42, save: bool = True) -> QuantumCircuit:
    random.seed(seed)
    np.random.seed(seed)

    qc = QuantumCircuit(n_qubits)

    gates_1q = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    gates_2q = ['cx', 'cz', 'swap']

    for _ in range(depth):
        for q in range(n_qubits):
            gate = random.choice(gates_1q)

            if gate in ['rx', 'ry', 'rz']:
                theta = np.random.uniform(0, 2*np.pi)
                getattr(qc, gate)(theta, q)
            else:
                getattr(qc, gate)(q)

        qubits = list(range(n_qubits))
        random.shuffle(qubits)

        for i in range(0, n_qubits - 1, 2):
            q1, q2 = qubits[i], qubits[i + 1]
            gate = random.choice(gates_2q)

            if gate == 'swap':
                qc.swap(q1, q2)
            else:
                getattr(qc, gate)(q1, q2)

    qc.measure_all()

    if save:
        save_circuit(qc, f"random_{n_qubits}", os.path.join(BASE_DIR, "random"))

    return qc


# ============================================================
# QAOA CIRCUIT
# ============================================================

def create_qaoa_circuit(n_qubits: int, p: int = 2, seed: int = 42, save: bool = True) -> QuantumCircuit:
    np.random.seed(seed)
    random.seed(seed)

    qc = QuantumCircuit(n_qubits)

    graph = nx.random_regular_graph(3, n_qubits, seed=seed)

    gamma = np.random.uniform(0, np.pi, p)
    beta = np.random.uniform(0, np.pi, p)

    qc.h(range(n_qubits))

    for layer in range(p):
        for u, v in graph.edges():
            qc.rzz(2 * gamma[layer], u, v)

        for q in range(n_qubits):
            qc.rx(2 * beta[layer], q)

    qc.measure_all()

    if save:
        save_circuit(qc, f"qaoa_{n_qubits}_p{p}", os.path.join(BASE_DIR, "qaoa"))

    return qc


# ============================================================
# BERNSTEIN–VAZIRANI CIRCUIT
# ============================================================

def create_bernstein_vazirani_circuit(n_qubits: int, seed: int = 42, save: bool = True) -> QuantumCircuit:
    random.seed(seed)

    secret = ''.join(random.choice('01') for _ in range(n_qubits))

    qc = QuantumCircuit(n_qubits + 1, n_qubits)

    qc.x(n_qubits)
    qc.h(n_qubits)

    qc.h(range(n_qubits))

    for i, bit in enumerate(secret):
        if bit == '1':
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    if save:
        save_circuit(qc, f"bv_{n_qubits}", os.path.join(BASE_DIR, "bv"))

    return qc


# ============================================================
# METADATA
# ============================================================

CIRCUIT_INFO = {
    'ghz': {'name': 'GHZ', 'sparsity': 'Extreme sparse'},
    'qft': {'name': 'QFT', 'sparsity': 'Extreme dense'},
    'random': {'name': 'Random Circuit', 'sparsity': 'Dense'},
    'qaoa': {'name': 'QAOA', 'sparsity': 'Moderate'},
    'bv': {'name': 'Bernstein-Vazirani', 'sparsity': 'Sparse output'}
}

QUBIT_RANGES = {
    'ghz': [20, 23, 26],
    'qft': [10, 15, 20],
    'random': [12, 14, 16],
    'qaoa': [14, 16, 18],
    'bv': [20, 23, 25]
}


# ============================================================
# GENERATOR
# ============================================================

def get_circuits(save: bool = True):
    circuits = {}

    circuits["ghz"] = [create_ghz_circuit(n, save) for n in QUBIT_RANGES["ghz"]]
    circuits["qft"] = [create_qft_circuit(n, save) for n in QUBIT_RANGES["qft"]]
    circuits["random"] = [create_random_circuit(n, save=save) for n in QUBIT_RANGES["random"]]
    circuits["qaoa"] = [create_qaoa_circuit(n, save=save) for n in QUBIT_RANGES["qaoa"]]
    circuits["bv"] = [create_bernstein_vazirani_circuit(n, save=save) for n in QUBIT_RANGES["bv"]]

    return circuits
