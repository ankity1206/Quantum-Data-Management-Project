"""
G7 Tier 1 Benchmark Suite - Circuit Generators
"""

import numpy as np
from qiskit import QuantumCircuit
import random

def create_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit


def create_qft_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Correct QFT implementation
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)

    for i in range(n_qubits):
        circuit.h(i)
        for j in range(i + 1, n_qubits):
            circuit.cp(np.pi / (2 ** (j - i)), j, i)

    # Reverse qubit order
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - i - 1)

    circuit.measure_all()
    return circuit


def create_random_circuit(n_qubits: int, depth: int = 20, seed: int = 42) -> QuantumCircuit:
    random.seed(seed)
    np.random.seed(seed)

    circuit = QuantumCircuit(n_qubits, n_qubits)

    gate_pool_1q = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz']
    gate_pool_2q = ['cx', 'cz', 'swap']

    for _ in range(depth):
        for qubit in range(n_qubits):
            gate = random.choice(gate_pool_1q)
            if gate in ['rx', 'ry', 'rz']:
                theta = np.random.uniform(0, 2*np.pi)
                getattr(circuit, gate)(theta, qubit)
            else:
                getattr(circuit, gate)(qubit)

        used = set()
        for _ in range(n_qubits // 2):
            available = [q for q in range(n_qubits) if q not in used]
            if len(available) < 2:
                break
            q1, q2 = random.sample(available, 2)
            used.update([q1, q2])

            gate = random.choice(gate_pool_2q)
            if gate == 'swap':
                circuit.swap(q1, q2)
            else:
                getattr(circuit, gate)(q1, q2)

    circuit.measure_all()
    return circuit


def create_qaoa_circuit(n_qubits: int, p: int = 2, seed: int = 42) -> QuantumCircuit:
    import networkx as nx

    np.random.seed(seed)
    random.seed(seed)

    G = nx.random_regular_graph(3, n_qubits, seed=seed)

    circuit = QuantumCircuit(n_qubits, n_qubits)

    gamma = [0.5] * p
    beta = [0.5] * p

    circuit.h(range(n_qubits))

    for layer in range(p):
        for u, v in G.edges():
            circuit.rzz(2 * gamma[layer], u, v)

        for q in range(n_qubits):
            circuit.rx(2 * beta[layer], q)

    circuit.measure_all()
    return circuit


def create_bernstein_vazirani_circuit(n_qubits: int, seed: int = 42) -> QuantumCircuit:
    random.seed(seed)
    secret = ''.join(random.choice('01') for _ in range(n_qubits))

    circuit = QuantumCircuit(n_qubits + 1, n_qubits)

    circuit.x(n_qubits)
    circuit.h(range(n_qubits + 1))

    for i, bit in enumerate(secret):
        if bit == '1':
            circuit.cx(i, n_qubits)

    circuit.h(range(n_qubits))
    circuit.measure(range(n_qubits), range(n_qubits))

    return circuit


CIRCUIT_INFO = {
    'ghz': {'name': 'GHZ', 'sparsity': 'Extreme sparse'},
    'qft': {'name': 'QFT', 'sparsity': 'Extreme dense'},
    'random': {'name': 'Random Circuit (depth=20)', 'sparsity': 'Dense'},
    'qaoa': {'name': 'QAOA (p=2)', 'sparsity': 'Moderate'},
    'bv': {'name': 'Bernstein-Vazirani', 'sparsity': 'Sparse output'}
}

QUBIT_RANGES = {
    'ghz': [20, 23, 26],
    'qft': [10, 15, 20],
    'random': [12, 14, 16],
    'qaoa': [14, 16, 18],
    'bv': [20, 23, 25]
}

def get_circuits():
    """
    Generate circuits based on defined ranges
    """

    circuits = {}

    circuits["ghz"] = [
        create_ghz_circuit(n) for n in QUBIT_RANGES["ghz"]
    ]

    circuits["qft"] = [
        create_qft_circuit(n) for n in QUBIT_RANGES["qft"]
    ]

    circuits["random"] = [
        create_random_circuit(n) for n in QUBIT_RANGES["random"]
    ]

    circuits["qaoa"] = [
        create_qaoa_circuit(n) for n in QUBIT_RANGES["qaoa"]
    ]

    circuits["bv"] = [
        create_bernstein_vazirani_circuit(n) for n in QUBIT_RANGES["bv"]
    ]

    return circuits
