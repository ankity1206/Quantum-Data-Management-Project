import numpy as np
from qiskit import QuantumCircuit

# -------------------------
# GHZ
# -------------------------
def ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    return qc


# -------------------------
# Bernstein–Vazirani (FIXED BUG)
# -------------------------
def bv(n):
    qc = QuantumCircuit(n)
    secret = np.random.randint(0, 2, n - 1)

    qc.x(n - 1)
    qc.h(range(n))

    for i, bit in enumerate(secret):
        if bit == 1:
            qc.cx(i, n - 1)

    qc.h(range(n - 1))
    return qc


# -------------------------
# QFT (inverse QFT style benchmark)
# -------------------------
def qft(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(np.pi / (2 ** (j - i)), i, j)
    return qc


# -------------------------
# GROVER (1 marked state)
# -------------------------
def grover(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))

    # simple oracle
    qc.x(n - 1)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(n - 1)

    qc.h(range(n))
    return qc


# -------------------------
# QAOA (toy layer)
# -------------------------
def qaoa(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))

    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rz(0.5, i + 1)
        qc.cx(i, i + 1)

    return qc


# -------------------------
# ROUTER
# -------------------------
def get_circuit(name, n):
    name = name.upper()
    if name == "GHZ":
        return ghz(n)
    if name == "BV":
        return bv(n)
    if name == "QFT":
        return qft(n)
    if name == "GROVER":
        return grover(n)
    if name == "QAOA":
        return qaoa(n)
    raise ValueError("Unknown circuit type")
