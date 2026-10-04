"""
RDBMS Quantum Simulator (FINAL FIXED VERSION)
Compatible with unified benchmark
"""

import sqlite3
import numpy as np
import math
import time
import psutil
import tracemalloc


class RDBMSQuantumSimulator:
    def __init__(self):
        self.conn = None

    # ---------------- INIT DB ----------------
    def _init_db(self):
        self.conn = sqlite3.connect(":memory:")
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE state (
            basis TEXT PRIMARY KEY,
            real FLOAT,
            imag FLOAT
        )
        """)

        cur.execute("""
        CREATE TABLE gate_cache (
            gate TEXT,
            inp TEXT,
            outp TEXT,
            real FLOAT,
            imag FLOAT
        )
        """)

        self._load_gates()
        self.conn.commit()

    # ---------------- GATES ----------------
    def _load_gates(self):
        cur = self.conn.cursor()
        s = 1 / math.sqrt(2)

        gates = [
            ("h","0","0", s,0),
            ("h","0","1", s,0),
            ("h","1","0", s,0),
            ("h","1","1",-s,0),

            ("x","0","1",1,0),
            ("x","1","0",1,0),

            ("z","0","0",1,0),
            ("z","1","1",-1,0),
        ]

        cur.executemany("INSERT INTO gate_cache VALUES (?,?,?,?,?)", gates)

    # ---------------- INIT STATE ----------------
    def _init_state(self, n):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM state")
        cur.execute("INSERT INTO state VALUES (?,?,?)", ("0"*n, 1.0, 0.0))
        self.n = n
        self.conn.commit()

    # ---------------- QUBIT INDEX FIX ----------------
    def _get_qubits(self, circuit, inst):
        return [circuit.find_bit(q).index for q in inst.qubits]

    # ---------------- SINGLE GATE ----------------
    def _apply_single(self, gate, qubit):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM state").fetchall()

        new_state = {}

        for basis, r, i in rows:
            amp = complex(r, i)
            bit = basis[qubit]

            for outp, gr, gi in cur.execute(
                "SELECT outp, real, imag FROM gate_cache WHERE gate=? AND inp=?",
                (gate, bit)
            ).fetchall():

                nb = list(basis)
                nb[qubit] = outp
                nb = "".join(nb)

                new_state[nb] = new_state.get(nb, 0+0j) + amp * complex(gr, gi)

        cur.execute("DELETE FROM state")

        for k, v in new_state.items():
            if abs(v) > 1e-12:
                cur.execute("INSERT INTO state VALUES (?,?,?)", (k, v.real, v.imag))

        self.conn.commit()

    # ---------------- CNOT ----------------
    def _apply_cnot(self, c, t):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM state").fetchall()

        new_state = {}

        for basis, r, i in rows:
            amp = complex(r, i)
            b = list(basis)

            if b[c] == "1":
                b[t] = "1" if b[t] == "0" else "0"

            nb = "".join(b)
            new_state[nb] = new_state.get(nb, 0+0j) + amp

        cur.execute("DELETE FROM state")

        for k, v in new_state.items():
            if abs(v) > 1e-12:
                cur.execute("INSERT INTO state VALUES (?,?,?)", (k, v.real, v.imag))

        self.conn.commit()

    # ---------------- DISPATCH ----------------
    def _apply(self, gate, qubits):
        if gate in ["h", "x", "z"]:
            self._apply_single(gate, qubits[0])
        elif gate == "cx":
            self._apply_cnot(qubits[0], qubits[1])

    # ---------------- PROBS ----------------
    def probs(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM state").fetchall()
        return {b: r*r + i*i for b, r, i in rows}

    # ---------------- MAIN SIM ----------------
    def simulate(self, circuit, shots=1024):

        process = psutil.Process()
        tracemalloc.start()

        mem0 = process.memory_info().rss / 1024 / 1024
        t0 = time.perf_counter()

        self._init_db()
        self._init_state(circuit.num_qubits)

        for inst in circuit.data:
            gate = inst.operation.name
            qubits = self._get_qubits(circuit, inst)
            self._apply(gate, qubits)

        probs = self.probs()

        keys = list(probs.keys())
        vals = np.array(list(probs.values()))
        vals = vals / vals.sum()

        samples = np.random.choice(len(keys), shots, p=vals)
        counts = {keys[i]: int(np.sum(samples == i)) for i in range(len(keys))}

        t1 = time.perf_counter()

        mem1 = process.memory_info().rss / 1024 / 1024
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "counts": counts,
            "metrics": {
                "time_s": t1 - t0,
                "cpu_memory_mb": mem1 - mem0,
                "peak_memory_mb": peak / 1024 / 1024,
                "state_size": len(probs)
            }
        }

    # ---------------- FIXED CLOSE ----------------
    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
