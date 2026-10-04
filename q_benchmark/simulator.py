import time
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator


class QuantumSimulator:
    def __init__(self):
        self.cpu = AerSimulator(method="statevector")
        self.gpu = AerSimulator(method="statevector")  # fallback-safe

    # -------------------------
    # SAFE RUN FUNCTION
    # -------------------------
    def _run(self, backend, qc):
        qc = qc.copy()
        qc.save_statevector()   # 🔥 CRITICAL FIX

        tqc = transpile(qc, backend)

        start = time.time()
        result = backend.run(tqc).result()
        end = time.time()

        # SAFE extraction
        state = result.get_statevector(tqc)

        return np.array(state), end - start

    def cpu_run(self, qc):
        return self._run(self.cpu, qc)

    def gpu_run(self, qc):
        return self._run(self.gpu, qc)
