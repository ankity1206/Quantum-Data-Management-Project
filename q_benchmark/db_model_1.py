import numpy as np
import time

class RDBMS:
    def store(self, state):
        start = time.time()

        # sparse storage simulation
        idx = np.where(np.abs(state) > 1e-3)[0]
        data = state[idx]

        # DB size = compressed representation
        db_size = len(idx) * 16  # complex128 approx

        end = time.time()

        return {
            "index": idx,
            "data": data,
            "size": db_size,
            "time": end - start
        }


def reconstruct(state, k=4):
    """
    Keep top-k amplitudes (FIXED + USED IN PAPER)
    """
    idx = np.argsort(np.abs(state))[-k:]
    recon = np.zeros_like(state)
    recon[idx] = state[idx]
    return recon


def fidelity(original, reconstructed):
    num = np.abs(np.vdot(original, reconstructed)) ** 2
    den = np.vdot(original, original) * np.vdot(reconstructed, reconstructed)
    if den == 0:
        return 0
    return float(num / den)
