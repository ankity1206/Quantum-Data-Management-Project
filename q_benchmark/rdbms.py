import numpy as np
import time


class RDBMS:

    def encode(self, state):
        """
        Simulated sparse storage:
        stores only significant amplitudes
        """
        return np.count_nonzero(np.abs(state) > 1e-6) * 16


    def reconstruct(self, state, k=4):
        """
        Keep top-k amplitudes instead of only 1.
        This fixes fidelity collapse for QFT/QAOA.
        """
        idx = np.argsort(np.abs(state))[-k:]
        recon = np.zeros_like(state)
        recon[idx] = state[idx]/np.linalg.norm(state[idx])
        return recon


    def decode(self, state):
        start = time.time()

        recon = self.reconstruct(state, k=4)

        return recon, time.time() - start, self.encode(state)
