"""
Fixed RDBMS model with adaptive threshold
"""

import numpy as np
import time


class RDBMS:
    def store(self, state):
        """
        Store state with adaptive threshold based on state characteristics
        """
        start = time.time()
        
        n_qubits = int(np.log2(len(state)))
        
        # Adaptive threshold: keep states that contain 99.9% of probability
        # Or use relative threshold instead of absolute
        probs = np.abs(state) ** 2
        sorted_indices = np.argsort(probs)[::-1]  # Descending
        
        # Keep states until we have 99.9% of probability mass
        cumsum = 0
        threshold_idx = 0
        for i, idx in enumerate(sorted_indices):
            cumsum += probs[idx]
            threshold_idx = i
            if cumsum >= 0.999:  # 99.9% probability mass
                break
        
        # Keep these indices
        idx = sorted_indices[:threshold_idx + 1]
        data = state[idx]
        
        # DB size = compressed representation
        db_size = len(idx) * 16  # complex128 approx
        
        end = time.time()
        
        return {
            "index": idx,
            "data": data,
            "size": db_size,
            "time": end - start,
            "n_stored": len(idx),
            "probability_mass": cumsum
        }


def reconstruct(state, k=None):
    """
    Reconstruct using top-k amplitudes or adaptive threshold
    """
    if k is None:
        # Adaptive: keep until 99% probability mass
        probs = np.abs(state) ** 2
        sorted_indices = np.argsort(probs)[::-1]
        
        cumsum = 0
        threshold_idx = 0
        for i, idx in enumerate(sorted_indices):
            cumsum += probs[idx]
            threshold_idx = i
            if cumsum >= 0.99:
                break
        k = threshold_idx + 1
    
    idx = np.argsort(np.abs(state))[-k:]
    recon = np.zeros_like(state, dtype=complex)
    recon[idx] = state[idx]
    
    # Renormalize
    norm = np.linalg.norm(recon)
    if norm > 0:
        recon = recon / norm
    
    return recon


def fidelity(original, reconstructed):
    """
    Compute fidelity between original and reconstructed states
    """
    num = np.abs(np.vdot(original, reconstructed)) ** 2
    den = np.vdot(original, original) * np.vdot(reconstructed, reconstructed)
    if den == 0:
        return 0
    return float(num / den)
