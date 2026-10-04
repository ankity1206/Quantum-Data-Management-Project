"""
Database model for quantum state storage and reconstruction
Fixed: No ComplexWarning
"""

import numpy as np
import time


class RDBMS:
    def store(self, state):
        """
        Store quantum state in simulated RDBMS (sparse format)
        
        Args:
            state: numpy array of complex amplitudes
            
        Returns:
            dict with index, data, size, time, n_stored, probability_mass
        """
        start = time.time()
        
        # Get probabilities
        probs = np.abs(state) ** 2
        sorted_indices = np.argsort(probs)[::-1]  # Descending order
        
        # Keep states until we have 99.9% of probability mass
        cumsum = 0
        threshold_idx = 0
        for i, idx in enumerate(sorted_indices):
            cumsum += probs[idx]
            threshold_idx = i
            if cumsum >= 0.999:  # 99.9% probability mass
                break
        
        # Keep these indices
        kept_indices = sorted_indices[:threshold_idx + 1]
        kept_data = state[kept_indices]
        
        # DB size = compressed representation (complex128 = 16 bytes per value)
        db_size = len(kept_indices) * 16
        
        end = time.time()
        
        return {
            "index": kept_indices,
            "data": kept_data,
            "size": db_size,
            "time": end - start,
            "n_stored": len(kept_indices),
            "probability_mass": cumsum
        }


def reconstruct(state, k=None):
    """
    Reconstruct state using top-k amplitudes
    
    Args:
        state: Original state vector
        k: Number of amplitudes to keep (if None, uses adaptive threshold)
    
    Returns:
        Reconstructed state vector
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
    
    # Keep top k amplitudes
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
    
    Fidelity = |⟨ψ|φ⟩|² / (⟨ψ|ψ⟩·⟨φ|φ⟩)
    
    Args:
        original: Original state vector
        reconstructed: Reconstructed state vector
    
    Returns:
        Fidelity as float between 0 and 1
    """
    # Inner product
    overlap = np.vdot(original, reconstructed)
    
    # Squared magnitude
    num = np.abs(overlap) ** 2
    
    # Norms
    den = np.vdot(original, original) * np.vdot(reconstructed, reconstructed)
    
    if den == 0:
        return 0.0
    
    # Use np.real to extract real part (imaginary is always zero for fidelity)
    result = np.real(num / den)
    
    # Ensure it's a Python float
    return float(result)
