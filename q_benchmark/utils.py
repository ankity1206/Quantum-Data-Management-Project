import numpy as np

def to_numpy(state):
    """
    Converts Qiskit Statevector or ndarray → NumPy array safely
    Future-proof against Qiskit updates
    """
    try:
        return np.asarray(state.data)
    except Exception:
        return np.asarray(state)
