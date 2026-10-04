# Quantum Data Management: RDBMS-Based Quantum Simulator

Thsi project explored using **sparse RDBMS representations** for quantum state simulation in the NISQ era. This simulator achieves **exponential memory compression** for sparse quantum circuits while providing **SQL query capabilities** for quantum data analysis.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Benchmark Circuit Suite](#-benchmark-circuit-suite)
- [Extensions](#-extensions)
- [Performance Comparison](#-performance-comparison)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

Classical simulation of quantum circuits faces an **exponential memory wall**: storing an $n$-qubit state vector requires $2^n \times 16$ bytes. For 30 qubits, this is 8 GB; for 32 qubits, 32 GB.

**Key Insight**: Many quantum circuits of interest are **sparse**. The GHZ state $(|00\ldots0\rangle + |11\ldots1\rangle)/\sqrt{2}$ has only **2 non-zero amplitudes** regardless of $n$.

This project implements a **Relational Database Management System (RDBMS)** approach that:
- Stores only non-zero amplitudes (sparse COO format)
- Represents quantum gates as **relational operations**
- Enables **SQL queries** on quantum states
- Supports **out-of-core simulation** beyond RAM limits

---

## ✨ Key Features

### 1. Sparse State Representation
- **COO format**: `(basis_index, amplitude_real, amplitude_imag)`
- **Exponential compression**: $2.1\times10^9\times$ at 32 qubits
- **Automatic pruning** of near-zero amplitudes

### 2. Multiple Simulation Backends
- **CPU** (Qiskit statevector)
- **GPU** (Qiskit-Aer with CUDA)
- **RDBMS** (SQLite sparse representation)

### 3. SQL Query Interface
- **Top-K probabilities**: Find most likely measurement outcomes
- **Threshold filtering**: `WHERE probability > 0.01`
- **Pattern matching**: Find states where specific qubits have values
- **Partial trace**: Compute reduced density matrices via `GROUP BY`
- **Statistical summaries**: Shannon entropy, total probability

### 4. Indexing Optimization
- **B-tree indexes** on probability for fast queries
- **119× speedup** for threshold queries
- **104× speedup** for Top-K queries

### 5. Out-of-Core Simulation
- **Adaptive checkpointing** (52% faster than baseline)
- **Resume capability** from checkpoints
- **Simulates 32+ qubits** on 16GB RAM
- **Force out-of-core** mode via memory limit

### 6. Comprehensive Benchmarking
- **5 circuit families**: GHZ, Bernstein-Vazirani, QFT, Grover, QAOA
- **10 iterations** per configuration for statistical validity
- **Full metrics**: time, memory, fidelity, compression

---

## 📊 Results

### Memory Compression (GHZ Circuit)

| Qubits | Dense Memory | RDBMS Storage | Compression |
|--------|--------------|---------------|-------------|
| 20 | 16 MB | 32 B | 524,288× |
| 24 | 256 MB | 32 B | 8,388,608× |
| 28 | 4 GB | 32 B | 134,217,728× |
| 32 | 32 GB | 32 B | **2,147,483,648×** |

### Performance Comparison (22 Qubits)

| Circuit | CPU (s) | GPU (s) | RDBMS (s) | Winner |
|---------|---------|---------|-----------|--------|
| GHZ | 0.195 ± 0.060 | **0.116 ± 0.011** | 0.842 ± 0.017 | GPU |
| BV | 0.136 ± 0.031 | **0.060 ± 0.011** | 0.288 ± 0.019 | GPU |
| QFT | 0.596 ± 0.254 | **0.411 ± 0.159** | 0.808 ± 0.012 | GPU |
| Grover | 0.394 ± 0.099 | 0.226 ± 0.058 | **0.203 ± 0.006** | **RDBMS** |
| QAOA | 0.196 ± 0.060 | **0.116 ± 0.011** | 0.842 ± 0.017 | GPU |

### Stress Test: Simulation Limits

| Method | Max Qubits | Time at 30 Qubits |
|--------|-----------|-------------------|
| CPU | 30 (1110s) | 18.5 minutes |
| GPU | 26 | OOM at 28 |
| **RDBMS Out-of-Core** | **32+** | **0.014 seconds** |

**Key Finding**: RDBMS out-of-core is **79,000× faster** than CPU at 30 qubits and succeeds where GPU fails.

### Indexing Performance

| Query Type | No Index | With Index | Speedup |
|------------|----------|------------|---------|
| TOP 10 | 0.834 ms | 0.008 ms | **104×** |
| THRESHOLD > 0.01 | 0.477 ms | 0.004 ms | **119×** |
| RANGE QUERY | 0.488 ms | 0.005 ms | **98×** |

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for GPU simulation)
- SQLite 3 (built into Python)

### Setup

```bash
# Clone the repository
git clone https://github.com/ankity1206/Quantum-Data-Management-Project.git
cd quantum-rdbms

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# For GPU support (optional)
pip install qiskit-aer-gpu
pip install GPUtil
```
