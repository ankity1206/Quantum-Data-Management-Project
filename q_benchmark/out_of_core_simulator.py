"""
Out-of-core quantum simulator using SQLite
Simulates circuits larger than RAM by storing state in database
FIXED: Compatible with latest Qiskit API
"""

import sqlite3
import numpy as np
import time
import psutil
import math
import os
from typing import Dict, Tuple, Optional, List


class OutOfCoreSimulator:
    """
    Quantum simulator that uses database for storage when state doesn't fit in RAM
    """
    
    def __init__(self, db_path: str = 'out_of_core.db', memory_limit_mb: int = 100):
        self.db_path = db_path
        self.memory_limit_mb = memory_limit_mb
        self.conn = None
        self.cursor = None
        self.n_qubits = 0
        
    def connect(self):
        """Connect to database with optimizations for large data"""
        # Remove old database file to start fresh
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Optimizations for large datasets
        self.cursor.execute('PRAGMA journal_mode=WAL')        # Write-Ahead Logging
        self.cursor.execute('PRAGMA synchronous=NORMAL')      # Faster writes
        self.cursor.execute('PRAGMA cache_size=-20000')       # 20MB cache
        self.cursor.execute('PRAGMA temp_store=MEMORY')       # Temp tables in RAM
        
        return self
    
    def close(self):
        if self.conn:
            self.conn.close()
        # Clean up database file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def _can_fit_in_memory(self, n_qubits: int) -> bool:
        """Check if state fits in available RAM"""
        required_mb = (2 ** n_qubits * 16) / (1024 * 1024)  # 16 bytes per complex
        return required_mb < self.memory_limit_mb
    
    def initialize_state(self, n_qubits: int):
        """
        Initialize quantum state in database
        Start with |0...0⟩ state
        """
        self.n_qubits = n_qubits
        
        # Drop existing tables
        self.cursor.execute('DROP TABLE IF EXISTS state')
        
        # Create state table
        self.cursor.execute('''
            CREATE TABLE state (
                basis_index INTEGER PRIMARY KEY,
                amplitude_real REAL NOT NULL,
                amplitude_imag REAL NOT NULL
            )
        ''')
        
        # Insert initial |0...0⟩ state (all zeros)
        self.cursor.execute(
            'INSERT INTO state (basis_index, amplitude_real, amplitude_imag) VALUES (?, ?, ?)',
            (0, 1.0, 0.0)
        )
        
        self.conn.commit()
        self._log_memory_usage()
        
        return self
    
    def _log_memory_usage(self):
        """Log current memory and database usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get database size
        self.cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = self.cursor.fetchone()
        db_size_mb = (row[0] if row and row[0] else 0) / 1024 / 1024
        
        self.cursor.execute("SELECT COUNT(*) FROM state")
        n_states = self.cursor.fetchone()[0]
        
        print(f"    Memory: {memory_mb:.1f} MB | DB: {db_size_mb:.1f} MB | States: {n_states:,}")
        
        return {'memory_mb': memory_mb, 'db_size_mb': db_size_mb, 'n_states': n_states}
    
    def _apply_hadamard(self, qubit: int):
        """
        Apply Hadamard gate using SQL operations
        H = 1/√2 [[1, 1], [1, -1]]
        """
        inv_sqrt2 = 1 / math.sqrt(2)
        
        # Get all current states
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        # Use dictionary to aggregate new states
        new_states = {}
        
        for basis_idx, real, imag in rows:
            amp = complex(real, imag)
            
            # Check qubit value at position 'qubit'
            qubit_value = (basis_idx >> qubit) & 1
            
            if qubit_value == 0:
                # Creates two new states: same and flipped
                # Case 1: qubit stays 0
                basis0 = basis_idx
                amp0 = amp * inv_sqrt2
                new_states[basis0] = new_states.get(basis0, 0) + amp0
                
                # Case 2: qubit flips to 1
                basis1 = basis_idx | (1 << qubit)
                amp1 = amp * inv_sqrt2
                new_states[basis1] = new_states.get(basis1, 0) + amp1
            else:
                # qubit_value == 1
                # Case 1: qubit flips to 0
                basis0 = basis_idx & ~(1 << qubit)
                amp0 = amp * inv_sqrt2
                new_states[basis0] = new_states.get(basis0, 0) + amp0
                
                # Case 2: qubit stays 1 (with negative sign)
                basis1 = basis_idx
                amp1 = amp * -inv_sqrt2
                new_states[basis1] = new_states.get(basis1, 0) + amp1
        
        # Clear and re-insert
        self.cursor.execute('DELETE FROM state')
        
        for basis_idx, amp in new_states.items():
            if abs(amp) > 1e-10:  # Prune near-zero
                self.cursor.execute(
                    'INSERT INTO state VALUES (?, ?, ?)',
                    (basis_idx, amp.real, amp.imag)
                )
        
        self.conn.commit()
    
    def _apply_cnot(self, control: int, target: int):
        """
        Apply CNOT gate
        CNOT: if control=1, flip target
        """
        # Get all current states
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        new_states = {}
        
        for basis_idx, real, imag in rows:
            amp = complex(real, imag)
            
            control_val = (basis_idx >> control) & 1
            
            if control_val == 1:
                # Flip target qubit
                new_idx = basis_idx ^ (1 << target)
                new_states[new_idx] = new_states.get(new_idx, 0) + amp
            else:
                new_states[basis_idx] = new_states.get(basis_idx, 0) + amp
        
        # Update database
        self.cursor.execute('DELETE FROM state')
        for basis_idx, amp in new_states.items():
            if abs(amp) > 1e-10:
                self.cursor.execute(
                    'INSERT INTO state VALUES (?, ?, ?)',
                    (basis_idx, amp.real, amp.imag)
                )
        
        self.conn.commit()
    
    def _apply_x(self, qubit: int):
        """Apply Pauli X (NOT) gate"""
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        new_states = {}
        for basis_idx, real, imag in rows:
            amp = complex(real, imag)
            new_idx = basis_idx ^ (1 << qubit)
            new_states[new_idx] = new_states.get(new_idx, 0) + amp
        
        self.cursor.execute('DELETE FROM state')
        for basis_idx, amp in new_states.items():
            if abs(amp) > 1e-10:
                self.cursor.execute(
                    'INSERT INTO state VALUES (?, ?, ?)',
                    (basis_idx, amp.real, amp.imag)
                )
        self.conn.commit()
    
    def _apply_z(self, qubit: int):
        """Apply Pauli Z gate (phase flip on |1⟩)"""
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        for basis_idx, real, imag in rows:
            qubit_val = (basis_idx >> qubit) & 1
            if qubit_val == 1:
                # Flip sign
                self.cursor.execute(
                    'UPDATE state SET amplitude_real = -amplitude_real, amplitude_imag = -amplitude_imag WHERE basis_index = ?',
                    (basis_idx,)
                )
        self.conn.commit()
    
    def apply_gate(self, gate: str, qubits: List[int]) -> float:
        """Apply a gate to the current state"""
        start_time = time.time()
        
        if gate == 'h':
            self._apply_hadamard(qubits[0])
        elif gate == 'cx':
            self._apply_cnot(qubits[0], qubits[1])
        elif gate == 'x':
            self._apply_x(qubits[0])
        elif gate == 'z':
            self._apply_z(qubits[0])
        else:
            # For unsupported gates, skip with warning
            print(f"    Warning: Gate {gate} not implemented, skipping")
            return 0
        
        gate_time = time.time() - start_time
        return gate_time
    
    def simulate_circuit(self, circuit, verbose: bool = True):
        """
        Simulate a full quantum circuit out-of-core
        """
        from qiskit import transpile
        
        n_qubits = circuit.num_qubits
        
        # Decide if in-memory or out-of-core
        if self._can_fit_in_memory(n_qubits):
            print(f"  State fits in memory ({self.memory_limit_mb} MB limit)")
            return self._simulate_in_memory(circuit)
        else:
            print(f"  State too large for memory, using out-of-core mode")
            return self._simulate_out_of_core(circuit, verbose)
    
    def _simulate_in_memory(self, circuit):
        """Fallback to in-memory statevector for small states"""
        from qiskit.quantum_info import Statevector
        start = time.time()
        state = Statevector.from_instruction(circuit)
        sim_time = time.time() - start
        return {'state': state, 'time': sim_time, 'mode': 'in-memory'}
    
    def _simulate_out_of_core(self, circuit, verbose):
        """Simulate using database storage"""
        start_time = time.time()
        
        # Initialize state in database
        self.initialize_state(circuit.num_qubits)
        
        # Apply gates - FIXED: Properly extract qubit indices
        gate_times = []
        
        # Get circuit instructions
        for instruction in circuit.data:
            gate = instruction.operation.name
            
            # Extract qubit indices - FIXED for Qiskit 1.0+
            # Qubit objects have 'index' attribute or we can use the register position
            qubits = []
            for q in instruction.qubits:
                if hasattr(q, 'index'):
                    qubits.append(q.index)
                elif hasattr(q, '_index'):
                    qubits.append(q._index)
                else:
                    # Fallback: try to get register index
                    qubits.append(q.register.index if hasattr(q, 'register') else 0)
            
            if verbose:
                print(f"    Applying {gate} on {qubits}...", end=" ", flush=True)
            
            gate_time = self.apply_gate(gate, qubits)
            gate_times.append(gate_time)
            
            if verbose:
                print(f"{gate_time:.4f}s")
        
        total_time = time.time() - start_time
        
        # Get final statistics
        self.cursor.execute("SELECT COUNT(*) FROM state")
        n_states = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = self.cursor.fetchone()
        db_size = row[0] if row and row[0] else 0
        
        # Calculate compression ratio
        total_possible = 2 ** circuit.num_qubits
        compression_ratio = total_possible / n_states if n_states > 0 else 0
        
        return {
            'time': total_time,
            'mode': 'out-of-core',
            'n_states_stored': n_states,
            'db_size_bytes': db_size,
            'gate_times': gate_times,
            'compression_ratio': compression_ratio,
            'n_qubits': circuit.num_qubits
        }
    
    def get_probabilities(self, top_k: Optional[int] = None) -> Dict[int, float]:
        """
        Retrieve probabilities from database
        """
        if top_k:
            query = """
                SELECT basis_index, (amplitude_real*amplitude_real + amplitude_imag*amplitude_imag) as prob 
                FROM state 
                ORDER BY prob DESC 
                LIMIT ?
            """
            self.cursor.execute(query, (top_k,))
        else:
            query = """
                SELECT basis_index, (amplitude_real*amplitude_real + amplitude_imag*amplitude_imag) as prob 
                FROM state
            """
            self.cursor.execute(query)
        
        return {row[0]: row[1] for row in self.cursor.fetchall()}
    
    def measure(self, shots: int = 1024) -> Dict[int, int]:
        """
        Sample measurement outcomes based on stored probabilities
        """
        probs = self.get_probabilities()
        
        if not probs:
            return {}
        
        basis_states = list(probs.keys())
        prob_values = list(probs.values())
        
        # Normalize
        prob_sum = sum(prob_values)
        if prob_sum > 0:
            prob_values = np.array(prob_values) / prob_sum
        
        # Sample
        samples = np.random.choice(len(basis_states), size=shots, p=prob_values)
        counts = {basis_states[i]: int(np.sum(samples == i)) for i in range(len(basis_states))}
        
        return counts


class MemoryMonitor:
    """Monitor memory usage during out-of-core simulation"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
    
    def snapshot(self, label: str):
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.snapshots.append((label, memory_mb))
        return memory_mb
    
    def report(self):
        print("\n📊 MEMORY USAGE TRACE:")
        for label, memory in self.snapshots:
            print(f"   {label}: {memory:.1f} MB")
        
        if len(self.snapshots) > 1:
            peak = max(self.snapshots, key=lambda x: x[1])
            print(f"\n   Peak memory: {peak[1]:.1f} MB at '{peak[0]}'")


def demo_out_of_core():
    """
    Demonstrate out-of-core simulation with a large circuit
    """
    from qiskit import QuantumCircuit
    
    def ghz(n):
        """Simple GHZ circuit generator"""
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(1, n):
            qc.cx(0, i)
        return qc
    
    print("=" * 80)
    print("OUT-OF-CORE QUANTUM SIMULATION DEMO")
    print("=" * 80)
    
    # Test with GHZ at 24 qubits (pushing memory limits)
    n_qubits = 24
    print(f"\nSimulating GHZ with {n_qubits} qubits...")
    dense_memory_mb = (2 ** n_qubits * 16) / (1024 * 1024)
    print(f"  Dense state would require {dense_memory_mb:.0f} MB")
    
    circuit = ghz(n_qubits)
    
    # Set memory limit to 200MB (forces out-of-core for n>=24)
    simulator = OutOfCoreSimulator(memory_limit_mb=200)
    simulator.connect()
    
    monitor = MemoryMonitor()
    monitor.snapshot("Before simulation")
    
    result = simulator.simulate_circuit(circuit, verbose=True)
    
    monitor.snapshot("After simulation")
    
    print(f"\n📊 SIMULATION RESULTS:")
    print(f"   Mode: {result['mode']}")
    print(f"   Time: {result['time']:.3f}s")
    print(f"   States stored: {result.get('n_states_stored', 'N/A'):,}")
    print(f"   DB size: {result.get('db_size_bytes', 0) / 1e6:.1f} MB")
    print(f"   Compression: {result.get('compression_ratio', 0):.0f}x")
    
    # Show top probabilities
    print("\n📊 TOP 5 MOST PROBABLE STATES:")
    probs = simulator.get_probabilities(top_k=5)
    for basis_idx, prob in probs.items():
        print(f"   State |{basis_idx:0{n_qubits}b}>: p={prob:.6f}")
    
    # Perform measurement
    print("\n📊 MEASUREMENT RESULTS (100 shots):")
    counts = simulator.measure(shots=100)
    for basis_idx, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   State |{basis_idx:0{n_qubits}b}>: {count} shots")
    
    monitor.report()
    simulator.close()
    
    return result


if __name__ == "__main__":
    demo_out_of_core()
