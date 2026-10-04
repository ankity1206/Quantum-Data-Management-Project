# out_of_core_v3_fixed.py
"""
Optimized Out-of-Core Quantum Simulator - Fixed for SQLite thread safety
"""

import sqlite3
import numpy as np
import time
import psutil
import math
import os
import pickle
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import gc
import hashlib


@dataclass
class SimulationState:
    """Checkpoint state for resuming simulations"""
    circuit_name: str
    current_gate_index: int
    total_gates: int
    n_qubits: int
    timestamp: str
    gates_completed: List[str]
    memory_usage_mb: float
    db_size_mb: float
    n_states: int
    state_hash: str


class OutOfCoreSimulatorV3:
    """
    Optimized out-of-core quantum simulator with adaptive checkpoints
    (No async due to SQLite thread safety - but checkpoints are already fast)
    """
    
    def __init__(self, db_path: str = 'out_of_core.db', 
                 memory_limit_mb: int = 100,
                 base_checkpoint_interval: int = 10,
                 max_checkpoint_interval: int = 50):
        
        self.db_path = db_path
        self.memory_limit_mb = memory_limit_mb
        self.base_checkpoint_interval = base_checkpoint_interval
        self.max_checkpoint_interval = max_checkpoint_interval
        
        self.conn = None
        self.cursor = None
        self.n_qubits = 0
        self.checkpoints_dir = "simulation_checkpoints_v3"
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'gates_applied': 0,
            'checkpoints_saved': 0,
            'total_time': 0,
            'peak_memory_mb': 0,
            'compression_ratio': 0,
            'checkpoint_times_ms': []
        }
        
        # Adaptive interval
        self.current_interval = base_checkpoint_interval
        self.gate_times = []  # Track gate times for adaptive interval
        
    def connect(self, resume_from: str = None):
        """Connect to database, optionally resume from checkpoint"""
        if resume_from and os.path.exists(resume_from):
            print(f"  Resuming from checkpoint: {resume_from}")
            return self._resume_from_checkpoint(resume_from)
        
        # Fresh start
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Optimizations
        self.cursor.execute('PRAGMA journal_mode=WAL')
        self.cursor.execute('PRAGMA synchronous=NORMAL')
        self.cursor.execute('PRAGMA cache_size=-20000')
        self.cursor.execute('PRAGMA temp_store=MEMORY')
        self.cursor.execute('PRAGMA mmap_size=268435456')
        
        return self
    
    def _get_state_hash(self) -> str:
        """Get hash of current state for integrity checking"""
        self.cursor.execute("SELECT COUNT(*), SUM(probability) FROM state")
        count, total_prob = self.cursor.fetchone()
        total_prob = total_prob if total_prob else 0
        return hashlib.md5(f"{count}_{total_prob:.10f}".encode()).hexdigest()[:8]
    
    def _save_checkpoint(self, circuit_name: str, gate_index: int, 
                         total_gates: int, gates_completed: List[str]):
        """Save checkpoint (sync but optimized)"""
        start = time.time()
        
        # Get current stats
        n_states = self._get_n_states()
        db_size_mb = self._get_db_size_mb()
        
        state = SimulationState(
            circuit_name=circuit_name,
            current_gate_index=gate_index,
            total_gates=total_gates,
            n_qubits=self.n_qubits,
            timestamp=datetime.now().isoformat(),
            gates_completed=gates_completed.copy(),
            memory_usage_mb=self.stats['peak_memory_mb'],
            db_size_mb=db_size_mb,
            n_states=n_states,
            state_hash=self._get_state_hash()
        )
        
        # Save checkpoint (metadata only, not full DB)
        checkpoint_file = os.path.join(self.checkpoints_dir, 
                                       f"checkpoint_{gate_index}_{int(time.time())}.pkl")
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'gate_index': gate_index,
                'state': state,
                'stats': self.stats.copy()
            }, f)
        
        elapsed_ms = (time.time() - start) * 1000
        self.stats['checkpoint_times_ms'].append(elapsed_ms)
        self.stats['checkpoints_saved'] += 1
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
        
        return elapsed_ms
    
    def _update_adaptive_interval(self):
        """Dynamically adjust checkpoint interval based on gate speed"""
        if len(self.gate_times) >= 10:
            # Average gate time
            avg_gate_time = sum(self.gate_times[-10:]) / 10
            
            # If gates are fast, we can checkpoint more frequently
            if avg_gate_time < 0.0005:  # <0.5ms per gate
                new_interval = min(self.current_interval + 5, self.max_checkpoint_interval)
                if new_interval != self.current_interval:
                    self.current_interval = new_interval
            elif avg_gate_time > 0.002:  # >2ms per gate (slow)
                new_interval = max(self.current_interval - 3, self.base_checkpoint_interval)
                if new_interval != self.current_interval:
                    self.current_interval = new_interval
    
    def _clean_old_checkpoints(self):
        """Keep only last 3 checkpoints to save disk space"""
        checkpoints = sorted([f for f in os.listdir(self.checkpoints_dir) 
                             if f.startswith('checkpoint_')])
        for old in checkpoints[:-3]:
            os.remove(os.path.join(self.checkpoints_dir, old))
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Restore simulation from checkpoint (metadata only - DB not backed up)"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # For true resume, we need the DB backup
        # For now, return state info
        state = checkpoint['state']
        self.stats = checkpoint['stats']
        self.current_interval = self.base_checkpoint_interval
        
        print(f"  Resumed from gate {state.current_gate_index}/{state.total_gates}")
        print(f"  State had {state.n_states} amplitudes, {state.db_size_mb:.2f} MB")
        
        return state
    
    def _get_db_size_mb(self) -> float:
        self.cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        row = self.cursor.fetchone()
        return (row[0] if row and row[0] else 0) / (1024 * 1024)
    
    def _get_n_states(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM state")
        return self.cursor.fetchone()[0]
    
    def _can_fit_in_memory(self, n_qubits: int) -> bool:
        required_mb = (2 ** n_qubits * 16) / (1024 * 1024)
        return required_mb < self.memory_limit_mb
    
    def initialize_state(self, n_qubits: int):
        self.n_qubits = n_qubits
        
        self.cursor.execute('DROP TABLE IF EXISTS state')
        self.cursor.execute('''
            CREATE TABLE state (
                basis_index INTEGER PRIMARY KEY,
                amplitude_real REAL NOT NULL,
                amplitude_imag REAL NOT NULL,
                probability REAL GENERATED ALWAYS AS 
                    (amplitude_real*amplitude_real + amplitude_imag*amplitude_imag) STORED
            )
        ''')
        
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_prob ON state(probability DESC)')
        self.cursor.execute('INSERT INTO state VALUES (?, ?, ?)', (0, 1.0, 0.0))
        self.conn.commit()
        return self
    
    def _apply_hadamard(self, qubit: int):
        inv_sqrt2 = 1 / math.sqrt(2)
        
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        new_states = {}
        for basis_idx, real, imag in rows:
            amp = complex(real, imag)
            qubit_val = (basis_idx >> qubit) & 1
            
            if qubit_val == 0:
                basis0 = basis_idx
                amp0 = amp * inv_sqrt2
                new_states[basis0] = new_states.get(basis0, 0) + amp0
                
                basis1 = basis_idx | (1 << qubit)
                amp1 = amp * inv_sqrt2
                new_states[basis1] = new_states.get(basis1, 0) + amp1
            else:
                basis0 = basis_idx & ~(1 << qubit)
                amp0 = amp * inv_sqrt2
                new_states[basis0] = new_states.get(basis0, 0) + amp0
                
                basis1 = basis_idx
                amp1 = amp * -inv_sqrt2
                new_states[basis1] = new_states.get(basis1, 0) + amp1
        
        self._update_state(new_states)
    
    def _apply_cnot(self, control: int, target: int):
        self.cursor.execute('SELECT basis_index, amplitude_real, amplitude_imag FROM state')
        rows = self.cursor.fetchall()
        
        new_states = {}
        for basis_idx, real, imag in rows:
            amp = complex(real, imag)
            if (basis_idx >> control) & 1:
                new_idx = basis_idx ^ (1 << target)
                new_states[new_idx] = new_states.get(new_idx, 0) + amp
            else:
                new_states[basis_idx] = new_states.get(basis_idx, 0) + amp
        
        self._update_state(new_states)
    
    def _update_state(self, new_states: dict):
        self.cursor.execute('DELETE FROM state')
        
        threshold = 1e-10
        for basis_idx, amp in new_states.items():
            if abs(amp) > threshold:
                self.cursor.execute(
                    'INSERT INTO state VALUES (?, ?, ?)',
                    (basis_idx, amp.real, amp.imag)
                )
        
        self.conn.commit()
    
    def apply_gate(self, gate: str, qubits: List[int]) -> float:
        start_time = time.time()
        
        if gate == 'h':
            self._apply_hadamard(qubits[0])
        elif gate in ['cx', 'cnot']:
            self._apply_cnot(qubits[0], qubits[1])
        else:
            return 0
        
        gate_time = time.time() - start_time
        self.stats['gates_applied'] += 1
        self.gate_times.append(gate_time)
        
        return gate_time
    
    def simulate_circuit(self, circuit, circuit_name: str = "circuit", 
                         verbose: bool = True, save_checkpoints: bool = True):
        n_qubits = circuit.num_qubits
        dense_memory_mb = (2 ** n_qubits * 16) / (1024 * 1024)
        
        if dense_memory_mb < self.memory_limit_mb and n_qubits < 25:
            if verbose:
                print(f"  State fits in memory ({dense_memory_mb:.1f} MB < {self.memory_limit_mb} MB)")
            return self._simulate_in_memory(circuit)
        else:
            if verbose:
                print(f"  State too large for memory ({dense_memory_mb:.1f} MB > {self.memory_limit_mb} MB)")
                print(f"  Using out-of-core mode with adaptive checkpoints")
            return self._simulate_out_of_core(circuit, circuit_name, verbose, save_checkpoints)
    
    def _simulate_in_memory(self, circuit):
        from qiskit.quantum_info import Statevector
        start = time.time()
        state = Statevector.from_instruction(circuit)
        sim_time = time.time() - start
        return {
            'mode': 'in-memory',
            'time': sim_time,
            'n_qubits': circuit.num_qubits,
            'n_states_stored': 2 ** circuit.num_qubits,
            'compression_ratio': 1
        }
    
    def _simulate_out_of_core(self, circuit, circuit_name, verbose, save_checkpoints):
        start_time = time.time()
        
        self.initialize_state(circuit.num_qubits)
        
        # Extract gates
        gates = []
        for instruction in circuit.data:
            gate = instruction.operation.name
            if gate in ['measure', 'barrier']:
                continue
            
            qubits = []
            for q in instruction.qubits:
                if hasattr(q, 'index'):
                    qubits.append(q.index)
                elif hasattr(q, '_index'):
                    qubits.append(q._index)
                else:
                    qubits.append(q.register.index if hasattr(q, 'register') else 0)
            
            gates.append((gate, qubits))
        
        total_gates = len(gates)
        gates_completed = []
        
        if verbose:
            print(f"    Total gates: {total_gates}")
            print(f"    Adaptive checkpoint interval: {self.current_interval} (auto-adjusting)")
        
        for i, (gate, qubits) in enumerate(gates):
            gate_time = self.apply_gate(gate, qubits)
            gates_completed.append(f"{gate}{qubits}")
            
            # Update adaptive interval every 10 gates
            if i > 0 and i % 10 == 0:
                self._update_adaptive_interval()
            
            # Update peak memory
            current_mem = psutil.Process().memory_info().rss / 1024 / 1024
            self.stats['peak_memory_mb'] = max(self.stats['peak_memory_mb'], current_mem)
            
            # Adaptive checkpoint
            if save_checkpoints and (i + 1) % self.current_interval == 0:
                checkpoint_ms = self._save_checkpoint(circuit_name, i + 1, total_gates, gates_completed)
                if verbose:
                    print(f"    ✓ Checkpoint at gate {i+1} (interval: {self.current_interval}, {checkpoint_ms:.1f}ms)")
            
            # Progress indicator
            if verbose and (i % 20 == 0 or i == total_gates - 1):
                print(f"    Gate {i+1}/{total_gates}: {gate}{qubits}... {gate_time*1000:.2f}ms")
            
            if i % 100 == 0:
                gc.collect()
        
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time
        
        n_states = self._get_n_states()
        db_size = self._get_db_size_mb()
        compression = (2 ** circuit.num_qubits) / n_states if n_states > 0 else 0
        
        return {
            'mode': 'out-of-core',
            'time': total_time,
            'n_states_stored': n_states,
            'db_size_mb': db_size,
            'compression_ratio': compression,
            'n_qubits': circuit.num_qubits,
            'stats': self.stats
        }
    
    def get_statistics(self) -> dict:
        avg_checkpoint = sum(self.stats['checkpoint_times_ms']) / len(self.stats['checkpoint_times_ms']) if self.stats['checkpoint_times_ms'] else 0
        return {
            'gates_applied': self.stats['gates_applied'],
            'checkpoints_saved': self.stats['checkpoints_saved'],
            'total_time_s': self.stats['total_time'],
            'peak_memory_mb': self.stats['peak_memory_mb'],
            'db_size_mb': self._get_db_size_mb(),
            'n_states': self._get_n_states(),
            'compression_ratio': (2 ** self.n_qubits) / self._get_n_states() if self.n_qubits > 0 else 0,
            'avg_checkpoint_ms': avg_checkpoint,
            'final_interval': self.current_interval
        }
    
    def close(self):
        if self.conn:
            self.conn.close()


def benchmark_optimized():
    """Compare V2 vs V3 performance"""
    from qiskit import QuantumCircuit
    
    print("=" * 80)
    print("OPTIMIZED CHECKPOINT BENCHMARK: V2 vs V3")
    print("=" * 80)
    
    n_qubits = 26
    print(f"\nTesting GHZ with {n_qubits} qubits...")
    
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    
    # V2 (old, sync checkpoints)
    print("\n1. V2 (SYNC checkpoints, fixed interval):")
    from out_of_core_v2 import OutOfCoreSimulatorV2
    sim1 = OutOfCoreSimulatorV2(memory_limit_mb=50, checkpoint_interval=5)
    sim1.connect()
    start = time.time()
    result1 = sim1.simulate_circuit(qc, verbose=False, save_checkpoints=True)
    time1 = time.time() - start
    print(f"   Time: {time1:.3f}s")
    print(f"   Checkpoints: {sim1.stats['checkpoints_saved']}")
    sim1.close()
    
    # V3 (optimized, adaptive)
    print("\n2. V3 (ADAPTIVE checkpoints, optimized):")
    sim2 = OutOfCoreSimulatorV3(memory_limit_mb=50, base_checkpoint_interval=5, max_checkpoint_interval=20)
    sim2.connect()
    start = time.time()
    result2 = sim2.simulate_circuit(qc, verbose=False, save_checkpoints=True)
    time2 = time.time() - start
    print(f"   Time: {time2:.3f}s")
    print(f"   Checkpoints: {sim2.stats['checkpoints_saved']}")
    
    stats = sim2.get_statistics()
    print(f"   Avg checkpoint time: {stats['avg_checkpoint_ms']:.1f}ms")
    print(f"   Final adaptive interval: {stats['final_interval']}")
    sim2.close()
    
    improvement = (time1 - time2) / time1 * 100 if time1 > time2 else (time2 - time1) / time2 * 100
    winner = "V3" if time2 < time1 else "V2"
    print(f"\n📊 WINNER: {winner} ({abs(improvement):.1f}% faster)")


def demo_improvements():
    """Demo out-of-core simulation with adaptive checkpoints"""
    from qiskit import QuantumCircuit
    
    print("=" * 80)
    print("OPTIMIZED OUT-OF-CORE SIMULATOR DEMO")
    print("=" * 80)
    
    n_qubits = 28
    print(f"\nCreating GHZ circuit with {n_qubits} qubits...")
    print(f"  Dense state would require {2**n_qubits * 16 / (1024**3):.2f} GB of RAM")
    
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    
    print(f"  Total gates: {qc.size()}")
    
    sim = OutOfCoreSimulatorV3(memory_limit_mb=50, base_checkpoint_interval=5, max_checkpoint_interval=20)
    sim.connect()
    
    print("\nStarting out-of-core simulation with adaptive checkpoints...")
    result = sim.simulate_circuit(qc, circuit_name="ghz_28", verbose=True, save_checkpoints=True)
    
    print(f"\n📊 RESULTS:")
    print(f"   Mode: {result['mode']}")
    print(f"   Time: {result['time']:.3f}s")
    print(f"   States stored: {result['n_states_stored']:,}")
    print(f"   DB size: {result['db_size_mb']:.2f} MB")
    print(f"   Compression: {result['compression_ratio']:.0f}x")
    
    stats = sim.get_statistics()
    print(f"\n📊 STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    sim.close()
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   At {n_qubits} qubits, dense simulation would need {2**n_qubits * 16 / (1024**3):.2f} GB.")
    print(f"   Our RDBMS out-of-core simulator stored it in {result['db_size_mb']:.2f} MB.")
    print(f"   That's {result['compression_ratio']:.0f}x compression!")


if __name__ == "__main__":
    print("=" * 80)
    print("OPTIMIZED OUT-OF-CORE SIMULATOR V3 (Thread-Safe)")
    print("=" * 80)
    
    print("\nChoose demo:")
    print("  1. Demo out-of-core (28 qubits GHZ)")
    print("  2. Benchmark V2 vs V3 (overhead comparison)")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        demo_improvements()
    elif choice == '2':
        benchmark_optimized()
