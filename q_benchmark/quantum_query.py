"""
SQL-like query interface for quantum states
Enables rich queries on stored quantum data
FIXED: Creates test data if no database exists
"""

import sqlite3
import numpy as np
import os
import time
from typing import List, Tuple, Optional, Dict


class QuantumQueryEngine:
    """
    Provides SQL-like queries on quantum states stored in RDBMS
    """
    
    def __init__(self, db_path: str = 'quantum_state.db'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.state_table = 'state'
        
    def connect(self, state_table: str = 'state'):
        """Connect to database, create if doesn't exist"""
        self.state_table = state_table
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Check if state table exists, create demo data if not
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (state_table,))
        if not self.cursor.fetchone():
            print(f"  No table '{state_table}' found, creating demo data...")
            self._create_demo_state()
        
        return self
    
    def _create_demo_state(self, n_qubits: int = 20):
        """
        Create a demo GHZ state for testing queries
        GHZ state has only two basis states with equal probability
        """
        # Drop existing table if any
        self.cursor.execute(f'DROP TABLE IF EXISTS {self.state_table}')
        
        # Create state table
        self.cursor.execute(f'''
            CREATE TABLE {self.state_table} (
                basis_index INTEGER PRIMARY KEY,
                amplitude_real REAL NOT NULL,
                amplitude_imag REAL NOT NULL,
                probability REAL,
                basis_string TEXT
            )
        ''')
        
        # GHZ state: |000...0⟩ and |111...1⟩ with amplitude 1/√2
        inv_sqrt2 = 1 / np.sqrt(2)
        prob = 0.5
        
        # State 0: all zeros
        basis_string_0 = '0' * n_qubits
        self.cursor.execute(f'''
            INSERT INTO {self.state_table} 
            (basis_index, amplitude_real, amplitude_imag, probability, basis_string)
            VALUES (?, ?, ?, ?, ?)
        ''', (0, inv_sqrt2, 0.0, prob, basis_string_0))
        
        # State 2^n - 1: all ones
        basis_index_1 = (1 << n_qubits) - 1
        basis_string_1 = '1' * n_qubits
        self.cursor.execute(f'''
            INSERT INTO {self.state_table} 
            (basis_index, amplitude_real, amplitude_imag, probability, basis_string)
            VALUES (?, ?, ?, ?, ?)
        ''', (basis_index_1, inv_sqrt2, 0.0, prob, basis_string_1))
        
        self.conn.commit()
        print(f"  Created demo GHZ state with {n_qubits} qubits (2 basis states)")
    
    def load_from_statevector(self, state_vector: np.ndarray, circuit_name: str = "circuit", n_qubits: int = None):
        """
        Load a state vector into the database for querying
        
        Args:
            state_vector: numpy array of complex amplitudes
            circuit_name: name for metadata
            n_qubits: number of qubits (auto-detected if None)
        """
        if n_qubits is None:
            n_qubits = int(np.log2(len(state_vector)))
        
        # Drop existing table
        self.cursor.execute(f'DROP TABLE IF EXISTS {self.state_table}')
        
        # Create state table
        self.cursor.execute(f'''
            CREATE TABLE {self.state_table} (
                basis_index INTEGER PRIMARY KEY,
                amplitude_real REAL NOT NULL,
                amplitude_imag REAL NOT NULL,
                probability REAL,
                basis_string TEXT
            )
        ''')
        
        # Insert non-zero amplitudes
        inserted = 0
        for idx, amp in enumerate(state_vector):
            prob = abs(amp) ** 2
            if prob > 1e-10:  # Only store non-zero
                basis_string = format(idx, f'0{n_qubits}b')
                self.cursor.execute(f'''
                    INSERT INTO {self.state_table} 
                    (basis_index, amplitude_real, amplitude_imag, probability, basis_string)
                    VALUES (?, ?, ?, ?, ?)
                ''', (idx, amp.real, amp.imag, prob, basis_string))
                inserted += 1
        
        # Create index on probability for faster queries
        self.cursor.execute(f'CREATE INDEX idx_prob_{self.state_table} ON {self.state_table}(probability DESC)')
        
        self.conn.commit()
        print(f"  Loaded state with {inserted:,} non-zero amplitudes")
        
        return inserted
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    # ============ Core Query Methods ============
    
    def query_top_k(self, k: int = 10) -> Dict:
        """
        Get top k most probable basis states
        
        SELECT basis_index, probability 
        FROM state 
        ORDER BY probability DESC 
        LIMIT k
        """
        query = f"""
            SELECT basis_index, probability, basis_string
            FROM {self.state_table} 
            ORDER BY probability DESC 
            LIMIT ?
        """
        start = time.time()
        result = self.cursor.execute(query, (k,)).fetchall()
        query_time = time.time() - start
        
        return {
            'results': [(idx, prob, basis_str) for idx, prob, basis_str in result],
            'query_time': query_time,
            'query_type': f'TOP {k}'
        }
    
    def query_by_probability_threshold(self, threshold: float = 0.01) -> Dict:
        """
        Find all states with probability above threshold
        
        SELECT basis_index, probability 
        FROM state 
        WHERE probability > threshold 
        ORDER BY probability DESC
        """
        query = f"""
            SELECT basis_index, probability, basis_string
            FROM {self.state_table} 
            WHERE probability > ? 
            ORDER BY probability DESC
        """
        start = time.time()
        result = self.cursor.execute(query, (threshold,)).fetchall()
        query_time = time.time() - start
        
        return {
            'results': [(idx, prob, basis_str) for idx, prob, basis_str in result],
            'query_time': query_time,
            'query_type': f'PROBABILITY > {threshold}',
            'n_results': len(result)
        }
    
    def query_by_qubit_pattern(self, pattern: str) -> Dict:
        """
        Find states matching a qubit pattern
        Example: pattern "10*" means qubit0=1, qubit1=0, others any
        """
        # Convert pattern to SQL LIKE pattern
        # "10*" -> "10%"
        sql_pattern = pattern.replace('*', '%').replace('?', '_')
        
        query = f"""
            SELECT basis_index, probability, basis_string
            FROM {self.state_table} 
            WHERE basis_string LIKE ?
            ORDER BY probability DESC
        """
        
        start = time.time()
        result = self.cursor.execute(query, (sql_pattern,)).fetchall()
        query_time = time.time() - start
        
        return {
            'results': [(idx, prob, basis_str) for idx, prob, basis_str in result],
            'query_time': query_time,
            'query_type': f'PATTERN: {pattern}',
            'n_results': len(result)
        }
    
    def query_by_basis_range(self, start_idx: int, end_idx: int) -> Dict:
        """
        Find states with basis index in range [start_idx, end_idx]
        """
        query = f"""
            SELECT basis_index, probability, basis_string
            FROM {self.state_table} 
            WHERE basis_index BETWEEN ? AND ?
            ORDER BY basis_index
        """
        start = time.time()
        result = self.cursor.execute(query, (start_idx, end_idx)).fetchall()
        query_time = time.time() - start
        
        return {
            'results': [(idx, prob, basis_str) for idx, prob, basis_str in result],
            'query_time': query_time,
            'query_type': f'BASIS INDEX IN [{start_idx}, {end_idx}]',
            'n_results': len(result)
        }
    
    def query_statistics(self) -> dict:
        """
        Get statistical summary of stored state
        """
        query = f"""
            SELECT 
                COUNT(*) as n_states,
                MIN(probability) as min_prob,
                MAX(probability) as max_prob,
                AVG(probability) as mean_prob,
                SUM(probability) as total_prob
            FROM {self.state_table}
        """
        start = time.time()
        result = self.cursor.execute(query).fetchone()
        query_time = time.time() - start
        
        # Calculate Shannon entropy
        self.cursor.execute(f"SELECT probability FROM {self.state_table} WHERE probability > 0")
        probs = [row[0] for row in self.cursor.fetchall()]
        entropy = -sum(p * np.log2(p) for p in probs) if probs else 0
        
        return {
            'n_states': result[0],
            'min_probability': result[1],
            'max_probability': result[2],
            'mean_probability': result[3],
            'total_probability': result[4],
            'shannon_entropy': entropy,
            'query_time': query_time
        }
    
    def query_partial_trace(self, qubit_to_trace: int, n_qubits: int = None) -> Dict:
        """
        Compute partial trace over a specific qubit
        Returns probabilities for remaining qubits
        """
        if n_qubits is None:
            # Try to infer from basis_string length
            self.cursor.execute(f"SELECT basis_string FROM {self.state_table} LIMIT 1")
            row = self.cursor.fetchone()
            if row:
                n_qubits = len(row[0])
            else:
                n_qubits = 20
        
        # For each state, mask out the traced qubit and sum probabilities
        mask_keep = ~(1 << qubit_to_trace)
        
        # Since SQLite bit operations are tricky, we'll use basis_string approach
        # Group by basis_string without the traced qubit
        query = f"""
            SELECT 
                SUBSTR(basis_string, 1, {qubit_to_trace}) || SUBSTR(basis_string, {qubit_to_trace + 2}) as reduced_state,
                SUM(probability) as prob
            FROM {self.state_table}
            GROUP BY reduced_state
            ORDER BY prob DESC
        """
        
        start = time.time()
        results = self.cursor.execute(query).fetchall()
        query_time = time.time() - start
        
        return {
            'results': [(state, prob) for state, prob in results],
            'query_time': query_time,
            'n_outcomes': len(results),
            'traced_qubit': qubit_to_trace
        }
    
    def execute_custom_query(self, sql_query: str, params: tuple = ()) -> Dict:
        """
        Execute a custom SQL query on the state table
        """
        start = time.time()
        try:
            result = self.cursor.execute(sql_query, params).fetchall()
            query_time = time.time() - start
            return {
                'results': result,
                'query_time': query_time,
                'success': True
            }
        except Exception as e:
            return {
                'results': [],
                'query_time': time.time() - start,
                'success': False,
                'error': str(e)
            }


# ============ Demo Function ============

def demo_quantum_queries():
    """
    Demonstrate quantum query capabilities
    """
    print("=" * 80)
    print("QUANTUM QUERY INTERFACE DEMO")
    print("=" * 80)
    
    # Initialize query engine (creates demo data automatically)
    qe = QuantumQueryEngine()
    qe.connect()
    
    print("\n📊 DATABASE STATISTICS:")
    stats = qe.query_statistics()
    for key, value in stats.items():
        if key != 'query_time':
            print(f"   {key}: {value}")
    
    # Example queries
    print("\n1. TOP 5 MOST PROBABLE STATES:")
    result = qe.query_top_k(5)
    for idx, prob, basis_str in result['results']:
        print(f"   State |{basis_str}>: p={prob:.6f}")
    print(f"   Query time: {result['query_time']:.6f}s")
    
    print("\n2. STATES WITH PROBABILITY > 0.4:")
    result = qe.query_by_probability_threshold(0.4)
    for idx, prob, basis_str in result['results']:
        print(f"   State |{basis_str}>: p={prob:.6f}")
    print(f"   Found {result['n_results']} states in {result['query_time']:.6f}s")
    
    print("\n3. PATTERN MATCHING: States starting with '1'")
    result = qe.query_by_qubit_pattern("1*")
    for idx, prob, basis_str in result['results']:
        print(f"   State |{basis_str}>: p={prob:.6f}")
    print(f"   Found {result['n_results']} states in {result['query_time']:.6f}s")
    
    print("\n4. PARTIAL TRACE (trace out qubit 0):")
    result = qe.query_partial_trace(0)
    for state, prob in result['results'][:5]:
        print(f"   Reduced state |{state}>: p={prob:.6f}")
    print(f"   Query time: {result['query_time']:.6f}s")
    
    print("\n5. CUSTOM SQL QUERY:")
    custom_result = qe.execute_custom_query(
        "SELECT COUNT(*) as count, SUM(probability) as total_prob FROM state"
    )
    if custom_result['success']:
        count, total_prob = custom_result['results'][0]
        print(f"   Total states: {count}, Total probability: {total_prob:.6f}")
    print(f"   Query time: {custom_result['query_time']:.6f}s")
    
    qe.close()
    
    print("\n✅ Quantum Query Interface Demo Complete!")


def demo_with_real_state():
    """
    Alternative demo: load a real state from your benchmark
    """
    print("=" * 80)
    print("QUANTUM QUERY INTERFACE - WITH REAL STATE")
    print("=" * 80)
    
    # Try to load a state from your existing data
    try:
        # Create a simple GHZ state for 20 qubits
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        
        print("\nGenerating GHZ state with 20 qubits...")
        qc = QuantumCircuit(20)
        qc.h(0)
        for i in range(1, 20):
            qc.cx(0, i)
        state = Statevector.from_instruction(qc)
        
        qe = QuantumQueryEngine(db_path='real_state.db')
        qe.connect()
        qe.load_from_statevector(state.data, circuit_name="GHZ_20", n_qubits=20)
        
        print("\n📊 QUERYING REAL GHZ STATE:")
        result = qe.query_top_k(10)
        print("   Top 5 states:")
        for idx, prob, basis_str in result['results'][:5]:
            print(f"     State |{basis_str}>: p={prob:.6f}")
        
        qe.close()
        
    except Exception as e:
        print(f"  Could not generate real state: {e}")


if __name__ == "__main__":
    demo_quantum_queries()
    print("\n" + "=" * 80)
    demo_with_real_state()
