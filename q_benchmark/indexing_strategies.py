"""
Compare different database indexing strategies for quantum state queries
"""

import sqlite3
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple


class IndexingBenchmark:
    """
    Benchmark different indexing strategies on quantum state tables
    """
    
    def __init__(self, db_path: str = 'index_benchmark.db'):
        self.db_path = db_path
        self.results = {}
        
    def create_table_with_state(self, state_vector: np.ndarray, n_qubits: int):
        """
        Create table and populate with state data
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Base table without indexes
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_no_index (
                basis_index INTEGER PRIMARY KEY,
                amplitude_real REAL,
                amplitude_imag REAL,
                probability REAL,
                basis_string TEXT
            )
        ''')
        
        # Clear existing data
        self.cursor.execute('DELETE FROM state_no_index')
        
        # Insert data
        data = []
        for idx, amp in enumerate(state_vector):
            prob = abs(amp) ** 2
            if prob > 1e-10:  # Only store non-zero
                basis_string = format(idx, f'0{n_qubits}b')
                data.append((idx, amp.real, amp.imag, prob, basis_string))
        
        self.cursor.executemany(
            'INSERT INTO state_no_index VALUES (?, ?, ?, ?, ?)',
            data
        )
        self.conn.commit()
        
        # Create indexed versions
        self._create_indexed_tables()
        
        return len(data)
    
    def _create_indexed_tables(self):
        """Create copies of the table with different indexes"""
        
        # 1. B-tree on probability (default)
        self.cursor.execute('DROP TABLE IF EXISTS state_btree')
        self.cursor.execute('CREATE TABLE state_btree AS SELECT * FROM state_no_index')
        self.cursor.execute('CREATE INDEX idx_btree_prob ON state_btree(probability DESC)')
        
        # 2. Hash index simulation (using INTEGER PRIMARY KEY which is hash-like)
        self.cursor.execute('DROP TABLE IF EXISTS state_hash')
        self.cursor.execute('CREATE TABLE state_hash AS SELECT * FROM state_no_index')
        # PRIMARY KEY is already a hash index in SQLite
        
        # 3. Composite index on (probability, basis_index)
        self.cursor.execute('DROP TABLE IF EXISTS state_composite')
        self.cursor.execute('CREATE TABLE state_composite AS SELECT * FROM state_no_index')
        self.cursor.execute('CREATE INDEX idx_composite ON state_composite(probability, basis_index)')
        
        # 4. Covering index (includes all needed columns)
        self.cursor.execute('DROP TABLE IF EXISTS state_covering')
        self.cursor.execute('CREATE TABLE state_covering AS SELECT * FROM state_no_index')
        self.cursor.execute('''
            CREATE INDEX idx_covering ON state_covering(probability, basis_index, amplitude_real, amplitude_imag)
        ''')
        
        # 5. Partial index (only high probability states)
        self.cursor.execute('DROP TABLE IF EXISTS state_partial')
        self.cursor.execute('CREATE TABLE state_partial AS SELECT * FROM state_no_index')
        self.cursor.execute('''
            CREATE INDEX idx_partial ON state_partial(probability) 
            WHERE probability > 0.01
        ''')
        
        self.conn.commit()
    
    def benchmark_query(self, query_name: str, query_sql: str, params: Tuple = ()) -> Dict:
        """
        Run query on all table variants and measure performance
        """
        tables = ['state_no_index', 'state_btree', 'state_hash', 'state_composite', 'state_covering', 'state_partial']
        results = {}
        
        for table in tables:
            # Check if table exists
            self.cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not self.cursor.fetchone():
                continue
            
            # Warmup
            self.cursor.execute(query_sql.format(table=table), params)
            
            # Measure
            times = []
            for _ in range(5):
                start = time.perf_counter()
                result = self.cursor.execute(query_sql.format(table=table), params).fetchall()
                end = time.perf_counter()
                times.append(end - start)
            
            results[table] = {
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'n_rows': len(result)
            }
        
        return results
    
    def run_full_benchmark(self, state_vector: np.ndarray, n_qubits: int):
        """
        Run complete benchmark suite
        """
        print("Creating test data...")
        n_states = self.create_table_with_state(state_vector, n_qubits)
        print(f"  Stored {n_states} non-zero states")
        
        # Define benchmark queries
        queries = {
            'TOP 10': (
                "SELECT basis_index, probability FROM {table} ORDER BY probability DESC LIMIT 10",
                ()
            ),
            'THRESHOLD > 0.01': (
                "SELECT basis_index, probability FROM {table} WHERE probability > 0.01 ORDER BY probability DESC",
                ()
            ),
            'EXACT LOOKUP': (
                "SELECT * FROM {table} WHERE basis_index = ?",
                (42,)
            ),
            'RANGE QUERY': (
                "SELECT * FROM {table} WHERE probability BETWEEN 0.001 AND 0.01",
                ()
            ),
            'PATTERN MATCH': (
                "SELECT * FROM {table} WHERE basis_string LIKE '101%'",
                ()
            )
        }
        
        print("\n" + "=" * 80)
        print("INDEXING BENCHMARK RESULTS")
        print("=" * 80)
        
        all_results = {}
        
        for query_name, (query_sql, params) in queries.items():
            print(f"\n{query_name}:")
            print("-" * 50)
            
            results = self.benchmark_query(query_name, query_sql, params)
            all_results[query_name] = results
            
            # Find fastest
            fastest = min(results.items(), key=lambda x: x[1]['time_mean'])
            
            for table, metrics in results.items():
                print(f"  {table:20s}: {metrics['time_mean']*1000:.3f}ms ± {metrics['time_std']*1000:.3f}ms")
            
            print(f"  → Fastest: {fastest[0]} ({fastest[1]['time_mean']*1000:.3f}ms)")
        
        return all_results
    
    def generate_recommendations(self, results: Dict) -> str:
        """
        Generate indexing recommendations based on query patterns
        """
        recommendations = []
        
        # Analyze which index wins for each query type
        index_wins = {}
        for query, tables in results.items():
            fastest = min(tables.items(), key=lambda x: x[1]['time_mean'])
            index_wins[query] = fastest[0]
        
        recommendations.append("\n" + "=" * 80)
        recommendations.append("INDEXING RECOMMENDATIONS")
        recommendations.append("=" * 80)
        
        # Statistical summary
        from collections import Counter
        win_counts = Counter(index_wins.values())
        
        recommendations.append("\n📊 INDEX PERFORMANCE SUMMARY:")
        for idx, count in win_counts.most_common():
            percentage = (count / len(index_wins)) * 100
            recommendations.append(f"   {idx}: {percentage:.0f}% win rate")
        
        recommendations.append("\n💡 RECOMMENDATIONS:")
        
        if win_counts.get('state_btree', 0) > 0:
            recommendations.append("   • Use BTREE index for ordered queries (TOP K, range queries)")
        if win_counts.get('state_hash', 0) > 0:
            recommendations.append("   • Use HASH/PRIMARY KEY for exact lookups by basis_index")
        if win_counts.get('state_composite', 0) > 0:
            recommendations.append("   • Use COMPOSITE index for multi-predicate queries")
        if win_counts.get('state_covering', 0) > 0:
            recommendations.append("   • Use COVERING index when you need amplitude values, not just probabilities")
        if win_counts.get('state_partial', 0) > 0:
            recommendations.append("   • Use PARTIAL index when querying only high-probability states")
        
        recommendations.append("\n📈 FOR QUANTUM STATE QUERIES:")
        recommendations.append("   • Default: BTREE on probability (most queries are probability-ordered)")
        recommendations.append("   • Add PRIMARY KEY on basis_index for state reconstruction")
        recommendations.append("   • Consider partial index for threshold queries (WHERE probability > X)")
        
        return "\n".join(recommendations)
    
    def close(self):
        self.conn.close()


def demo_indexing():
    """
    Demo indexing with a sample quantum state (QFT at 14 qubits)
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    
    print("Generating test state (QFT, 14 qubits)...")
    qc = QuantumCircuit(14)
    for i in range(14):
        qc.h(i)
        for j in range(i + 1, 14):
            qc.cp(np.pi / (2 ** (j - i)), i, j)
    
    state = Statevector.from_instruction(qc)
    
    benchmark = IndexingBenchmark()
    results = benchmark.run_full_benchmark(state.data, 14)
    
    recommendations = benchmark.generate_recommendations(results)
    print(recommendations)
    
    benchmark.close()


if __name__ == "__main__":
    demo_indexing()
