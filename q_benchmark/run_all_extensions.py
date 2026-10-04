"""
Run all three extensions and generate comprehensive results
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Import extensions
from quantum_query import QuantumQueryEngine
from indexing_strategies import IndexingBenchmark
from out_of_core_simulator import OutOfCoreSimulator


def setup_test_state():
    """Generate a test quantum state for demonstrations"""
    from circuits import ghz, qft
    
    print("Generating test states...")
    
    # GHZ state (sparse)
    qc_ghz = ghz(20)
    state_ghz = Statevector.from_instruction(qc_ghz)
    
    # QFT state (dense)
    qc_qft = qft(12)
    state_qft = Statevector.from_instruction(qc_qft)
    
    return {
        'ghz': state_ghz.data,
        'qft': state_qft.data,
        'n_qubits': {'ghz': 20, 'qft': 12}
    }


def run_extension2():
    """Run Quantum Query Interface demo"""
    print("\n" + "=" * 80)
    print("EXTENSION 2: QUANTUM QUERY INTERFACE")
    print("=" * 80)
    
    from quantum_query import demo_quantum_queries
    demo_quantum_queries()


def run_extension4():
    """Run Indexing Strategies benchmark"""
    print("\n" + "=" * 80)
    print("EXTENSION 4: INDEXING STRATEGIES")
    print("=" * 80)
    
    from indexing_strategies import demo_indexing
    demo_indexing()


def run_extension5():
    """Run Out-of-Core simulation demo"""
    print("\n" + "=" * 80)
    print("EXTENSION 5: OUT-OF-CORE SIMULATION")
    print("=" * 80)
    
    from out_of_core_simulator import demo_out_of_core
    demo_out_of_core()


def generate_comparison_table():
    """Generate comparison table of all approaches"""
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    table = """
    ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
    │ Feature             │ Query        │ Indexing     │ Out-of-Core  │
    │                     │ Interface    │ Strategies   │ Simulator    │
    ├─────────────────────┼──────────────┼──────────────┼──────────────┤
    │ SQL-based queries   │ ✓✓✓          │ ✓✓           │ ✓            │
    │ Performance opt.    │ ✓            │ ✓✓✓          │ ✓✓           │
    │ Memory efficient    │ ✓            │ ✓            │ ✓✓✓          │
    │ Handles >20 qubits  │ ✓✓           │ ✓            │ ✓✓✓          │
    │ Production ready    │ ✓✓✓          │ ✓✓✓          │ ✓✓           │
    │ Course project fit  │ ✓✓✓          │ ✓✓✓          │ ✓✓✓          │
    └─────────────────────┴──────────────┴──────────────┴──────────────┘
    """
    print(table)


def main():
    print("=" * 80)
    print("COMPLETE RDBMS EXTENSIONS DEMO")
    print("=" * 80)
    print("\nThis will run all three extensions sequentially.")
    print("Each extension demonstrates different database capabilities.\n")
    
    # Ask user which to run
    print("Options:")
    print("  1. Run all extensions (recommended)")
    print("  2. Extension 2 only (Query Interface)")
    print("  3. Extension 4 only (Indexing Strategies)")
    print("  4. Extension 5 only (Out-of-Core Simulator)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1' or choice == '':
        run_extension2()
        run_extension4()
        run_extension5()
        generate_comparison_table()
    elif choice == '2':
        run_extension2()
    elif choice == '3':
        run_extension4()
    elif choice == '4':
        run_extension5()
    else:
        print("Invalid choice")
    
    print("\n✅ Extensions complete!")


if __name__ == "__main__":
    main()
