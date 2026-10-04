#!/usr/bin/env python3

import sys
import traceback

from benchmark import run_benchmark
from circuits import *
from simulators import *


def test_single_case():
    """
    Minimal debug test
    """
    print("\n[DEBUG] Running single test case...\n")

    try:
        circuit = create_ghz_circuit(5)
        sim = StatevectorSimulator()

        result = sim.simulate(circuit)

        print("✅ Test successful")
        print("Time:", result['metrics']['time_s'])
        print("Memory:", result['metrics']['memory_peak_mb'])

    except Exception as e:
        print("❌ Error in single test:")
        traceback.print_exc()


def test_all_simulators():
    """
    Test all simulators on small circuit
    """
    print("\n[DEBUG] Testing all simulators...\n")

    circuit = create_ghz_circuit(5)

    simulators = [
        StatevectorSimulator(),
        AerSimulatorBackend(method='statevector', device='GPU'),
    ]

    for sim in simulators:
        print(f"Testing {sim.name}")

        try:
            result = sim.simulate(circuit)
            print("  ✅ OK")

        except Exception:
            print("  ❌ FAILED")
            traceback.print_exc()


def run_full():
    """
    Run full benchmark with error visibility
    """
    print("\n[RUN] Full Benchmark\n")

    try:
        run_benchmark()
        print("\n✅ Benchmark completed successfully")

    except Exception:
        print("\n❌ Benchmark crashed!")
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("MODULAR BENCHMARK DEBUGGER")
    print("=" * 80)

    print("\nChoose mode:")
    print("1 → Single test")
    print("2 → Test simulators")
    print("3 → Full benchmark")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        test_single_case()
    elif choice == "2":
        test_all_simulators()
    elif choice == "3":
        run_full()
    else:
        print("Invalid choice")
