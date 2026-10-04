# run_all.py

from benchmark import run_benchmark
from circuits import get_circuits
from simulators import get_simulators


def run_full():
    print("=" * 80)
    print("G7 BENCHMARK RUNNER")
    print("=" * 80)

    # Load circuits
    circuits = get_circuits()

    # Load simulators
    simulators = get_simulators()

    print("\nLoaded Circuits:")
    for name, lst in circuits.items():
        print(f"  {name}: {len(lst)} circuits")

    print("\nLoaded Simulators:")
    for name in simulators.keys():
        print(f"  {name}")

    # Run benchmark
    run_benchmark(
        circuits=circuits,
        simulators=simulators,
        output_csv="results/benchmark.csv"
    )

    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    run_full()
