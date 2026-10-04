# analyze_enhanced.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")


# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    df = pd.read_csv(path)
    df["simulator"] = df["simulator"].str.strip()

    df["backend"] = df["simulator"].apply(
        lambda x: "GPU" if "gpu" in x.lower() else "CPU"
    )

    return df


# ==============================
# TIME vs QUBITS
# ==============================
def plot_time_vs_qubits(df):
    df = df[df["success"] == True]

    fig, ax = plt.subplots(figsize=(10, 6))

    for circuit in df["circuit"].unique():
        data = df[df["circuit"] == circuit]
        means = data.groupby("n_qubits")["time_s"].mean()

        ax.plot(means.index, means.values, marker="o", label=circuit.upper())

    ax.set_title("Execution Time vs Number of Qubits")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Time (seconds)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/time_vs_qubits.png", dpi=200)
    plt.show()


# ==============================
# RESOURCE vs QUBITS
# ==============================
def plot_resources_vs_qubits(df):
    df = df[df["success"] == True]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---------------- CPU MEMORY ----------------
    cpu = df[df["backend"] == "CPU"]
    for c in cpu["circuit"].unique():
        d = cpu[cpu["circuit"] == c]
        m = d.groupby("n_qubits")["cpu_memory_mb"].mean()
        axes[0].plot(m.index, m.values, marker="o", label=c.upper())

    axes[0].set_title("CPU Memory vs Qubits")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("MB")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------- GPU MEMORY (THEORETICAL) ----------------
    gpu = df[df["backend"] == "GPU"]
    for c in gpu["circuit"].unique():
        d = gpu[gpu["circuit"] == c]
        m = d.groupby("n_qubits")["gpu_memory_estimated_mb"].mean()
        axes[1].plot(m.index, m.values, marker="s", label=c.upper())

    axes[1].set_title("GPU Memory (Theoretical)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("MB")
    axes[1].legend()
    axes[1].grid(True)

    # ---------------- TIME ----------------
    for c in df["circuit"].unique():
        d = df[df["circuit"] == c]
        m = d.groupby("n_qubits")["time_s"].mean()
        axes[2].plot(m.index, m.values, marker="o", label=c.upper())

    axes[2].set_title("Time vs Qubits")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Qubits")
    axes[2].set_ylabel("Seconds")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("results/resource_time_scaling.png", dpi=200)
    plt.show()


# ==============================
# SPEEDUP + CROSSOVER DETECTION
# ==============================
def plot_speedup_and_crossover(df):
    df = df[df["success"] == True]

    pivot = df.pivot_table(
        index=["circuit", "n_qubits"],
        columns="backend",
        values="time_s",
        aggfunc="mean"
    ).dropna()

    pivot["speedup"] = pivot["CPU"] / pivot["GPU"]

    fig, ax = plt.subplots(figsize=(10, 6))

    crossover_points = []

    for circuit in pivot.index.get_level_values(0).unique():
        data = pivot.loc[circuit]

        ax.plot(data.index, data["speedup"], marker="o", label=circuit.upper())

        # crossover detection (first GPU advantage point)
        cross = data[data["speedup"] > 1]
        if not cross.empty:
            q = cross.index[0]
            crossover_points.append((circuit, q))
            ax.scatter(q, 1, s=120, marker="x")

    ax.axhline(1, linestyle="--", color="red", label="Break-even")

    ax.set_title("GPU Speedup vs Qubits (Crossover Detection)")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    # annotations
    for c, q in crossover_points:
        ax.annotate(
            f"{c.upper()} crossover",
            xy=(q, 1),
            xytext=(q + 1, 2),
            arrowprops=dict(arrowstyle="->")
        )

    plt.tight_layout()
    plt.savefig("results/speedup_crossover.png", dpi=200)
    plt.show()


# ==============================
# FIGURE 3 (PUBLICATION-READY)
# ==============================
def plot_figure3(df):
    df = df[df["success"] == True]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------------- LEFT: TIME ----------------
    for c in df["circuit"].unique():
        d = df[df["circuit"] == c]
        m = d.groupby("n_qubits")["time_s"].mean()
        axes[0].plot(m.index, m.values, marker="o", label=c.upper())

    axes[0].set_title("A. Execution Time Scaling")
    axes[0].set_xlabel("Number of Qubits")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    axes[0].legend()

    # ---------------- RIGHT: MEMORY ----------------
    cpu = df[df["backend"] == "CPU"]
    for c in cpu["circuit"].unique():
        d = cpu[cpu["circuit"] == c]
        m = d.groupby("n_qubits")["cpu_memory_mb"].mean()
        axes[1].plot(m.index, m.values, marker="o", label=f"{c.upper()} CPU")

    gpu = df[df["backend"] == "GPU"]
    for c in gpu["circuit"].unique():
        d = gpu[gpu["circuit"] == c]
        m = d.groupby("n_qubits")["gpu_memory_estimated_mb"].mean()
        axes[1].plot(m.index, m.values, marker="s", linestyle="--", label=f"{c.upper()} GPU (theory)")

    axes[1].set_title("B. Memory Scaling")
    axes[1].set_xlabel("Number of Qubits")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_yscale("log")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle("Figure 3: Quantum Simulation Scaling Laws", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figure3_paper_ready.png", dpi=300)
    plt.show()


# ==============================
# MAIN
# ==============================
def main():
    df = load_data("results/benchmark.csv")

    plot_time_vs_qubits(df)
    plot_resources_vs_qubits(df)
    plot_speedup_and_crossover(df)
    plot_figure3(df)


if __name__ == "__main__":
    main()
