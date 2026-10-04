import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_data.csv")

print(df.head())

# -------------------------
# CPU vs GPU scaling
# -------------------------
for c in df["circuit"].unique():
    sub = df[df["circuit"] == c]

    plt.plot(sub["qubits"], sub["cpu_time"], label=f"{c} CPU")
    plt.plot(sub["qubits"], sub["gpu_time"], label=f"{c} GPU")

plt.xlabel("Qubits")
plt.ylabel("Time (s)")
plt.title("Quantum Simulation Scaling")
plt.legend()
plt.show()


# -------------------------
# RDBMS advantage plot
# -------------------------
for c in df["circuit"].unique():
    sub = df[df["circuit"] == c]
    plt.plot(sub["qubits"], sub["rdbms_time"], label=c)

plt.xlabel("Qubits")
plt.ylabel("RDBMS Time (s)")
plt.title("Sparse DB Advantage Scaling")
plt.legend()
plt.show()
