import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os


def load_latest():
    files = glob("results/*/benchmark.csv")
    latest = max(files, key=os.path.getctime)
    return pd.read_csv(latest), os.path.dirname(latest)


def plot_time(df):
    df = df[df['success'] == True]

    for sim in df['simulator'].unique():
        data = df[df['simulator'] == sim]
        mean = data.groupby('n_qubits')['time_s'].mean()
        std = data.groupby('n_qubits')['time_s'].std().fillna(0)

        plt.errorbar(mean.index, mean.values, yerr=std.values, label=sim)

    plt.yscale('log')
    plt.legend()
    plt.title("Time vs Qubits")
    plt.show()


def plot_throughput(df):
    df = df[df['success'] == True]
    df['throughput'] = df['n_qubits'] / df['time_s']

    for sim in df['simulator'].unique():
        data = df[df['simulator'] == sim]
        mean = data.groupby('n_qubits')['throughput'].mean()
        plt.plot(mean.index, mean.values, marker='o', label=sim)

    plt.legend()
    plt.title("Throughput (Qubits/sec)")
    plt.show()


def main():
    df, _ = load_latest()
    plot_time(df)
    plot_throughput(df)


if __name__ == "__main__":
    main()
