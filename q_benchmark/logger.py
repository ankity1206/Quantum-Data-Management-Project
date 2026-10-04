import csv
import os


class Logger:

    def __init__(self, file="benchmark_data.csv"):
        self.file = file

        if not os.path.exists(file):
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "circuit",
                    "qubits",
                    "cpu_time",
                    "gpu_time",
                    "rdbms_time",
                    "db_size_bytes",
                    "fidelity"
                ])

    def log(self, row):
        with open(self.file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
