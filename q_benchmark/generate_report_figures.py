# generate_report_figures_fixed.py
"""
Generate all figures for the LaTeX report - Compatible with your data format
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("=" * 60)
print("GENERATING REPORT FIGURES")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

df = pd.read_csv('benchmark_detailed.csv')

print("\nData loaded successfully!")
print(f"Columns: {df.columns.tolist()}")
print(f"Circuits: {df['circuit'].unique()}")
print(f"Qubits: {sorted(df['qubits'].unique())}")

# ============================================================================
# 2. FIGURE 1: Performance Comparison (CPU vs GPU vs RDBMS)
# ============================================================================

print("\n1. Generating performance comparison figure...")

fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

circuits = df['circuit'].unique()

for idx, circuit in enumerate(circuits):
    if idx >= 6:
        break
    ax = axes[idx]
    circuit_df = df[df['circuit'] == circuit]
    
    # CPU data
    cpu_means = circuit_df.groupby('qubits')['cpu_time'].mean()
    cpu_stds = circuit_df.groupby('qubits')['cpu_time'].std()
    
    # GPU data
    gpu_means = circuit_df.groupby('qubits')['gpu_time'].mean()
    gpu_stds = circuit_df.groupby('qubits')['gpu_time'].std()
    
    # RDBMS data
    rdbms_means = circuit_df.groupby('qubits')['rdbms_time'].mean()
    rdbms_stds = circuit_df.groupby('qubits')['rdbms_time'].std()
    
    ax.errorbar(cpu_means.index, cpu_means.values, yerr=cpu_stds.values,
               marker='o', markersize=6, capsize=3, linewidth=1.5,
               label='CPU', color='#1f77b4')
    
    ax.errorbar(gpu_means.index, gpu_means.values, yerr=gpu_stds.values,
               marker='s', markersize=6, capsize=3, linewidth=1.5,
               label='GPU', color='#ff7f0e')
    
    ax.errorbar(rdbms_means.index, rdbms_means.values, yerr=rdbms_stds.values,
               marker='^', markersize=6, capsize=3, linewidth=1.5,
               label='RDBMS', color='#2ca02c')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(circuit.upper())
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# Hide unused subplot
if len(circuits) < 6:
    for i in range(len(circuits), 6):
        axes[i].set_visible(False)

plt.suptitle('CPU vs GPU vs RDBMS: Simulation Time Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/performance_comparison.png")

# ============================================================================
# 3. FIGURE 2: Boxplot Distributions (Statistical Variance)
# ============================================================================

print("\n2. Generating boxplot distributions...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, circuit in enumerate(circuits):
    if idx >= 6:
        break
    ax = axes[idx]
    circuit_df = df[df['circuit'] == circuit]
    
    qubits = sorted(circuit_df['qubits'].unique())
    
    # Prepare data for boxplots
    cpu_data = [circuit_df[circuit_df['qubits'] == n]['cpu_time'].values for n in qubits]
    gpu_data = [circuit_df[circuit_df['qubits'] == n]['gpu_time'].values for n in qubits]
    rdbms_data = [circuit_df[circuit_df['qubits'] == n]['rdbms_time'].values for n in qubits]
    
    x = np.arange(len(qubits))
    width = 0.25
    
    bp_cpu = ax.boxplot(cpu_data, positions=x - width, widths=width, patch_artist=True,
                        boxprops=dict(facecolor='#1f77b4', alpha=0.7))
    bp_gpu = ax.boxplot(gpu_data, positions=x, widths=width, patch_artist=True,
                        boxprops=dict(facecolor='#ff7f0e', alpha=0.7))
    bp_rdbms = ax.boxplot(rdbms_data, positions=x + width, widths=width, patch_artist=True,
                          boxprops=dict(facecolor='#2ca02c', alpha=0.7))
    
    ax.set_xticks(x)
    ax.set_xticklabels(qubits)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(circuit.upper())
    ax.set_yscale('log')
    ax.legend([bp_cpu["boxes"][0], bp_gpu["boxes"][0], bp_rdbms["boxes"][0]], 
              ['CPU', 'GPU', 'RDBMS'], loc='upper left')
    ax.grid(True, alpha=0.3)

if len(circuits) < 6:
    for i in range(len(circuits), 6):
        axes[i].set_visible(False)

plt.suptitle('Time Distributions Across 10 Iterations', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/boxplot_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/boxplot_distributions.png")

# ============================================================================
# 4. FIGURE 3: GPU Speedup Over CPU
# ============================================================================

print("\n3. Generating GPU speedup figure...")

fig, ax = plt.subplots(figsize=(10, 6))

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    
    # Calculate speedup per qubit
    speedups = []
    qubit_list = []
    
    for n in circuit_df['qubits'].unique():
        n_df = circuit_df[circuit_df['qubits'] == n]
        cpu_mean = n_df['cpu_time'].mean()
        gpu_mean = n_df['gpu_time'].mean()
        if cpu_mean > 0 and gpu_mean > 0:
            speedups.append(cpu_mean / gpu_mean)
            qubit_list.append(n)
    
    ax.plot(qubit_list, speedups, 'o-', linewidth=2, markersize=8, label=circuit.upper())

ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even (CPU=GPU)')
ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Speedup (CPU Time / GPU Time)')
ax.set_title('GPU Speedup Over CPU (>1 means GPU is faster)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('figures/gpu_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/gpu_speedup.png")

# ============================================================================
# 5. FIGURE 4: Memory Compression (Stress Test Results)
# ============================================================================

print("\n4. Generating compression ratio figure...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data from your stress test
qubits = [20, 22, 24, 26, 28, 30, 32]
compression_ratios = [1, 1, 1, 33554432, 134217728, 536870912, 2147483648]

ax.semilogy(qubits, compression_ratios, 'o-', linewidth=2, markersize=10, color='green')
ax.set_xlabel('Number of Qubits')
ax.set_ylabel('Compression Ratio (log scale)')
ax.set_title('RDBMS Memory Compression vs Dense State Vector (GHZ Circuit)')
ax.grid(True, alpha=0.3)

# Add annotations
for q, c in zip(qubits, compression_ratios):
    if c > 1:
        ax.annotate(f'{c/1e6:.0f}M×', xy=(q, c), xytext=(q+0.5, c*2),
                   fontsize=8, ha='center')

ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Dense baseline (no compression)')
ax.legend()

plt.tight_layout()
plt.savefig('figures/compression_ratio.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/compression_ratio.png")

# ============================================================================
# 6. FIGURE 5: Coefficient of Variation (Stability Analysis)
# ============================================================================

print("\n5. Generating stability analysis figure...")

fig, ax = plt.subplots(figsize=(10, 6))

circuits_list = []
cv_cpu = []
cv_gpu = []
cv_rdbms = []

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    
    # Calculate CV across all qubits for this circuit
    cpu_cv = (circuit_df['cpu_time'].std() / circuit_df['cpu_time'].mean()) * 100
    gpu_cv = (circuit_df['gpu_time'].std() / circuit_df['gpu_time'].mean()) * 100
    rdbms_cv = (circuit_df['rdbms_time'].std() / circuit_df['rdbms_time'].mean()) * 100
    
    circuits_list.append(circuit.upper())
    cv_cpu.append(cpu_cv)
    cv_gpu.append(gpu_cv)
    cv_rdbms.append(rdbms_cv)

x = np.arange(len(circuits_list))
width = 0.25

ax.bar(x - width, cv_cpu, width, label='CPU', color='#1f77b4')
ax.bar(x, cv_gpu, width, label='GPU', color='#ff7f0e')
ax.bar(x + width, cv_rdbms, width, label='RDBMS', color='#2ca02c')

ax.set_xlabel('Circuit')
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_title('Stability Analysis: Lower CV = More Reproducible Results')
ax.set_xticks(x)
ax.set_xticklabels(circuits_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/stability_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/stability_analysis.png")

# ============================================================================
# 7. FIGURE 6: Summary Bar Chart at 22 Qubits
# ============================================================================

print("\n6. Generating summary bar chart...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Performance at 22 qubits
ax1 = axes[0]
circuits_22 = []
cpu_times = []
gpu_times = []
rdbms_times = []

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    df_22 = circuit_df[circuit_df['qubits'] == 22]
    if len(df_22) > 0:
        circuits_22.append(circuit.upper())
        cpu_times.append(df_22['cpu_time'].mean())
        gpu_times.append(df_22['gpu_time'].mean())
        rdbms_times.append(df_22['rdbms_time'].mean())

x = np.arange(len(circuits_22))
width = 0.25

ax1.bar(x - width, cpu_times, width, label='CPU', color='#1f77b4')
ax1.bar(x, gpu_times, width, label='GPU', color='#ff7f0e')
ax1.bar(x + width, rdbms_times, width, label='RDBMS', color='#2ca02c')
ax1.set_yscale('log')
ax1.set_xlabel('Circuit')
ax1.set_ylabel('Time (seconds) [log scale]')
ax1.set_title('Performance at 22 Qubits')
ax1.set_xticks(x)
ax1.set_xticklabels(circuits_22, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Subplot 2: RDBMS advantage (CPU/RDBMS ratio)
ax2 = axes[1]
ratios = [cpu/rdbms for cpu, rdbms in zip(cpu_times, rdbms_times) if rdbms > 0]
circuits_ratio = [c for c, r in zip(circuits_22, rdbms_times) if r > 0]

colors_ratio = ['green' if r > 1 else 'red' for r in ratios]
ax2.bar(circuits_ratio, ratios, color=colors_ratio, alpha=0.7)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
ax2.set_xlabel('Circuit')
ax2.set_ylabel('CPU / RDBMS Time Ratio')
ax2.set_title('RDBMS Advantage (>1 means RDBMS is faster)')
ax2.set_xticklabels(circuits_ratio, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Summary: RDBMS Performance at 22 Qubits', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/summary_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/summary_bar_chart.png")

# ============================================================================
# 8. Generate Summary Table
# ============================================================================

print("\n7. Generating summary table...")

summary_data = []
for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    
    for n in sorted(circuit_df['qubits'].unique()):
        n_df = circuit_df[circuit_df['qubits'] == n]
        
        cpu_mean = n_df['cpu_time'].mean()
        cpu_std = n_df['cpu_time'].std()
        gpu_mean = n_df['gpu_time'].mean()
        gpu_std = n_df['gpu_time'].std()
        rdbms_mean = n_df['rdbms_time'].mean()
        rdbms_std = n_df['rdbms_time'].std()
        
        # Determine best
        times = {'CPU': cpu_mean, 'GPU': gpu_mean, 'RDBMS': rdbms_mean}
        best = min(times, key=times.get)
        
        summary_data.append({
            'Circuit': circuit.upper(),
            'Qubits': n,
            'CPU (s)': f"{cpu_mean:.4f} ± {cpu_std:.4f}",
            'GPU (s)': f"{gpu_mean:.4f} ± {gpu_std:.4f}",
            'RDBMS (s)': f"{rdbms_mean:.4f} ± {rdbms_std:.4f}",
            'Winner': best
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('figures/summary_table.csv', index=False)
print("  ✓ Saved: figures/summary_table.csv")

# ============================================================================
# 9. Generate LaTeX Table Code
# ============================================================================

print("\n8. Generating LaTeX table code...")

latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison (mean $\pm$ std over 10 iterations)}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Circuit} & \textbf{Qubits} & \textbf{CPU (s)} & \textbf{GPU (s)} & \textbf{RDBMS (s)} & \textbf{Winner} \\
\midrule
"""

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    for n in sorted(circuit_df['qubits'].unique()):
        n_df = circuit_df[circuit_df['qubits'] == n]
        
        cpu_mean = n_df['cpu_time'].mean()
        cpu_std = n_df['cpu_time'].std()
        gpu_mean = n_df['gpu_time'].mean()
        gpu_std = n_df['gpu_time'].std()
        rdbms_mean = n_df['rdbms_time'].mean()
        rdbms_std = n_df['rdbms_time'].std()
        
        times = {'CPU': cpu_mean, 'GPU': gpu_mean, 'RDBMS': rdbms_mean}
        best = min(times, key=times.get)
        
        # Bold the best
        cpu_str = f"\\textbf{{{cpu_mean:.4f} $\\pm$ {cpu_std:.4f}}}" if best == 'CPU' else f"{cpu_mean:.4f} $\\pm$ {cpu_std:.4f}"
        gpu_str = f"\\textbf{{{gpu_mean:.4f} $\\pm$ {gpu_std:.4f}}}" if best == 'GPU' else f"{gpu_mean:.4f} $\\pm$ {gpu_std:.4f}"
        rdbms_str = f"\\textbf{{{rdbms_mean:.4f} $\\pm$ {rdbms_std:.4f}}}" if best == 'RDBMS' else f"{rdbms_mean:.4f} $\\pm$ {rdbms_std:.4f}"
        
        latex_table += f"{circuit.upper()} & {n} & {cpu_str} & {gpu_str} & {rdbms_str} & {best} \\\\\n"

latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

with open('figures/latex_table.tex', 'w') as f:
    f.write(latex_table)
print("  ✓ Saved: figures/latex_table.tex")

# ============================================================================
# 10. Print Summary
# ============================================================================

print("\n" + "=" * 60)
print("FIGURES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated files:")
for f in os.listdir('figures'):
    print(f"  - figures/{f}")

print("\n✅ Done!")
