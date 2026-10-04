# generate_indexing_figure.py
"""
Generate figure for indexing strategies benchmark
Based on your actual output from indexing_strategies.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from your indexing benchmark output
queries = ['TOP 10', 'THRESHOLD > 0.01', 'EXACT LOOKUP', 'RANGE QUERY', 'PATTERN MATCH']

# Times in milliseconds (from your output)
# state_no_index, state_btree, state_hash, state_composite, state_covering, state_partial
no_index = [0.834, 0.477, 0.006, 0.488, 2.277]
btree = [0.009, 0.004, 0.381, 0.005, 2.195]
hash_idx = [0.803, 0.483, 0.340, 0.489, 2.190]
composite = [0.008, 0.004, 0.368, 0.005, 2.184]
covering = [0.008, 0.004, 0.367, 0.005, 2.189]
partial = [0.790, 0.004, 0.367, 0.485, 2.182]

# Speedup over no index for threshold query (most dramatic)
threshold_speedup = [1, 0.477/0.004, 0.477/0.483, 0.477/0.004, 0.477/0.004, 0.477/0.004]
threshold_labels = ['No Index', 'B-tree', 'Hash', 'Composite', 'Covering', 'Partial']
threshold_speedup_values = [1, 119.25, 0.99, 119.25, 119.25, 119.25]

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of times for each query type
ax1 = axes[0]
x = np.arange(len(queries))
width = 0.12

# Only show most relevant index types for clarity
ax1.bar(x - 2*width, no_index, width, label='No Index', color='#1f77b4', alpha=0.8)
ax1.bar(x - width, btree, width, label='B-tree', color='#ff7f0e', alpha=0.8)
ax1.bar(x, covering, width, label='Covering', color='#2ca02c', alpha=0.8)
ax1.bar(x + width, partial, width, label='Partial', color='#d62728', alpha=0.8)
ax1.bar(x + 2*width, hash_idx, width, label='Hash', color='#9467bd', alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(queries, rotation=45, ha='right')
ax1.set_ylabel('Query Time (ms)')
ax1.set_title('Indexing Performance Comparison')
ax1.set_yscale('log')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Add annotation for the dramatic speedup
ax1.annotate('119× faster!', xy=(1, 0.004), xytext=(0.5, 0.01),
             arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)

# Plot 2: Speedup bar chart for threshold query
ax2 = axes[1]
colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#2ca02c', '#d62728']
bars = ax2.bar(threshold_labels, threshold_speedup_values, color=colors, alpha=0.7)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Index Baseline')
ax2.set_ylabel('Speedup (No Index Time / Index Time)')
ax2.set_title('Speedup for THRESHOLD > 0.01 Query')
ax2.set_yscale('log')

# Add value labels on bars
for bar, val in zip(bars, threshold_speedup_values):
    if val > 1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Database Indexing for Quantum State Queries', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/indexing_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: figures/indexing_results.png")

# Also create a standalone table for the report
print("\n" + "=" * 60)
print("INDEXING RESULTS SUMMARY")
print("=" * 60)

print("\nQuery Performance (ms):")
print("-" * 70)
print(f"{'Index Type':<15} {'TOP 10':<12} {'THRESHOLD':<12} {'EXACT':<12} {'RANGE':<12} {'PATTERN':<12}")
print("-" * 70)
print(f"{'No Index':<15} {no_index[0]:<12.3f} {no_index[1]:<12.3f} {no_index[2]:<12.3f} {no_index[3]:<12.3f} {no_index[4]:<12.3f}")
print(f"{'B-tree':<15} {btree[0]:<12.3f} {btree[1]:<12.3f} {btree[2]:<12.3f} {btree[3]:<12.3f} {btree[4]:<12.3f}")
print(f"{'Covering':<15} {covering[0]:<12.3f} {covering[1]:<12.3f} {covering[2]:<12.3f} {covering[3]:<12.3f} {covering[4]:<12.3f}")
print(f"{'Partial':<15} {partial[0]:<12.3f} {partial[1]:<12.3f} {partial[2]:<12.3f} {partial[3]:<12.3f} {partial[4]:<12.3f}")

print("\n📊 KEY FINDING:")
print("   B-tree/Covering indexes provide 119× speedup for THRESHOLD queries!")
print("   Exact lookups are fastest with PRIMARY KEY (No Index needed)")

# Generate LaTeX table for indexing results
latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Indexing Strategy Performance Comparison (times in ms)}
\label{tab:indexing}
\begin{tabular}{lccccc}
\toprule
\textbf{Index Type} & \textbf{TOP 10} & \textbf{THRESHOLD} & \textbf{EXACT} & \textbf{RANGE} & \textbf{PATTERN} \\
\midrule
No Index & 0.834 & 0.477 & \textbf{0.006} & 0.488 & 2.277 \\
B-tree & \textbf{0.009} & \textbf{0.004} & 0.381 & \textbf{0.005} & 2.195 \\
Covering & \textbf{0.008} & \textbf{0.004} & 0.367 & \textbf{0.005} & 2.189 \\
Partial & 0.790 & \textbf{0.004} & 0.367 & 0.485 & \textbf{2.182} \\
\bottomrule
\end{tabular}
\end{table}
"""

with open('figures/indexing_latex_table.tex', 'w') as f:
    f.write(latex_table)
print("\n✓ Saved: figures/indexing_latex_table.tex")

print("\n✅ Indexing figure and table generated!")
