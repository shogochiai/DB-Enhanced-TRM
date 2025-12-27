"""
Visualization for Improved Adversarial Evaluation Results
==========================================================
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path("results_v2/real_trm_evaluation_results.json")
with open(results_file) as f:
    results = json.load(f)

# Create output directory
output_dir = Path("results_v2")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Generating Visualizations for Improved Results")
print("=" * 80)
print()

# ============================================================================
# Figure 1: Adversarial Score Distribution
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Adversarial Score Distribution: Improved Dataset', fontsize=16, fontweight='bold')

configs = ['Vanilla TRM', 'TRM + DB prefilter', 'TRM + DB postfilter', 'TRM + DB pre + post']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for config, (row, col) in zip(configs, positions):
    ax = axes[row, col]
    
    # Get adversarial predictions
    adv_predictions = results[config]['val_adversarial']['predictions']
    
    # Plot histogram
    ax.hist(adv_predictions, bins=20, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.set_xlabel('Predicted Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(config, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    fp_rate = results[config]['val_adversarial']['false_positive_rate']
    robustness = results[config]['val_adversarial']['robustness_score']
    ax.text(0.98, 0.95, f'FP Rate: {fp_rate:.3f}\nRobustness: {robustness:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig1_path = output_dir / "figure1_improved_adversarial_distribution.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig1_path}")
plt.close()

# ============================================================================
# Figure 2: Comparison Bar Chart
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Performance Comparison: Improved Dataset', fontsize=16, fontweight='bold')

x = np.arange(len(configs))
width = 0.25

# Left plot: Clean Accuracy and Adversarial FP
ax1 = axes[0]
clean_acc = [results[c]['val_clean']['accuracy'] for c in configs]
adv_fp = [results[c]['val_adversarial']['false_positive_rate'] for c in configs]

bars1 = ax1.bar(x - width/2, clean_acc, width, label='Clean Accuracy', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, adv_fp, width, label='Adversarial FP', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Clean Accuracy vs Adversarial False Positive Rate', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace('TRM + DB ', '').replace('TRM', 'Vanilla') for c in configs], rotation=15, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.0)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# Right plot: Robustness Score
ax2 = axes[1]
robustness = [results[c]['val_adversarial']['robustness_score'] for c in configs]

bars3 = ax2.bar(x, robustness, width*2, color='#3498db', alpha=0.8)

ax2.set_ylabel('Robustness Score', fontsize=12)
ax2.set_title('Robustness Score (1 - FP_adv)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace('TRM + DB ', '').replace('TRM', 'Vanilla') for c in configs], rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.0)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig2_path = output_dir / "figure2_improved_comparison_bars.png"
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig2_path}")
plt.close()

# ============================================================================
# Figure 3: Category-wise FP Rate
# ============================================================================

# Analyze by adversarial category
fig, ax = plt.subplots(figsize=(12, 6))

# Load adversarial dataset to get categories
import json
adv_data = []
with open("data_v2/val_adversarial_v2.jsonl") as f:
    for line in f:
        if line.strip():
            adv_data.append(json.loads(line))

# Group by category
categories = {}
for i, item in enumerate(adv_data):
    cat = item['adversarial_type']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(i)

# Calculate FP rate per category for each config
category_names = list(categories.keys())
category_fp_rates = {config: [] for config in configs}

for config in configs:
    predictions = results[config]['val_adversarial']['predictions']
    
    for cat_name in category_names:
        indices = categories[cat_name]
        cat_predictions = [predictions[i] for i in indices]
        
        # FP rate: how many adversarial samples scored >= 0.5
        fp_count = sum(1 for p in cat_predictions if p >= 0.5)
        fp_rate = fp_count / len(cat_predictions) if cat_predictions else 0.0
        
        category_fp_rates[config].append(fp_rate)

# Plot grouped bar chart
x = np.arange(len(category_names))
width = 0.2

for i, config in enumerate(configs):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, category_fp_rates[config], width, 
                   label=config.replace('TRM + DB ', '').replace('TRM', 'Vanilla'),
                   alpha=0.8)

ax.set_ylabel('False Positive Rate', fontsize=12)
ax.set_title('Category-wise Adversarial False Positive Rate', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in category_names], rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

plt.tight_layout()
fig3_path = output_dir / "figure3_category_wise_fp.png"
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig3_path}")
plt.close()

# ============================================================================
# Generate Markdown Table
# ============================================================================

table_md = """# Improved Adversarial Evaluation Results

## Table 1: Main Experimental Results

| Model                 | Clean Acc | Adv FP | Robustness | ms/pair |
|-----------------------|-----------|--------|------------|---------|
"""

for config in configs:
    clean_acc = results[config]['val_clean']['accuracy']
    adv_fp = results[config]['val_adversarial']['false_positive_rate']
    robustness = results[config]['val_adversarial']['robustness_score']
    latency = results[config]['val_adversarial']['avg_latency_ms']
    
    config_name = config.replace('TRM + DB ', '').replace('TRM', 'Vanilla TRM')
    table_md += f"| {config_name:21s} | {clean_acc:9.3f} | {adv_fp:6.3f} | {robustness:10.3f} | {latency:7.1f} |\n"

table_path = output_dir / "table1_improved_results.md"
with open(table_path, 'w') as f:
    f.write(table_md)

print(f"Saved: {table_path}")
print()
print("=" * 80)
print("Visualization complete!")
print("=" * 80)
