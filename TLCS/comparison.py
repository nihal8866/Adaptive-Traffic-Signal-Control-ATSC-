import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    with open(filepath, 'r') as f:
        return [float(line.strip()) for line in f]

# Load data
rl_queue = load_data('models/model_15/test/plot_queue_data.txt')
fixed_queue = load_data('comparison/fixed_time_baseline_2000/plot_queue_data.txt')

# Calculate averages
rl_avg = np.mean(rl_queue)
fixed_avg = np.mean(fixed_queue)
improvement = ((fixed_avg - rl_avg) / fixed_avg) * 100

# Create figure with 2 plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Line comparison over time
axes[0].plot(rl_queue, label='RL Agent', linewidth=1.5, alpha=0.8, color='#2ecc71')
axes[0].plot(fixed_queue, label='Fixed-Time', linewidth=1.5, alpha=0.7, color='#e74c3c')
axes[0].set_xlabel('Simulation Step', fontsize=11)
axes[0].set_ylabel('Queue Length (vehicles)', fontsize=11)
axes[0].set_title('Queue Length Over Time', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Bar chart comparison
bars = axes[1].bar(['RL Agent', 'Fixed-Time'], [rl_avg, fixed_avg], 
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Average Queue Length (vehicles)', fontsize=11)
axes[1].set_title('Average Queue Length Comparison', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation on bar chart
if improvement > 0:
    axes[1].annotate(f'Queue decreased\nby {improvement:.1f}%',
                     xy=(0.5, max(rl_avg, fixed_avg) * 0.5),
                     xytext=(0.5, max(rl_avg, fixed_avg) * 0.7),
                     ha='center', fontsize=11, fontweight='bold', color='green',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))
else:
    axes[1].annotate(f'Queue increased\nby {abs(improvement):.1f}%',
                     xy=(0.5, max(rl_avg, fixed_avg) * 0.5),
                     xytext=(0.5, max(rl_avg, fixed_avg) * 0.7),
                     ha='center', fontsize=11, fontweight='bold', color='red',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('comparison/rl_vs_fixed_comparison_2000.png', dpi=300, bbox_inches='tight')
print("Saved: comparison/rl_vs_fixed_comparison_2000.png")
plt.close()