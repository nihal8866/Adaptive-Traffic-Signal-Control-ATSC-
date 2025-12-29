import numpy as np

def load_data(filepath):
    """Load data from text file"""
    with open(filepath, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def calculate_metrics(queue_data, name="Model"):
    """Calculate evaluation metrics"""
    metrics = {
        'name': name,
        'avg_queue': np.mean(queue_data),
        'median_queue': np.median(queue_data),
        'std_queue': np.std(queue_data),
        'min_queue': np.min(queue_data),
        'max_queue': np.max(queue_data),
    }
    return metrics

def print_metrics(metrics):
    """Print metrics in a formatted table"""
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS: {metrics['name']}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':<20}")
    print(f"{'-'*60}")
    print(f"{'Average Queue Length':<30} {metrics['avg_queue']:.2f} vehicles")
    print(f"{'Median Queue Length':<30} {metrics['median_queue']:.2f} vehicles")
    print(f"{'Standard Deviation':<30} {metrics['std_queue']:.2f}")
    print(f"{'Minimum Queue Length':<30} {metrics['min_queue']:.2f} vehicles")
    print(f"{'Maximum Queue Length':<30} {metrics['max_queue']:.2f} vehicles")
    print(f"{'='*60}\n")

def compare_models(baseline_metrics, model_metrics):
    """Compare model against baseline and show improvement"""
    baseline_avg = baseline_metrics['avg_queue']
    model_avg = model_metrics['avg_queue']
    
    improvement = ((baseline_avg - model_avg) / baseline_avg) * 100
    difference = baseline_avg - model_avg
    
    print("\n" + "="*60)
    print(f"COMPARISON: {model_metrics['name']} vs {baseline_metrics['name']}")
    print("="*60)
    print(f"{baseline_metrics['name']} Average:     {baseline_avg:.2f} vehicles")
    print(f"{model_metrics['name']} Average:       {model_avg:.2f} vehicles")
    print(f"Difference:                {difference:+.2f} vehicles")
    
    if improvement > 0:
        print(f"\n{'IMPROVEMENT:':>30} {improvement:.2f}%")
        print(f"{'':>30} ✓ Queue decreased by {improvement:.2f}%")
    else:
        print(f"\n{'DEGRADATION:':>30} {abs(improvement):.2f}%")
        print(f"{'':>30} ✗ Queue increased by {abs(improvement):.2f}%")
    
    print("="*60 + "\n")

# Main evaluation
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load Fixed-Time baseline
    fixed_queue = load_data('comparison/fixed_time_baseline_2000/plot_queue_data.txt')
    fixed_metrics = calculate_metrics(fixed_queue, "Fixed-Time Baseline")
    print_metrics(fixed_metrics)
    
    # Load RL Agent
    rl_queue = load_data('models/model_15/test/plot_queue_data.txt')
    rl_metrics = calculate_metrics(rl_queue, "RL Agent (DQN)")
    print_metrics(rl_metrics)
    
    # Compare
    compare_models(fixed_metrics, rl_metrics)
    
    print("✓ Evaluation complete!")