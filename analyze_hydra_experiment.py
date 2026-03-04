import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyze_experiment(experiment_dir):
    csv_path = os.path.join(experiment_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Plot Score vs Generation for each Head
    plt.figure(figsize=(12, 6))
    for head in df['head'].unique():
        head_data = df[df['head'] == head]
        plt.plot(head_data['generation'], head_data['score'], label=head, marker='o')
    
    plt.title("Hydra Evolution: Best Score per Head over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, "score_history.png"))
    print(f"Saved score_history.png to {experiment_dir}")

    # 2. Plot Model Size (n_emb * n_layer) evolution
    plt.figure(figsize=(12, 6))
    for head in df['head'].unique():
        head_data = df[df['head'] == head]
        # Approximation of complexity
        complexity = head_data['n_emb'] * head_data['n_layer'] 
        plt.plot(head_data['generation'], complexity, label=head, marker='x')

    plt.title("Hydra Evolution: Model Complexity (Emb * Layer)")
    plt.xlabel("Generation")
    plt.ylabel("Complexity Index")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_dir, "complexity_history.png"))
    print(f"Saved complexity_history.png to {experiment_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_hydra_experiment.py <experiment_dir>")
        sys.exit(1)
    
    analyze_experiment(sys.argv[1])
