#!/usr/bin/env python3
"""
Analyze evolution results from log files
Usage: python analyze_results.py <log_file>
"""

import sys
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file):
    """Parse evolution log file and extract key metrics"""
    
    generations = []
    best_losses = []
    species_counts = []
    timestamps = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Parse generation lines
        gen_match = re.search(r'--- Generation (\d+)/(\d+) ---', line)
        if gen_match:
            gen_num = int(gen_match.group(1))
            generations.append(gen_num)
            timestamps.append(line.strip())
            continue
            
        # Parse best loss lines
        best_match = re.search(r'Best: (\d+\.\d+)', line)
        if best_match and generations:
            best_losses.append(float(best_match.group(1)))
            continue
            
        # Parse species count
        species_match = re.search(r'Species: (\d+)', line)
        if species_match and generations:
            species_counts.append(int(species_match.group(1)))
    
    return generations, best_losses, species_counts, timestamps

def extract_final_stats(log_file):
    """Extract final statistics from log"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    stats = {}
    
    # Extract final best loss
    best_loss_match = re.search(r'Best loss: (\d+\.\d+)', content)
    if best_loss_match:
        stats['final_best_loss'] = float(best_loss_match.group(1))
    
    # Extract total time
    time_match = re.search(r'Total time: (\d+\.\d+)s', content)
    if time_match:
        stats['total_time_seconds'] = float(time_match.group(1))
        stats['total_time_minutes'] = stats['total_time_seconds'] / 60
    
    # Extract total evaluations
    eval_match = re.search(r'Total evaluations: (\d+)', content)
    if eval_match:
        stats['total_evaluations'] = int(eval_match.group(1))
    
    # Extract best config
    config_match = re.search(r'Best config: (.+)', content)
    if config_match:
        stats['best_config'] = config_match.group(1)
    
    return stats

def create_plots(generations, best_losses, species_counts, output_prefix):
    """Create visualization plots"""
    
    if not generations or not best_losses:
        print("No data to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss over generations
    ax1.plot(generations, best_losses, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Loss')
    ax1.set_title('Evolution Progress: Best Loss Over Generations')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Species diversity
    if species_counts:
        ax2.plot(generations[:len(species_counts)], species_counts, 'g-', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Species')
        ax2.set_title('Species Diversity Over Generations')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss improvement rate
    if len(best_losses) > 1:
        improvement_rate = np.diff(best_losses)
        ax3.plot(generations[1:], improvement_rate, 'r-', linewidth=1, marker='d', markersize=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Loss Change')
        ax3.set_title('Loss Improvement Rate')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Loss distribution histogram
    ax4.hist(best_losses, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Loss Value Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_evolution_plots.png', dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {output_prefix}_evolution_plots.png")

def analyze_convergence(generations, best_losses):
    """Analyze convergence characteristics"""
    
    if len(best_losses) < 10:
        return {}
    
    # Find plateau point (where improvement becomes minimal)
    improvements = np.abs(np.diff(best_losses))
    threshold = np.std(improvements) * 0.1
    
    plateau_start = None
    for i, improvement in enumerate(improvements):
        if i > 10 and all(imp < threshold for imp in improvements[i-10:i]):
            plateau_start = generations[i]
            break
    
    # Calculate convergence metrics
    initial_loss = best_losses[0]
    final_loss = best_losses[-1]
    total_improvement = initial_loss - final_loss
    improvement_rate = total_improvement / len(best_losses)
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'total_improvement': total_improvement,
        'improvement_rate_per_gen': improvement_rate,
        'plateau_start_generation': plateau_start,
        'converged': plateau_start is not None
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_prefix = log_file.replace('.log', '').replace('results/', '')
    
    print(f"Analyzing evolution results from: {log_file}")
    print("=" * 50)
    
    try:
        # Parse the log file
        generations, best_losses, species_counts, timestamps = parse_log_file(log_file)
        
        if not generations:
            print("No generation data found in log file")
            return
        
        # Extract final statistics
        stats = extract_final_stats(log_file)
        
        # Analyze convergence
        convergence = analyze_convergence(generations, best_losses)
        
        # Print summary
        print(f"Total generations: {len(generations)}")
        print(f"Final best loss: {best_losses[-1]:.4f}")
        print(f"Initial best loss: {best_losses[0]:.4f}")
        print(f"Total improvement: {convergence.get('total_improvement', 0):.4f}")
        
        if 'total_time_minutes' in stats:
            print(f"Total time: {stats['total_time_minutes']:.1f} minutes")
            print(f"Time per generation: {stats['total_time_minutes']/len(generations):.2f} minutes")
        
        if 'total_evaluations' in stats:
            print(f"Total evaluations: {stats['total_evaluations']}")
        
        if convergence.get('converged'):
            print(f"Convergence detected at generation: {convergence['plateau_start_generation']}")
        else:
            print("No clear convergence detected (still improving)")
        
        if 'best_config' in stats:
            print(f"Best configuration: {stats['best_config']}")
        
        # Create plots
        create_plots(generations, best_losses, species_counts, output_prefix)
        
        # Save analysis results
        analysis_results = {
            'summary': {
                'total_generations': len(generations),
                'final_best_loss': best_losses[-1],
                'initial_best_loss': best_losses[0],
                'convergence': convergence,
                'final_stats': stats
            },
            'data': {
                'generations': generations,
                'best_losses': best_losses,
                'species_counts': species_counts
            }
        }
        
        with open(f'{output_prefix}_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\nAnalysis results saved to: {output_prefix}_analysis.json")
        
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
