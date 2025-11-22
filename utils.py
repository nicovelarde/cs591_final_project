"""
Utilities for PPO Pong Experiments
===================================

This module provides functions for:
1. Loading and processing experimental data
2. Generating publication-quality plots
3. Computing statistics and metrics
4. Creating summary tables

All plotting functions include detailed comments explaining what's being visualized.
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

# Set plotting style for professional-looking figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_csv_results(csv_path: str) -> Dict[str, List]:
    """
    Load results from a CSV file.
    
    Each CSV file contains episode-by-episode data:
    - episode: Episode number
    - reward: Reward for that episode
    - episode_length: Length of the episode
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary with keys 'episode', 'reward', 'episode_length'
    """
    # Initialize lists to store data
    episodes = []
    rewards = []
    lengths = []
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Iterate through each row
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            lengths.append(int(row['episode_length']))
    
    return {
        'episodes': np.array(episodes),
        'rewards': np.array(rewards),
        'lengths': np.array(lengths)
    }


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load all experiment results from the results directory.
    
    Scans for all CSV files matching the pattern: {config_name}_seed{seed}.csv
    Groups results by configuration name and seed.
    
    Args:
        results_dir: Path to results directory
    
    Returns:
        Nested dictionary: {config_name: {seed: data}}
    """
    # Dictionary to store results
    all_results = {}
    
    logs_dir = os.path.join(results_dir, 'logs')
    
    # Check if logs directory exists
    if not os.path.exists(logs_dir):
        print(f"Error: {logs_dir} not found")
        return {}
    
    # Iterate through all CSV files
    for filename in sorted(os.listdir(logs_dir)):
        # Only process CSV files
        if not filename.endswith('.csv'):
            continue
        
        # Parse filename: {config_name}_seed{seed}.csv
        parts = filename.replace('.csv', '').rsplit('_seed', 1)
        if len(parts) != 2:
            continue
        
        config_name = parts[0]
        seed = int(parts[1])
        
        # Load data from CSV
        csv_path = os.path.join(logs_dir, filename)
        data = load_csv_results(csv_path)
        
        # Store in nested dictionary
        if config_name not in all_results:
            all_results[config_name] = {}
        
        all_results[config_name][seed] = data
    
    print(f"Loaded {len(all_results)} configurations")
    for config_name in all_results:
        print(f"  {config_name}: {len(all_results[config_name])} seeds")
    
    return all_results


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_moving_average(data: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Compute moving average of data.
    
    This smooths out noise to show underlying trends better.
    For each point, we average over a window of surrounding points.
    
    Args:
        data: Input array
        window_size: Size of moving average window
    
    Returns:
        Array of moving averages (same length as input)
    """
    # Initialize output array
    moving_avg = np.zeros(len(data))
    
    # Iterate through each point
    for i in range(len(data)):
        # Define window: [max(0, i - window_size//2), min(len(data), i + window_size//2)]
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        
        # Compute average over window
        moving_avg[i] = np.mean(data[start:end])
    
    return moving_avg


def compute_statistics_by_config(all_results: Dict) -> Dict:
    """
    Compute aggregate statistics for each configuration.
    
    Computes:
    - Mean and std of final performance (last 100 episodes)
    - Convergence metrics
    - Variance across seeds
    
    Args:
        all_results: Nested dictionary of results
    
    Returns:
        Dictionary with statistics per configuration
    """
    stats = {}
    
    # Process each configuration
    for config_name, seed_results in all_results.items():
        # Extract final rewards from all seeds
        final_rewards = []
        convergence_episodes = []
        
        for seed, data in seed_results.items():
            # Last 100 episodes average (final performance)
            final_reward = np.mean(data['rewards'][-100:])
            final_rewards.append(final_reward)
            
            # Find convergence episode (first episode with reward > 18)
            convergence_idx = np.where(data['rewards'] > 18)[0]
            if len(convergence_idx) > 0:
                convergence_episodes.append(convergence_idx[0])
            else:
                convergence_episodes.append(len(data['rewards']))
        
        # Compute statistics
        stats[config_name] = {
            'mean_reward': np.mean(final_rewards),
            'std_reward': np.std(final_rewards),
            'max_reward': np.max(final_rewards),
            'min_reward': np.min(final_rewards),
            'mean_convergence': np.mean(convergence_episodes),
            'std_convergence': np.std(convergence_episodes),
            'num_seeds': len(seed_results)
        }
    
    return stats


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_reward_curves(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Plot reward vs episode for all configurations.
    
    Creates individual learning curves for each configuration,
    with shaded confidence bands showing variance across seeds.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    # Organize by learning rate (3 columns) and entropy (2-3 rows)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('PPO on Pong: Reward per Episode', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Sort configurations by name for consistent display
    config_names = sorted(all_results.keys())
    
    # Plot each configuration
    for ax_idx, config_name in enumerate(config_names):
        ax = axes[ax_idx]
        seed_results = all_results[config_name]
        
        # Collect all episodes and rewards across seeds
        all_episodes = None
        all_rewards_by_seed = []
        
        for seed in sorted(seed_results.keys()):
            data = seed_results[seed]
            episodes = data['episodes']
            rewards = data['rewards']
            
            # Store for later
            if all_episodes is None:
                all_episodes = episodes
            
            all_rewards_by_seed.append(rewards)
        
        # Convert to numpy array for easier computation
        all_rewards = np.array(all_rewards_by_seed)
        
        # Compute mean and std across seeds
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        
        # Plot individual seed curves (faint)
        for rewards in all_rewards:
            ax.plot(all_episodes, rewards, alpha=0.2, color='blue')
        
        # Plot mean curve (bold)
        ax.plot(all_episodes, mean_rewards, color='darkblue', linewidth=2, label='Mean')
        
        # Plot confidence band (shaded region)
        ax.fill_between(
            all_episodes,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.3,
            color='lightblue',
            label='±1 std'
        )
        
        # Plot convergence threshold line (18 = good performance on Pong)
        ax.axhline(y=18, color='green', linestyle='--', alpha=0.5, label='Target (18)')
        
        # Configure subplot
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{config_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(len(config_names), len(axes)):
        fig.delaxes(axes[idx])
    
    # Save figure
    plot_path = os.path.join(output_dir, 'reward_vs_episode.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_moving_average(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Plot 20-episode moving average for all configurations.
    
    The moving average smooths out noise and shows the trend more clearly.
    This makes it easier to compare convergence speed between configurations.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('PPO on Pong: 20-Episode Moving Average', fontsize=16, fontweight='bold')
    
    config_names = sorted(all_results.keys())
    
    # Create a subplot for each configuration
    for idx, config_name in enumerate(config_names):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        seed_results = all_results[config_name]
        
        # Collect data from all seeds
        all_episodes = None
        all_moving_avgs = []
        
        for seed in sorted(seed_results.keys()):
            data = seed_results[seed]
            episodes = data['episodes']
            rewards = data['rewards']
            
            if all_episodes is None:
                all_episodes = episodes
            
            # Compute moving average for this seed
            moving_avg = compute_moving_average(rewards, window_size=20)
            all_moving_avgs.append(moving_avg)
        
        # Convert to numpy array
        all_moving_avgs = np.array(all_moving_avgs)
        
        # Compute mean and std across seeds
        mean_ma = np.mean(all_moving_avgs, axis=0)
        std_ma = np.std(all_moving_avgs, axis=0)
        
        # Plot individual curves (faint)
        for ma in all_moving_avgs:
            ax.plot(all_episodes, ma, alpha=0.2, color='darkblue')
        
        # Plot mean curve
        ax.plot(all_episodes, mean_ma, color='darkblue', linewidth=2.5)
        
        # Plot confidence band
        ax.fill_between(
            all_episodes,
            mean_ma - std_ma,
            mean_ma + std_ma,
            alpha=0.3,
            color='lightblue'
        )
        
        # Target line
        ax.axhline(y=18, color='green', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (20-ep MA)')
        ax.set_title(f'{config_name}')
        ax.grid(True, alpha=0.3)
    
    # Save figure
    plot_path = os.path.join(output_dir, 'moving_average.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_convergence_comparison(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Bar chart comparing convergence speed across configurations.
    
    Convergence is measured as the episode number where reward first exceeds 18
    (a good performance threshold for Pong).
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute convergence metrics
    config_names = []
    convergence_means = []
    convergence_stds = []
    
    for config_name in sorted(all_results.keys()):
        seed_results = all_results[config_name]
        convergence_episodes = []
        
        # For each seed, find when it reached performance > 18
        for seed, data in seed_results.items():
            idx = np.where(data['rewards'] > 18)[0]
            if len(idx) > 0:
                convergence_episodes.append(idx[0])
            else:
                # If never converged, use total episodes
                convergence_episodes.append(len(data['rewards']))
        
        config_names.append(config_name)
        convergence_means.append(np.mean(convergence_episodes))
        convergence_stds.append(np.std(convergence_episodes))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(config_names))
    width = 0.8
    
    # Plot bars with error bars
    ax.bar(x, convergence_means, width, yerr=convergence_stds, 
           capsize=5, color='steelblue', edgecolor='navy', linewidth=1.5,
           error_kw={'linewidth': 2, 'ecolor': 'gray'})
    
    # Formatting
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episodes to Convergence (>18 reward)', fontsize=12, fontweight='bold')
    ax.set_title('PPO on Pong: Convergence Speed by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best and worst
    best_idx = np.argmin(convergence_means)
    worst_idx = np.argmax(convergence_means)
    
    ax.patches[best_idx].set_facecolor('green')
    ax.patches[best_idx].set_alpha(0.7)
    ax.patches[worst_idx].set_facecolor('red')
    ax.patches[worst_idx].set_alpha(0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_entropy_heatmap(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Heatmap showing final performance for each (learning_rate, entropy) combination.
    
    This visualization makes it easy to see how two hyperparameters interact.
    Bright colors = high performance, dark colors = low performance.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract hyperparameters and performance
    data_dict = {}
    
    for config_name, seed_results in all_results.items():
        # Parse config name to extract hyperparameters
        # Format: ppo_lr{lr}_entropy{entropy}
        parts = config_name.split('_')
        
        # Extract learning rate
        lr_str = parts[1]  # e.g., "lr3e-5"
        lr = float(lr_str.replace('lr', '').replace('e', 'e-').replace('--', '-'))
        
        # Extract entropy coefficient
        entropy_str = parts[2]  # e.g., "entropy0.01"
        entropy = float(entropy_str.replace('entropy', ''))
        
        # Compute mean performance across seeds (last 100 episodes)
        performances = []
        for seed, data in seed_results.items():
            perf = np.mean(data['rewards'][-100:])
            performances.append(perf)
        
        mean_perf = np.mean(performances)
        
        if lr not in data_dict:
            data_dict[lr] = {}
        
        data_dict[lr][entropy] = mean_perf
    
    # Create heatmap data
    lrs = sorted(data_dict.keys())
    entropies = sorted(set(e for lr_dict in data_dict.values() for e in lr_dict.keys()))
    
    heatmap_data = np.zeros((len(entropies), len(lrs)))
    
    for i, entropy in enumerate(entropies):
        for j, lr in enumerate(lrs):
            heatmap_data[i, j] = data_dict[lr].get(entropy, np.nan)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[f'{lr:.0e}' for lr in lrs],
                yticklabels=[f'{e:.2f}' for e in entropies],
                ax=ax, cbar_kws={'label': 'Final Reward'})
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('PPO on Pong: Final Performance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def create_summary_table(all_results: Dict, output_dir: str = 'results/logs') -> pd.DataFrame:
    """
    Create a comprehensive summary table of all results.
    
    Returns a DataFrame with one row per configuration, showing:
    - Hyperparameters
    - Mean and std of final performance
    - Convergence metrics
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save results
    
    Returns:
        Pandas DataFrame with results
    """
    # Compute statistics
    stats = compute_statistics_by_config(all_results)
    
    # Create list of rows for DataFrame
    rows = []
    
    for config_name in sorted(stats.keys()):
        stat = stats[config_name]
        
        # Parse config name to extract hyperparameters
        parts = config_name.split('_')
        lr = float(parts[1].replace('lr', '').replace('e', 'e-').replace('--', '-'))
        entropy = float(parts[2].replace('entropy', ''))
        
        row = {
            'Configuration': config_name,
            'Learning Rate': f'{lr:.0e}',
            'Entropy Coeff': f'{entropy:.2f}',
            'Final Reward (μ)': f'{stat["mean_reward"]:.2f}',
            'Final Reward (σ)': f'{stat["std_reward"]:.2f}',
            'Best Seed': f'{stat["max_reward"]:.2f}',
            'Worst Seed': f'{stat["min_reward"]:.2f}',
            'Convergence Ep (μ)': f'{stat["mean_convergence"]:.0f}',
            'Convergence Ep (σ)': f'{stat["std_convergence"]:.0f}',
            'Num Seeds': stat['num_seeds']
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    return df


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def plot_learning_rate_effect(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Plot showing learning rate effect (keeping entropy constant).
    One plot per entropy level showing how different LRs perform.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique entropy values
    entropies = set()
    lr_dict = {}
    
    for config_name, seed_results in all_results.items():
        parts = config_name.split('_')
        lr_str = parts[1]
        lr = float(lr_str.replace('lr', '').replace('e', 'e-').replace('--', '-'))
        entropy_str = parts[2]
        entropy = float(entropy_str.replace('entropy', ''))
        
        entropies.add(entropy)
        
        if entropy not in lr_dict:
            lr_dict[entropy] = {}
        
        # Get mean reward for this config
        final_rewards = [np.mean(seed_results[seed]['rewards'][-100:]) 
                        for seed in seed_results]
        lr_dict[entropy][lr] = {
            'mean': np.mean(final_rewards),
            'std': np.std(final_rewards)
        }
    
    # Create subplots for each entropy level
    entropies = sorted(entropies)
    fig, axes = plt.subplots(1, len(entropies), figsize=(15, 4))
    
    if len(entropies) == 1:
        axes = [axes]
    
    for ax_idx, entropy in enumerate(entropies):
        ax = axes[ax_idx]
        
        lrs = sorted(lr_dict[entropy].keys())
        means = [lr_dict[entropy][lr]['mean'] for lr in lrs]
        stds = [lr_dict[entropy][lr]['std'] for lr in lrs]
        
        x = np.arange(len(lrs))
        ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', 
               edgecolor='navy', linewidth=1.5, error_kw={'linewidth': 2})
        
        ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('Final Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'LR Effect (Entropy={entropy:.2f})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'learning_rate_effect.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_entropy_effect(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Plot showing entropy effect (keeping learning rate constant).
    One plot per learning rate showing how different entropy levels perform.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique learning rates
    lrs = set()
    entropy_dict = {}
    
    for config_name, seed_results in all_results.items():
        parts = config_name.split('_')
        lr_str = parts[1]
        lr = float(lr_str.replace('lr', '').replace('e', 'e-').replace('--', '-'))
        entropy_str = parts[2]
        entropy = float(entropy_str.replace('entropy', ''))
        
        lrs.add(lr)
        
        if lr not in entropy_dict:
            entropy_dict[lr] = {}
        
        # Get mean reward for this config
        final_rewards = [np.mean(seed_results[seed]['rewards'][-100:]) 
                        for seed in seed_results]
        entropy_dict[lr][entropy] = {
            'mean': np.mean(final_rewards),
            'std': np.std(final_rewards)
        }
    
    # Create subplots for each learning rate
    lrs = sorted(lrs, reverse=True)
    fig, axes = plt.subplots(1, len(lrs), figsize=(15, 4))
    
    if len(lrs) == 1:
        axes = [axes]
    
    for ax_idx, lr in enumerate(lrs):
        ax = axes[ax_idx]
        
        entropies = sorted(entropy_dict[lr].keys())
        means = [entropy_dict[lr][e]['mean'] for e in entropies]
        stds = [entropy_dict[lr][e]['std'] for e in entropies]
        
        x = np.arange(len(entropies))
        ax.bar(x, means, yerr=stds, capsize=5, color='coral', 
               edgecolor='darkred', linewidth=1.5, error_kw={'linewidth': 2})
        
        ax.set_xlabel('Entropy Coefficient', fontsize=11, fontweight='bold')
        ax.set_ylabel('Final Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Entropy Effect (LR={lr:.0e})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.2f}' for e in entropies], rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'entropy_effect.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_final_performance_boxplot(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Box plot showing distribution of final rewards across all configurations.
    Shows stability and best/worst performers.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_names = []
    reward_data = []
    
    for config_name in sorted(all_results.keys()):
        seed_results = all_results[config_name]
        
        # Collect final rewards from all seeds
        final_rewards = [np.mean(seed_results[seed]['rewards'][-100:]) 
                        for seed in seed_results]
        
        config_names.append(config_name)
        reward_data.append(final_rewards)
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bp = ax.boxplot(reward_data, labels=config_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Reward (Last 100 Episodes)', fontsize=12, fontweight='bold')
    ax.set_title('PPO on Pong: Final Performance Distribution by Configuration', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'final_performance_boxplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_episode_length_analysis(all_results: Dict, output_dir: str = 'results/plots'):
    """
    Plot showing average episode length for each configuration.
    Longer episodes mean the agent is playing longer (potentially better policy).
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_names = []
    avg_lengths = []
    std_lengths = []
    
    for config_name in sorted(all_results.keys()):
        seed_results = all_results[config_name]
        
        # Collect average episode lengths across all seeds
        all_lengths = []
        for seed, data in seed_results.items():
            # Average length of last 100 episodes
            avg_len = np.mean(data['lengths'][-100:])
            all_lengths.append(avg_len)
        
        config_names.append(config_name)
        avg_lengths.append(np.mean(all_lengths))
        std_lengths.append(np.std(all_lengths))
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(config_names))
    ax.bar(x, avg_lengths, yerr=std_lengths, capsize=5, 
           color='lightgreen', edgecolor='darkgreen', linewidth=1.5,
           error_kw={'linewidth': 2, 'ecolor': 'gray'})
    
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Episode Length (Last 100 Episodes)', fontsize=12, fontweight='bold')
    ax.set_title('PPO on Pong: Episode Length by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'episode_length_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def analyze_all_results(results_dir: str = 'results'):
    """
    Complete analysis pipeline: load data, compute stats, generate plots.
    
    Args:
        results_dir: Directory containing results
    """
    print("Loading results...")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    print("\nGenerating plots...")
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    # Generate all plots (original 4)
    plot_reward_curves(all_results, os.path.join(results_dir, 'plots'))
    plot_moving_average(all_results, os.path.join(results_dir, 'plots'))
    plot_convergence_comparison(all_results, os.path.join(results_dir, 'plots'))
    plot_entropy_heatmap(all_results, os.path.join(results_dir, 'plots'))
    
    # Generate new plots
    plot_learning_rate_effect(all_results, os.path.join(results_dir, 'plots'))
    plot_entropy_effect(all_results, os.path.join(results_dir, 'plots'))
    plot_final_performance_boxplot(all_results, os.path.join(results_dir, 'plots'))
    plot_episode_length_analysis(all_results, os.path.join(results_dir, 'plots'))
    
    print("\nCreating summary table...")
    df = create_summary_table(all_results, os.path.join(results_dir, 'logs'))
    print(df)
    
    print(f"\nAll plots saved to {os.path.join(results_dir, 'plots')}")


if __name__ == "__main__":
    # Run analysis on results directory
    analyze_all_results('results')
