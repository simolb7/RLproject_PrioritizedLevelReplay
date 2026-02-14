#!/usr/bin/env python3
"""
Analyze and compare PLR vs Baseline training results across Procgen environments.

Usage:
    python analyze_results.py --log-dir outputs/
    python analyze_results.py --log-dir outputs/ --games coinrun bigfish
    python analyze_results.py --log-dir outputs/ --save-path results/analysis.png
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter1d


def find_log_files(log_dir: str, env_name: str = None) -> Dict[str, Dict[str, Path]]:
    """
    Find all training logs organized by environment and method.
    
    Returns:
        {
            'coinrun': {
                'plr': Path('outputs/coinrun_plr_123/train_log.csv'),
                'baseline': Path('outputs/coinrun_baseline_456/train_log.csv')
            },
            ...
        }
    """
    log_dir = Path(log_dir)
    results = {}
    
    # Search for all train_log.csv files
    for log_file in log_dir.rglob("train_log.csv"):
        run_name = log_file.parent.name
        
        # Extract environment and method from run name
        # Expected format: envname_method_timestamp
        parts = run_name.split('_')
        if len(parts) < 2:
            continue
        
        # Find method (plr or baseline)
        method = None
        for i, part in enumerate(parts):
            if part in ['plr', 'baseline']:
                method = part
                env = '_'.join(parts[:i])
                break
        
        if method is None:
            continue
        
        # Filter by environment if specified
        if env_name and env != env_name:
            continue
        
        if env not in results:
            results[env] = {}
        
        # Keep the most recent run for each method
        if method not in results[env]:
            results[env][method] = log_file
        else:
            # Compare timestamps (last part of run_name)
            old_timestamp = int(results[env][method].parent.name.split('_')[-1])
            new_timestamp = int(run_name.split('_')[-1])
            if new_timestamp > old_timestamp:
                results[env][method] = log_file
    
    return results


def load_training_data(log_path: Path) -> pd.DataFrame:
    """Load and preprocess training log."""
    df = pd.read_csv(log_path)
    
    # Handle NaN values in mean_ep_return
    df['mean_ep_return'] = df['mean_ep_return'].fillna(method='ffill')
    
    return df


def smooth_curve(data: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply uniform smoothing to curve."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


def compute_statistics(df: pd.DataFrame, window: int = 100) -> Dict:
    """Compute training statistics."""
    returns = df['mean_ep_return'].dropna().values
    scores = df['score'].values
    
    if len(returns) == 0:
        return {
            'final_return_mean': 0.0,
            'final_return_std': 0.0,
            'max_return': 0.0,
            'mean_score': 0.0,
            'score_std': 0.0
        }
    
    # Final performance (last 10% of training)
    final_window = max(1, len(returns) // 10)
    final_returns = returns[-final_window:]
    
    return {
        'final_return_mean': float(np.mean(final_returns)),
        'final_return_std': float(np.std(final_returns)),
        'max_return': float(np.max(returns)),
        'mean_return': float(np.mean(returns)),
        'mean_score': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'convergence_speed': compute_convergence_speed(returns)
    }


def compute_convergence_speed(returns: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute how many updates it takes to reach threshold of max performance.
    
    Returns:
        Number of updates to reach threshold, or -1 if never reached.
    """
    if len(returns) == 0:
        return -1
    
    max_return = np.max(returns)
    target = threshold * max_return
    
    for i, r in enumerate(returns):
        if r >= target:
            return i
    
    return -1


def plot_training_curves(
    results: Dict[str, Dict[str, Path]],
    save_path: str = None,
    smooth_window: int = 50,
    figsize: Tuple[int, int] = None
):
    """
    Plot training curves comparing PLR vs Baseline for all environments.
    """
    n_envs = len(results)
    
    if figsize is None:
        figsize = (16, 4 * n_envs)
    
    fig, axes = plt.subplots(n_envs, 3, figsize=figsize)
    if n_envs == 1:
        axes = axes.reshape(1, -1)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'plr': '#2E86AB', 'baseline': '#A23B72'}
    
    for idx, (env_name, methods) in enumerate(sorted(results.items())):
        ax_return = axes[idx, 0]
        ax_score = axes[idx, 1]
        ax_buffer = axes[idx, 2]
        
        stats_text = []
        
        for method_name, log_path in methods.items():
            df = load_training_data(log_path)
            color = colors.get(method_name, 'gray')
            
            # Plot 1: Episode Returns
            returns = df['mean_ep_return'].dropna().values
            updates = df['update'].values[:len(returns)]
            
            if len(returns) > 0:
                # Raw data (transparent)
                ax_return.plot(updates, returns, alpha=0.2, color=color)
                # Smoothed data
                smoothed = smooth_curve(returns, smooth_window)
                ax_return.plot(updates, smoothed, label=method_name.upper(), 
                             color=color, linewidth=2)
            
            # Plot 2: Level Scores
            scores = df['score'].values
            updates_scores = df['update'].values[:len(scores)]
            
            if len(scores) > 0:
                smoothed_scores = smooth_curve(scores, smooth_window)
                ax_score.plot(updates_scores, smoothed_scores, 
                            label=method_name.upper(), color=color, linewidth=2)
            
            # Plot 3: Buffer size (PLR only)
            if method_name == 'plr' and 'buffer_size' in df.columns:
                buffer_sizes = df['buffer_size'].values
                updates_buffer = df['update'].values[:len(buffer_sizes)]
                ax_buffer.plot(updates_buffer, buffer_sizes, 
                             color=color, linewidth=2, label='Buffer Size')
                
                # Replay ratio (secondary y-axis)
                if 'replay_ratio' in df.columns:
                    ax_buffer_twin = ax_buffer.twinx()
                    replay_ratios = df['replay_ratio'].values
                    smoothed_replay = smooth_curve(replay_ratios, smooth_window)
                    ax_buffer_twin.plot(updates_buffer, smoothed_replay, 
                                       color='#F18F01', linewidth=2, 
                                       linestyle='--', label='Replay Ratio')
                    ax_buffer_twin.set_ylabel('Replay Ratio', color='#F18F01')
                    ax_buffer_twin.tick_params(axis='y', labelcolor='#F18F01')
                    ax_buffer_twin.set_ylim([0, 1])
                    ax_buffer_twin.legend(loc='upper right')
            
            # Compute statistics
            stats = compute_statistics(df)
            stats_text.append(f"{method_name.upper()}: "
                            f"Final={stats['final_return_mean']:.1f}±{stats['final_return_std']:.1f}, "
                            f"Max={stats['max_return']:.1f}")
        
        # Formatting
        ax_return.set_title(f"{env_name.upper()} - Episode Returns", fontsize=14, fontweight='bold')
        ax_return.set_xlabel("Update")
        ax_return.set_ylabel("Mean Episode Return")
        ax_return.legend(loc='best')
        ax_return.grid(True, alpha=0.3)
        
        ax_score.set_title(f"{env_name.upper()} - Level Scores", fontsize=14, fontweight='bold')
        ax_score.set_xlabel("Update")
        ax_score.set_ylabel("Level Score (GAE magnitude)")
        ax_score.legend(loc='best')
        ax_score.grid(True, alpha=0.3)
        
        ax_buffer.set_title(f"{env_name.upper()} - PLR Buffer Dynamics", fontsize=14, fontweight='bold')
        ax_buffer.set_xlabel("Update")
        ax_buffer.set_ylabel("Buffer Size")
        ax_buffer.legend(loc='upper left')
        ax_buffer.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def create_comparison_table(results: Dict[str, Dict[str, Path]]) -> pd.DataFrame:
    """
    Create a comparison table of final performance metrics.
    """
    rows = []
    
    for env_name, methods in sorted(results.items()):
        row = {'Environment': env_name}
        
        for method_name, log_path in methods.items():
            df = load_training_data(log_path)
            stats = compute_statistics(df)
            
            row[f'{method_name}_final'] = f"{stats['final_return_mean']:.1f} ± {stats['final_return_std']:.1f}"
            row[f'{method_name}_max'] = f"{stats['max_return']:.1f}"
            row[f'{method_name}_convergence'] = stats['convergence_speed']
        
        # Compute improvement
        if 'plr' in methods and 'baseline' in methods:
            plr_df = load_training_data(methods['plr'])
            baseline_df = load_training_data(methods['baseline'])
            
            plr_stats = compute_statistics(plr_df)
            baseline_stats = compute_statistics(baseline_df)
            
            improvement = ((plr_stats['final_return_mean'] - baseline_stats['final_return_mean']) 
                          / (baseline_stats['final_return_mean'] + 1e-6) * 100)
            row['improvement_%'] = f"{improvement:.1f}%"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_level_distribution(results: Dict[str, Dict[str, Path]], save_path: str = None):
    """
    Plot level sampling distribution for PLR runs.
    """
    plr_envs = {env: methods['plr'] for env, methods in results.items() if 'plr' in methods}
    
    if not plr_envs:
        print("No PLR runs found for level distribution plot")
        return
    
    n_envs = len(plr_envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    
    if n_envs == 1:
        axes = [axes]
    
    for ax, (env_name, log_path) in zip(axes, plr_envs.items()):
        df = load_training_data(log_path)
        
        # Count level visits
        level_counts = df['level_id'].value_counts().sort_index()
        
        # Plot
        ax.bar(level_counts.index, level_counts.values, alpha=0.7, color='#2E86AB')
        ax.set_title(f"{env_name.upper()} - Level Visit Distribution", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Level ID")
        ax.set_ylabel("Number of Visits")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = (f"Unique levels: {len(level_counts)}\n"
                     f"Mean visits: {level_counts.mean():.1f}\n"
                     f"Std visits: {level_counts.std():.1f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved level distribution plot to {save_path}")
    else:
        plt.show()


def perform_statistical_test(results: Dict[str, Dict[str, Path]]) -> pd.DataFrame:
    """
    Perform statistical tests comparing PLR vs Baseline.
    """
    test_results = []
    
    for env_name, methods in sorted(results.items()):
        if 'plr' not in methods or 'baseline' not in methods:
            continue
        
        plr_df = load_training_data(methods['plr'])
        baseline_df = load_training_data(methods['baseline'])
        
        # Get final returns (last 10%)
        plr_returns = plr_df['mean_ep_return'].dropna().values
        baseline_returns = baseline_df['mean_ep_return'].dropna().values
        
        if len(plr_returns) == 0 or len(baseline_returns) == 0:
            continue
        
        final_window = max(10, len(plr_returns) // 10)
        plr_final = plr_returns[-final_window:]
        baseline_final = baseline_returns[-final_window:]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(plr_final, baseline_final)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(plr_final) + np.var(baseline_final)) / 2)
        cohens_d = (np.mean(plr_final) - np.mean(baseline_final)) / (pooled_std + 1e-6)
        
        test_results.append({
            'Environment': env_name,
            'PLR_mean': np.mean(plr_final),
            'Baseline_mean': np.mean(baseline_final),
            'Difference': np.mean(plr_final) - np.mean(baseline_final),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    return pd.DataFrame(test_results)


def generate_report(results: Dict[str, Dict[str, Path]], output_dir: str = "analysis_results"):
    """
    Generate a comprehensive analysis report with all plots and tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PLR vs Baseline Analysis Report")
    print("="*60)
    
    # 1. Comparison Table
    print("\n1. Final Performance Comparison")
    print("-"*60)
    comparison_table = create_comparison_table(results)
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(output_dir / "comparison_table.csv", index=False)
    print(f"\nSaved to: {output_dir / 'comparison_table.csv'}")
    
    # 2. Statistical Tests
    print("\n2. Statistical Significance Tests")
    print("-"*60)
    stat_tests = perform_statistical_test(results)
    if not stat_tests.empty:
        print(stat_tests.to_string(index=False))
        stat_tests.to_csv(output_dir / "statistical_tests.csv", index=False)
        print(f"\nSaved to: {output_dir / 'statistical_tests.csv'}")
    else:
        print("Not enough data for statistical tests")
    
    # 3. Training Curves
    print("\n3. Generating training curves...")
    plot_training_curves(results, save_path=output_dir / "training_curves.png")
    
    # 4. Level Distribution
    print("\n4. Generating level distribution plots...")
    plot_level_distribution(results, save_path=output_dir / "level_distribution.png")
    
    print("\n" + "="*60)
    print(f"Report generation complete! Results saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare PLR vs Baseline training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all games in outputs directory
  python analyze_results.py --log-dir outputs/
  
  # Analyze specific games
  python analyze_results.py --log-dir outputs/ --games coinrun bigfish
  
  # Generate full report
  python analyze_results.py --log-dir outputs/ --report
  
  # Save to custom location
  python analyze_results.py --log-dir outputs/ --save-path results/curves.png
        """
    )
    
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--games", nargs="+", default=None,
                       help="Specific games to analyze (default: all)")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save plots (default: show interactive)")
    parser.add_argument("--smooth-window", type=int, default=50,
                       help="Smoothing window size (default: 50)")
    parser.add_argument("--report", action="store_true",
                       help="Generate full analysis report")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                       help="Output directory for report (default: analysis_results)")
    
    args = parser.parse_args()
    
    # Find all log files
    print(f"Searching for logs in: {args.log_dir}")
    all_results = find_log_files(args.log_dir)
    
    if not all_results:
        print(f"No training logs found in {args.log_dir}")
        print("Expected directory structure: outputs/envname_method_timestamp/train_log.csv")
        return
    
    # Filter by games if specified
    if args.games:
        all_results = {k: v for k, v in all_results.items() if k in args.games}
    
    if not all_results:
        print(f"No logs found for specified games: {args.games}")
        return
    
    # Print found logs
    print(f"\nFound logs for {len(all_results)} environment(s):")
    for env_name, methods in sorted(all_results.items()):
        print(f"  {env_name}:")
        for method, path in methods.items():
            print(f"    - {method}: {path}")
    
    # Generate report or single plot
    if args.report:
        generate_report(all_results, args.output_dir)
    else:
        plot_training_curves(
            all_results, 
            save_path=args.save_path,
            smooth_window=args.smooth_window
        )


if __name__ == "__main__":
    main()