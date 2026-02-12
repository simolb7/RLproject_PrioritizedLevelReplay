#!/usr/bin/env python3
"""
Analyze training logs to diagnose NaN issues.
Usage: python diagnose_nan.py path/to/train_log.csv
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_log(log_path: str):
    """Analyze training log for NaN patterns."""
    print(f"Analyzing: {log_path}\n")
    
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"ERROR reading log: {e}")
        return
    
    print("="*60)
    print("LOG STATISTICS")
    print("="*60)
    print(f"Total updates: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check for NaNs in each column
    print("NaN counts by column:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            first_nan_idx = df[col].isna().idxmax()
            print(f"  {col:20s}: {nan_count:4d} NaNs (first at update {df.loc[first_nan_idx, 'update']})")
        else:
            print(f"  {col:20s}: 0 NaNs")
    
    print("\n" + "="*60)
    print("CRITICAL EVENTS")
    print("="*60)
    
    # Find when NaNs first appear
    score_nans = df['score'].isna()
    ret_nans = df['mean_ep_return'].isna()
    
    if score_nans.any():
        first_score_nan = df[score_nans].iloc[0]
        print(f"\n❌ First score NaN at update {first_score_nan['update']}")
        print(f"   Level: {first_score_nan['level_id']}")
        print(f"   Mode: {first_score_nan['mode']}")
        
        # Show context around first NaN
        idx = score_nans.idxmax()
        print("\n   Context (5 updates before NaN):")
        context = df.iloc[max(0, idx-5):idx+1]
        print(context[['update', 'mode', 'level_id', 'score', 'mean_ep_return']])
    
    if ret_nans.any() and not ret_nans.all():
        first_ret_nan = df[ret_nans].iloc[0]
        print(f"\n⚠️  First return NaN at update {first_ret_nan['update']}")
    
    # Check if scores are exploding before NaN
    if not score_nans.all():
        valid_scores = df[~score_nans]['score']
        print(f"\nScore statistics (before NaN):")
        print(f"  Mean: {valid_scores.mean():.6f}")
        print(f"  Std:  {valid_scores.std():.6f}")
        print(f"  Max:  {valid_scores.max():.6f}")
        print(f"  Min:  {valid_scores.min():.6f}")
        
        if valid_scores.max() > 100:
            print("  ⚠️  WARNING: Scores > 100 detected (possible explosion)")
        
        # Find rapid score changes
        score_diff = valid_scores.diff().abs()
        large_jumps = score_diff[score_diff > 10]
        if len(large_jumps) > 0:
            print(f"\n  ⚠️  {len(large_jumps)} large score jumps (>10) detected")
            print(f"     Largest jump: {score_diff.max():.2f}")
    
    # Check episode returns
    if not ret_nans.all():
        valid_rets = df[~ret_nans]['mean_ep_return']
        print(f"\nReturn statistics:")
        print(f"  Mean: {valid_rets.mean():.2f}")
        print(f"  Std:  {valid_rets.std():.2f}")
        print(f"  Max:  {valid_rets.max():.2f}")
        print(f"  Min:  {valid_rets.min():.2f}")
    
    # Check buffer growth
    if 'buffer_size' in df.columns:
        print(f"\nBuffer statistics:")
        print(f"  Final size: {df['buffer_size'].iloc[-1]}")
        print(f"  Max size:   {df['buffer_size'].max()}")
    
    if 'replay_ratio' in df.columns:
        valid_replay = df[~df['replay_ratio'].isna()]['replay_ratio']
        if len(valid_replay) > 0:
            print(f"  Final replay ratio: {valid_replay.iloc[-1]:.3f}")
    
    # Visualization
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score over time
    ax = axes[0, 0]
    ax.plot(df['update'], df['score'], 'b-', alpha=0.7, label='Score')
    ax.set_xlabel('Update')
    ax.set_ylabel('Score')
    ax.set_title('Level Score Over Time')
    ax.grid(True, alpha=0.3)
    if score_nans.any():
        first_nan = score_nans.idxmax()
        ax.axvline(df.loc[first_nan, 'update'], color='red', linestyle='--', 
                   label=f'First NaN (update {df.loc[first_nan, "update"]})')
    ax.legend()
    
    # Plot 2: Returns over time
    ax = axes[0, 1]
    ax.plot(df['update'], df['mean_ep_return'], 'g-', alpha=0.7, label='Mean Return')
    ax.set_xlabel('Update')
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Returns Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Buffer size and replay ratio
    ax = axes[1, 0]
    if 'buffer_size' in df.columns:
        ax.plot(df['update'], df['buffer_size'], 'purple', label='Buffer Size')
    ax.set_xlabel('Update')
    ax.set_ylabel('Buffer Size')
    ax.set_title('Buffer Growth')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Replay ratio
    ax = axes[1, 1]
    if 'replay_ratio' in df.columns:
        ax.plot(df['update'], df['replay_ratio'], 'orange', label='Replay Ratio')
    ax.set_xlabel('Update')
    ax.set_ylabel('Replay Ratio')
    ax.set_title('Replay vs New Levels')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = log_path.replace('.csv', '_diagnostics.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    
    # Summary diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if score_nans.any():
        first_nan_update = df[score_nans].iloc[0]['update']
        
        if first_nan_update < 50:
            print("❌ IMMEDIATE CRASH (NaN in first 50 updates)")
            print("   Likely causes:")
            print("   1. Bug in score calculation")
            print("   2. Observation normalization issue")
            print("   3. Incompatible PyTorch/CUDA versions")
        elif first_nan_update < 500:
            print("⚠️  EARLY CRASH (NaN before update 500)")
            print("   Likely causes:")
            print("   1. Exploding gradients")
            print("   2. Learning rate too high")
            print("   3. Advantage normalization issue")
        else:
            print("⚠️  LATE CRASH (NaN after update 500)")
            print("   Likely causes:")
            print("   1. Policy collapse on specific hard level")
            print("   2. Value function divergence")
            print("   3. Buffer selecting pathological levels")
    else:
        print("✓ No NaNs detected in scores")
    
    print("\nRecommended fixes:")
    print("1. Use train_plr_stable.py with NaN detection")
    print("2. Reduce learning rate to 0.00025")
    print("3. Use config_stable.yaml with temperature=0.2")
    print("4. Check GPU/CUDA compatibility")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_nan.py path/to/train_log.csv")
        sys.exit(1)
    
    analyze_log(sys.argv[1])