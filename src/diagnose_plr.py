#!/usr/bin/env python3
"""
Quick diagnostic script per capire perché PLR non funziona.
Usa: python diagnose_plr.py runs/coinrun_plr_<timestamp>/train_log.csv
"""

import sys
import pandas as pd
import numpy as np

def diagnose_plr(log_path):
    print("="*70)
    print("🔍 PLR DIAGNOSTIC REPORT")
    print("="*70)
    
    # Carica log
    df = pd.read_csv(log_path)
    df_clean = df.dropna(subset=['score', 'mean_ep_return'])
    
    print(f"\n📊 BASIC INFO")
    print(f"Total updates: {len(df)}")
    print(f"Updates with valid data: {len(df_clean)}")
    
    # === 1. MODE DISTRIBUTION ===
    print(f"\n🎲 SAMPLING MODE DISTRIBUTION")
    mode_counts = df['mode'].value_counts()
    total = len(df)
    
    for mode, count in mode_counts.items():
        pct = (count / total) * 100
        print(f"  {mode:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Check replay ratio
    warmup_count = mode_counts.get('warmup', 0)
    new_count = mode_counts.get('new', 0)
    replay_count = mode_counts.get('replay', 0)
    
    if warmup_count + new_count + replay_count > 0:
        replay_ratio = replay_count / (new_count + replay_count) if (new_count + replay_count) > 0 else 0
        print(f"\n  Replay Ratio (after warmup): {replay_ratio:.2%}")
        
        if replay_ratio < 0.3:
            print(f"  ⚠️  WARNING: Replay ratio is LOW! PLR might not be working.")
            print(f"      → Try: reduce p_new to 0.3 or 0.2")
        elif replay_ratio > 0.7:
            print(f"  ✅ Good replay ratio!")
    
    # === 2. SCORE ANALYSIS ===
    print(f"\n📈 SCORE DISTRIBUTION")
    scores = df_clean['score']
    
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {scores.median():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    
    # Check if using GAE vs value loss
    if scores.mean() < 0.1:
        print(f"\n  ❌ CRITICAL: Scores are too low ({scores.mean():.4f})")
        print(f"      This suggests you're using GAE instead of value loss!")
        print(f"      → Fix: score = float((ret_t - values_t).abs().mean().item())")
    elif scores.mean() > 0.15:
        print(f"  ✅ Scores look good (using value loss)")
    
    # Check score variance
    if scores.std() < 0.05:
        print(f"\n  ⚠️  WARNING: Low score variance ({scores.std():.4f})")
        print(f"      Levels might be too similar in difficulty.")
        print(f"      → Try: increase train_level_max or use different distribution_mode")
    
    # === 3. RETURN ANALYSIS ===
    print(f"\n🎯 EPISODE RETURN ANALYSIS")
    returns = df_clean['mean_ep_return']
    
    print(f"  Mean:   {returns.mean():.2f}")
    print(f"  Median: {returns.median():.2f}")
    print(f"  Std:    {returns.std():.2f}")
    print(f"  Min:    {returns.min():.2f}")
    print(f"  Max:    {returns.max():.2f}")
    
    # Check learning progress
    first_half = returns.iloc[:len(returns)//2].mean()
    second_half = returns.iloc[len(returns)//2:].mean()
    improvement = ((second_half / first_half) - 1) * 100
    
    print(f"\n  First half mean:  {first_half:.2f}")
    print(f"  Second half mean: {second_half:.2f}")
    print(f"  Improvement:      {improvement:+.1f}%")
    
    if improvement < 5:
        print(f"\n  ⚠️  WARNING: Low improvement ({improvement:.1f}%)")
        print(f"      Agent might not be learning much.")
        print(f"      → Try: longer training, easier levels, or check PPO hyperparams")
    
    # === 4. LEVEL DIVERSITY ===
    print(f"\n🗺️  LEVEL DIVERSITY")
    unique_levels = df['level_id'].nunique()
    total_levels_seen = len(df)
    
    print(f"  Unique levels seen: {unique_levels}")
    print(f"  Total updates:      {total_levels_seen}")
    print(f"  Diversity ratio:    {unique_levels/total_levels_seen:.2%}")
    
    # Most visited levels
    top_levels = df['level_id'].value_counts().head(10)
    print(f"\n  Top 10 most visited levels:")
    for i, (level, count) in enumerate(top_levels.items(), 1):
        pct = (count / total_levels_seen) * 100
        print(f"    {i:2d}. Level {level:5d}: {count:3d} visits ({pct:4.1f}%)")
    
    # Check if curriculum is working
    max_visit_pct = (top_levels.iloc[0] / total_levels_seen) * 100
    if max_visit_pct > 10:
        print(f"\n  ⚠️  Top level visited {max_visit_pct:.1f}% of the time")
        print(f"      → This might indicate strong curriculum (good!)")
        print(f"      → Or temperature is too low (overfitting to few levels)")
    
    # === 5. RECOMMENDATIONS ===
    print(f"\n💡 RECOMMENDATIONS")
    
    issues = []
    
    if replay_ratio < 0.3:
        issues.append("• LOW REPLAY RATIO: Reduce p_new to 0.2-0.3")
    
    if scores.mean() < 0.1:
        issues.append("• WRONG SCORE METRIC: Use value loss instead of GAE")
    
    if scores.std() < 0.05:
        issues.append("• LOW SCORE VARIANCE: Increase level diversity")
    
    if improvement < 5:
        issues.append("• LOW LEARNING: Increase training length or check hyperparams")
    
    if max_visit_pct > 15:
        issues.append("• POSSIBLE OVERFITTING: Increase temperature (0.1 or higher)")
    
    if len(issues) == 0:
        print("  ✅ No major issues detected!")
        print("  → If PLR still equals baseline, try:")
        print("    - Reduce temperature to 0.05 or 0.01")
        print("    - Increase staleness_coef to 0.3-0.5")
        print("    - Reduce p_new to 0.2-0.3")
    else:
        for issue in issues:
            print(f"  {issue}")
    
    print("\n" + "="*70)
    print("End of diagnostic report")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_plr.py <path_to_train_log.csv>")
        print("Example: python diagnose_plr.py runs/coinrun_plr_1234567/train_log.csv")
        sys.exit(1)
    
    log_path = sys.argv[1]
    diagnose_plr(log_path)