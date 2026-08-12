"""
Regenerate all output plots from saved training history and simulation results.
Uses the best history for stages 1-4 (20260730) and latest SAC (20260808_153957).
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')

PLOTS_DIR = Path("output/plots")
RESULTS_DIR = Path("output/results")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load(path):
    with open(path) as f:
        return json.load(f)

def smooth(values, window=5):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')

def plot_vit(hist, out_dir):
    """ViT training curve: loss + accuracy."""
    h = hist['vit']
    epochs = range(1, len(h['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, h['train_loss'], alpha=0.3, label='Train Loss')
    ax1.plot(epochs, smooth(h['train_loss']), lw=2, label='Train Loss (smooth)')
    if h.get('val_loss'):
        ax1.plot(epochs, h['val_loss'], alpha=0.3, label='Val Loss')
        ax1.plot(epochs, smooth(h['val_loss']), lw=2, label='Val Loss (smooth)')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='ViT - Training Loss')
    ax1.legend(); ax1.grid(True)
    if h.get('val_acc'):
        ax2.plot(epochs, h['val_acc'], alpha=0.3, label='Val Accuracy')
        ax2.plot(epochs, smooth(h['val_acc']), lw=2, label='Val Accuracy (smooth)')
        ax2.axhline(max(h['val_acc']), ls='--', color='green', alpha=0.5,
                    label=f'Peak: {max(h["val_acc"]):.4f}')
        ax2.set(xlabel='Epoch', ylabel='Accuracy', title='ViT - Validation Accuracy')
        ax2.legend(); ax2.grid(True)
    plt.suptitle('ViT Road Surface Classification Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"vit_training_curve_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")

def plot_temporal(hist, out_dir):
    """Temporal LSTM training curve."""
    h = hist['temporal']
    epochs = range(1, len(h['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, h['train_loss'], alpha=0.3, label='Train Loss')
    ax1.plot(epochs, smooth(h['train_loss']), lw=2, label='Train Loss (smooth)')
    if h.get('val_loss'):
        ax1.plot(epochs, h['val_loss'], alpha=0.3, label='Val Loss')
        ax1.plot(epochs, smooth(h['val_loss']), lw=2, label='Val Loss (smooth)')
    ax1.set(xlabel='Epoch', ylabel='Loss (MSE)', title='Temporal Network - Training Loss')
    ax1.legend(); ax1.grid(True)
    if h.get('val_mae'):
        ax2.plot(epochs, h['val_mae'], alpha=0.3, label='Val MAE')
        ax2.plot(epochs, smooth(h['val_mae']), lw=2, label='Val MAE (smooth)')
        ax2.set(xlabel='Epoch', ylabel='MAE', title='Temporal Network - Validation MAE')
        ax2.legend(); ax2.grid(True)
    plt.suptitle('Temporal Network (LSTM) - CAN Signal Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"temporal_network_training_curve_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")

def plot_fusion(hist, out_dir):
    """Fusion network training curve."""
    h = hist['fusion']
    epochs = range(1, len(h['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(epochs, h['train_loss'], alpha=0.3, label='Train Loss')
    ax1.plot(epochs, smooth(h['train_loss']), lw=2, label='Train Loss (smooth)')
    if h.get('val_loss'):
        ax1.plot(epochs, h['val_loss'], alpha=0.3, label='Val Loss')
        ax1.plot(epochs, smooth(h['val_loss']), lw=2, label='Val Loss (smooth)')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Fusion Network - Training Loss')
    ax1.legend(); ax1.grid(True)
    if h.get('val_mae'):
        ax2.plot(epochs, h['val_mae'], alpha=0.3, label='Val MAE')
        ax2.plot(epochs, smooth(h['val_mae']), lw=2, label='Val MAE (smooth)')
        ax2.set(xlabel='Epoch', ylabel='MAE', title='Fusion Network - Validation MAE')
        ax2.legend(); ax2.grid(True)
    plt.suptitle('Cross-Modal Attention Fusion Network Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"fusion_network_training_curve_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")

def plot_pinn(hist, out_dir):
    """PINN training curve with physics loss components."""
    h = hist['pinn']
    epochs = range(1, len(h['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Total loss
    axes[0].plot(epochs, h['train_loss'], alpha=0.3, label='Train Loss')
    axes[0].plot(epochs, smooth(h['train_loss']), lw=2, label='Train Loss (smooth)')
    if h.get('val_loss'):
        axes[0].plot(epochs, h['val_loss'], alpha=0.3, label='Val Loss')
        axes[0].plot(epochs, smooth(h['val_loss']), lw=2, label='Val Loss (smooth)')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='PINN - Total Loss')
    axes[0].legend(); axes[0].grid(True)
    # Physics loss
    if h.get('physics_loss'):
        axes[1].plot(epochs, h['physics_loss'], alpha=0.3, label='Physics Loss', color='orange')
        axes[1].plot(epochs, smooth(h['physics_loss']), lw=2, color='orange', label='Physics (smooth)')
        axes[1].set(xlabel='Epoch', ylabel='Physics Loss', title='PINN - Physics Constraint Loss')
        axes[1].legend(); axes[1].grid(True)
    # MAE
    if h.get('val_mae'):
        axes[2].plot(epochs, h['val_mae'], alpha=0.3, label='Val MAE', color='green')
        axes[2].plot(epochs, smooth(h['val_mae']), lw=2, color='green', label='Val MAE (smooth)')
        axes[2].axhline(min(h['val_mae']), ls='--', color='red', alpha=0.5,
                        label=f'Best: {min(h["val_mae"]):.4f}')
        axes[2].set(xlabel='Epoch', ylabel='MAE', title='PINN - Friction Estimation MAE')
        axes[2].legend(); axes[2].grid(True)
    plt.suptitle('PINN - Physics-Informed Friction Estimation Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"pinn_training_loss_components_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")

def plot_sac(hist_full, hist_sac, out_dir):
    """SAC reward curve — full run + latest SAC-only retrain side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: best full run (20260730)
    rewards_full = hist_full['sac']['reward']
    ep_full = range(1, len(rewards_full) + 1)
    axes[0].plot(ep_full, rewards_full, alpha=0.35, lw=1, color='steelblue', label='Episode Reward')
    axes[0].plot(ep_full, smooth(rewards_full, 3), lw=2, color='steelblue', label='Reward (smooth)')
    axes[0].axhline(max(rewards_full), ls='--', color='green', alpha=0.7,
                    label=f'Best: {max(rewards_full):.2f}')
    axes[0].axhline(0, ls=':', color='gray', alpha=0.5, label='Zero baseline')
    axes[0].set(xlabel='Epoch', ylabel='Average Reward', title='SAC Training (Full Run, 20260730)',
                ylim=(min(rewards_full)*1.1, max(rewards_full)*1.1 if max(rewards_full)>0 else 50))
    axes[0].legend(); axes[0].grid(True)

    # Right: latest SAC retrain (20260808)
    rewards_new = hist_sac['sac']['reward']
    ep_new = range(1, len(rewards_new) + 1)
    axes[1].plot(ep_new, rewards_new, alpha=0.35, lw=1, color='darkorange', label='Episode Reward')
    axes[1].plot(ep_new, smooth(rewards_new, 3), lw=2, color='darkorange', label='Reward (smooth)')
    best_new = max(rewards_new)
    best_ep = rewards_new.index(best_new) + 1
    axes[1].axhline(best_new, ls='--', color='green', alpha=0.7,
                    label=f'Best: {best_new:.2f} (ep {best_ep})')
    axes[1].axhline(0, ls=':', color='gray', alpha=0.5, label='Zero baseline')
    # Mark convergence band
    last_20 = rewards_new[-20:]
    axes[1].axhspan(min(last_20), max(last_20), alpha=0.1, color='orange',
                    label=f'Final band: [{min(last_20):.1f}, {max(last_20):.1f}]')
    axes[1].set(xlabel='Epoch', ylabel='Average Reward',
                title='SAC Retrain - Aug 8, 2026 (output/models/20260808_153957)',
                ylim=(min(rewards_new)*1.1, max(rewards_new)*1.1 if max(rewards_new)>0 else 50))
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle('SAC Braking Policy — Reward Progression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"sac_training_reward_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")

def _val(v):
    """Extract scalar from a plain value or a {'mean':...} stats dict."""
    if isinstance(v, dict):
        return v.get('mean', v.get('values', [0])[0])
    return float(v) if v is not None else 0.0

def plot_braking_simulation(sim_results, out_dir):
    """Regenerate the braking simulation plot from simulation_results.json."""
    surfaces = list(sim_results.keys())
    colors = {'dry': '#2196F3', 'wet': '#4CAF50', 'icy': '#9C27B0', 'rough': '#FF9800'}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Subplot 1: Velocity profiles from sampled data
    ax = axes[0, 0]
    for surf in surfaces:
        r = sim_results[surf]
        t = np.asarray(r.get('sample_times', []), dtype=float)
        v = np.asarray(r.get('sample_velocities', []), dtype=float)
        mu = _val(r.get('true_mu', 0.5))
        if len(t) and len(v):
            ax.plot(t, v, lw=2.5, color=colors.get(surf, 'gray'),
                    label=f'{surf.capitalize()} (μ={mu:.1f})')
    ax.set(xlabel='Time [s]', ylabel='Velocity [m/s]', title='Velocity Profiles by Surface')
    ax.legend(); ax.grid(True)

    # Subplot 2: Stopping distance bar chart
    ax = axes[0, 1]
    surf_names = [s.capitalize() for s in surfaces]
    stop_dists = [_val(sim_results[s].get('stopping_distance', 0)) for s in surfaces]
    theory_dists = []
    for s in surfaces:
        r = sim_results[s]
        mu = _val(r.get('true_mu', 0.5))
        v0 = _val(r.get('initial_velocity', 20.0))
        theory_dists.append(v0**2 / (2 * max(mu, 0.01) * 9.81))
    x = np.arange(len(surf_names))
    bars = ax.bar(x - 0.2, stop_dists, 0.4, label='Actual (SAC)',
                  color=[colors.get(s, 'gray') for s in surfaces])
    ax.bar(x + 0.2, theory_dists, 0.4, label='Theory (physics)', alpha=0.5, color='gray')
    ax.set_xticks(x); ax.set_xticklabels(surf_names)
    ax.set(ylabel='Distance [m]', title='Stopping Distance: Actual vs Theory')
    ax.legend(); ax.grid(True, axis='y')
    for bar, d in zip(bars, stop_dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{d:.1f}m', ha='center', fontsize=10, fontweight='bold')

    # Subplot 3: Friction coefficient per surface
    ax = axes[1, 0]
    for surf in surfaces:
        r = sim_results[surf]
        mu = _val(r.get('true_mu', 0.5))
        ax.barh(surf.capitalize(), mu, color=colors.get(surf, 'gray'), alpha=0.8)
        ax.text(mu + 0.01, surf.capitalize(), f'μ={mu:.2f}', va='center', fontsize=11)
    ax.set(xlabel='Friction Coefficient (μ)', title='Road Surface Friction Coefficients')
    ax.set_xlim(0, 1.15); ax.grid(True, axis='x')

    # Subplot 4: Performance summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [['Surface', 'μ', 'Stop Dist (m)', 'Stop Time (s)', 'Max Jerk']]
    for surf in surfaces:
        r = sim_results[surf]
        table_data.append([
            surf.capitalize(),
            f"{_val(r.get('true_mu', 0)):.2f}",
            f"{_val(r.get('stopping_distance', 0)):.1f}",
            f"{_val(r.get('stopping_time', 0)):.1f}",
            f"{_val(r.get('max_jerk', 0)):.3f}",
        ])
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    ax.set_title('Braking Performance Summary — SAC Policy (Aug 2026)', pad=20, fontweight='bold')

    plt.suptitle('SAC Adaptive Braking — Simulation Results\n(Model: output/models/20260808)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "braking_simulation.png", dpi=150, bbox_inches='tight')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(out_dir / f"braking_simulation_{ts}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ braking_simulation.png + timestamped copy")

def plot_all_stages_summary(hist_vit_full, hist_sac_new, out_dir):
    """Single overview figure showing all 5 stages on one canvas."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    stages = [
        ('vit', 'val_acc', 'ViT - Val Accuracy', 'steelblue', gs[0, 0]),
        ('temporal', 'val_mae', 'Temporal - Val MAE', 'teal', gs[0, 1]),
        ('fusion', 'val_mae', 'Fusion - Val MAE', 'darkcyan', gs[0, 2]),
        ('pinn', 'val_mae', 'PINN - Val MAE (Friction)', 'coral', gs[1, 0]),
    ]

    for stage, metric, title, color, loc in stages:
        ax = fig.add_subplot(loc)
        h = hist_vit_full.get(stage, {})
        data = h.get(metric, h.get('train_loss', []))
        if data:
            ep = range(1, len(data) + 1)
            ax.plot(ep, data, alpha=0.3, color=color)
            ax.plot(ep, smooth(data), lw=2.5, color=color, label=metric)
            best = min(data) if 'mae' in metric or 'loss' in metric else max(data)
            ax.axhline(best, ls='--', color='green', alpha=0.6,
                       label=f'Best: {best:.4f}')
            ax.set(xlabel='Epoch', title=title)
            ax.legend(fontsize=9); ax.grid(True)

    # SAC reward (latest run)
    ax_sac = fig.add_subplot(gs[1, 1])
    rewards = hist_sac_new['sac']['reward']
    ep = range(1, len(rewards) + 1)
    ax_sac.plot(ep, rewards, alpha=0.3, color='darkorange')
    ax_sac.plot(ep, smooth(rewards, 3), lw=2.5, color='darkorange', label='Reward')
    ax_sac.axhline(max(rewards), ls='--', color='green', alpha=0.6,
                   label=f'Best: {max(rewards):.2f}')
    ax_sac.set(xlabel='Epoch', title='SAC - Training Reward (Aug 8, 2026)')
    ax_sac.legend(fontsize=9); ax_sac.grid(True)

    # Metrics summary
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')
    summary = [
        ['Stage', 'Key Metric', 'Value'],
        ['ViT', 'Accuracy', '69.27%'],
        ['ViT', 'Macro-F1', '0.4233'],
        ['LSTM', 'Recon. MSE', '~0.0015'],
        ['Fusion', 'R² (contrib.)', '+0.7%'],
        ['PINN', 'RMSE', '0.1368'],
        ['PINN', 'R²', '0.5931'],
        ['SAC', 'Stop Success', '100%'],
        ['SAC', 'Best Reward', f'{max(rewards):.2f}'],
    ]
    tbl = ax_summary.table(cellText=summary[1:], colLabels=summary[0],
                           cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.6)
    ax_summary.set_title('Final Performance Summary', pad=20, fontweight='bold')

    fig.suptitle('Intelligent Multi-Modal Braking System — All Training Stages',
                 fontsize=15, fontweight='bold', y=1.01)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = out_dir / f"all_stages_overview_{ts}.png"
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {p}")


if __name__ == '__main__':
    print("\n🔄 Regenerating all plots from training history...\n")

    hist_full = load('output/training_history_20260730_202001.json')
    hist_sac  = load('output/training_history_20260808_153957.json')
    sim_data  = load('output/results/simulation_results.json') if Path('output/results/simulation_results.json').exists() else None

    print("📈 Stage plots:")
    plot_vit(hist_full, PLOTS_DIR)
    plot_temporal(hist_full, PLOTS_DIR)
    plot_fusion(hist_full, PLOTS_DIR)
    plot_pinn(hist_full, PLOTS_DIR)

    print("\n🤖 SAC reward plot:")
    plot_sac(hist_full, hist_sac, PLOTS_DIR)

    if sim_data:
        print("\n🚗 Braking simulation plot:")
        plot_braking_simulation(sim_data, PLOTS_DIR)
    else:
        print("\n⚠️  No simulation_results.json found — skipping braking plot")

    print("\n📊 All-stages overview:")
    plot_all_stages_summary(hist_full, hist_sac, PLOTS_DIR)

    print("\n✅ Done! All plots saved to output/plots/")
    print(f"   Total plots: {len(list(PLOTS_DIR.glob('*.png')))}")
