"""
Comprehensive Multi-Model Basin-wise Comparison
High-impact publication-ready figure for top-tier journals
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
MATCH_DISTANCE_THRESHOLD = 500

BASIN_INFO = {
    'NA': {'name': 'N. Atlantic', 'full': 'North Atlantic'},
    'EP': {'name': 'E. Pacific', 'full': 'East Pacific'},
    'WP': {'name': 'W. Pacific', 'full': 'West Pacific'},
    'NI': {'name': 'N. Indian', 'full': 'North Indian'},
    'SI': {'name': 'S. Indian', 'full': 'South Indian'},
    'SP': {'name': 'S. Pacific', 'full': 'South Pacific'},
}
BASIN_ORDER = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']

# Model configuration
MODEL_CONFIG = {
    'proposed': {'name': 'Proposed', 'color': '#E64B35', 'marker': 'o', 'file': 'proposed.jsonl'},
    'gpt': {'name': 'GPT-5.2', 'color': '#00A087', 'marker': 's', 'file': 'gpt.jsonl'},
    'claude': {'name': 'Gemini-3', 'color': '#4DBBD5', 'marker': '^', 'file': 'claude.jsonl'},
    'gemini': {'name': 'Claude-4.5', 'color': '#3C5488', 'marker': 'D', 'file': 'qwen235b.jsonl'},
    'weytcnet': {'name': 'WEY-TCNet', 'color': '#F39B7F', 'marker': 'v', 'file': 'weytcnet.jsonl'},
}
MODEL_ORDER = ['proposed', 'gpt', 'claude', 'gemini', 'weytcnet']

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ==================== Data Functions ====================
def load_jsonl(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def match_tcs(pred_tcs, gt_tcs):
    if not pred_tcs or not gt_tcs:
        return []
    cost_matrix = np.array([[haversine_distance(p['lat'], p['lon'], g['lat'], g['lon']) 
                            for g in gt_tcs] for p in pred_tcs])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(i, j, cost_matrix[i,j]) for i, j in zip(row_ind, col_ind) if cost_matrix[i,j] < MATCH_DISTANCE_THRESHOLD]

def extract_basin_metrics(records):
    basin_data = {b: {'distances': [], 'count_errors': [], 'tp': 0, 'fp': 0, 'fn': 0} for b in BASIN_ORDER}
    
    for record in records:
        basin = record.get('file', '').split('_')[-1].replace('.npy', '')
        if basin not in BASIN_ORDER:
            continue
        
        pred = record['pred']
        gt = record['gt']
        pred_count = pred.get('current_tc_count', 0)
        gt_count = gt.get('current_tc_count', 0)
        basin_data[basin]['count_errors'].append(pred_count - gt_count)
        
        pred_tcs = pred.get('current_tcs', []) or []
        gt_tcs = gt.get('current_tcs', []) or []
        matches = match_tcs(pred_tcs, gt_tcs)
        
        basin_data[basin]['tp'] += len(matches)
        basin_data[basin]['fp'] += max(0, len(pred_tcs) - len(matches))
        basin_data[basin]['fn'] += max(0, len(gt_tcs) - len(matches))
        basin_data[basin]['distances'].extend([m[2] for m in matches])
    
    for b in BASIN_ORDER:
        d = basin_data[b]
        d['pos_mae'] = np.mean(d['distances']) if d['distances'] else 0
        d['count_mae'] = np.mean(np.abs(d['count_errors'])) if d['count_errors'] else 0
        d['precision'] = d['tp'] / (d['tp'] + d['fp']) if (d['tp'] + d['fp']) > 0 else 0
        d['recall'] = d['tp'] / (d['tp'] + d['fn']) if (d['tp'] + d['fn']) > 0 else 0
        d['f1'] = 2 * d['precision'] * d['recall'] / (d['precision'] + d['recall']) if (d['precision'] + d['recall']) > 0 else 0
    
    return basin_data

# ==================== Visualization ====================
def plot_comprehensive_basin_comparison(all_basin_data, save_path):
    """Create a comprehensive multi-panel figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1, 0.9])
    
    # ========== Panel (a): Position MAE Grouped Bar ==========
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(BASIN_ORDER))
    width = 0.15
    
    for i, model in enumerate(MODEL_ORDER):
        vals = [all_basin_data[model][b]['pos_mae'] for b in BASIN_ORDER]
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, vals, width, label=MODEL_CONFIG[model]['name'],
                      color=MODEL_CONFIG[model]['color'], edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Position MAE (km)')
    ax1.set_title('(a) Position Error by Basin', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=30, ha='right')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.set_ylim(0, 200)
    
    # ========== Panel (b): F1 Score Grouped Bar ==========
    ax2 = fig.add_subplot(gs[0, 1])
    for i, model in enumerate(MODEL_ORDER):
        vals = [all_basin_data[model][b]['f1'] * 100 for b in BASIN_ORDER]
        offset = (i - 2) * width
        ax2.bar(x + offset, vals, width, label=MODEL_CONFIG[model]['name'],
               color=MODEL_CONFIG[model]['color'], edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_title('(b) Detection Performance by Basin', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=30, ha='right')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right', ncol=2, fontsize=8)
    
    # ========== Panel (c): Heatmap - Position MAE ==========
    ax3 = fig.add_subplot(gs[0, 2])
    mae_matrix = np.array([[all_basin_data[m][b]['pos_mae'] for b in BASIN_ORDER] for m in MODEL_ORDER])
    
    im = ax3.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto', vmin=30, vmax=180)
    ax3.set_xticks(range(len(BASIN_ORDER)))
    ax3.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=30, ha='right')
    ax3.set_yticks(range(len(MODEL_ORDER)))
    ax3.set_yticklabels([MODEL_CONFIG[m]['name'] for m in MODEL_ORDER])
    
    for i in range(len(MODEL_ORDER)):
        for j in range(len(BASIN_ORDER)):
            val = mae_matrix[i, j]
            color = 'white' if val > 100 else 'black'
            ax3.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9, 
                    color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('MAE (km)', fontsize=9)
    ax3.set_title('(c) Position MAE Heatmap', fontweight='bold')
    
    # ========== Panel (d): Radar Chart - Overall Performance ==========
    ax4 = fig.add_subplot(gs[1, 0], polar=True)
    categories = [BASIN_INFO[b]['name'] for b in BASIN_ORDER]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=9)
    
    # Normalize: higher F1 is better
    for model in MODEL_ORDER:
        f1_vals = [all_basin_data[model][b]['f1'] for b in BASIN_ORDER]
        f1_vals += f1_vals[:1]
        ax4.plot(angles, f1_vals, 'o-', linewidth=1.5, label=MODEL_CONFIG[model]['name'],
                color=MODEL_CONFIG[model]['color'], markersize=5)
        ax4.fill(angles, f1_vals, alpha=0.1, color=MODEL_CONFIG[model]['color'])
    
    ax4.set_ylim(0, 1.05)
    ax4.set_title('(d) F1 Score Radar', fontweight='bold', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # ========== Panel (e): Line Plot - Position MAE Trend ==========
    ax5 = fig.add_subplot(gs[1, 1])
    for model in MODEL_ORDER:
        vals = [all_basin_data[model][b]['pos_mae'] for b in BASIN_ORDER]
        ax5.plot(range(len(BASIN_ORDER)), vals, 'o-', linewidth=2, markersize=8,
                label=MODEL_CONFIG[model]['name'], color=MODEL_CONFIG[model]['color'],
                marker=MODEL_CONFIG[model]['marker'])
    
    ax5.set_xticks(range(len(BASIN_ORDER)))
    ax5.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=30, ha='right')
    ax5.set_ylabel('Position MAE (km)')
    ax5.set_title('(e) Position Error Trend Across Basins', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_ylim(0, 200)
    
    # ========== Panel (f): Heatmap - F1 Score ==========
    ax6 = fig.add_subplot(gs[1, 2])
    f1_matrix = np.array([[all_basin_data[m][b]['f1'] * 100 for b in BASIN_ORDER] for m in MODEL_ORDER])
    
    im2 = ax6.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    ax6.set_xticks(range(len(BASIN_ORDER)))
    ax6.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=30, ha='right')
    ax6.set_yticks(range(len(MODEL_ORDER)))
    ax6.set_yticklabels([MODEL_CONFIG[m]['name'] for m in MODEL_ORDER])
    
    for i in range(len(MODEL_ORDER)):
        for j in range(len(BASIN_ORDER)):
            val = f1_matrix[i, j]
            color = 'white' if val < 70 else 'black'
            ax6.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax6, shrink=0.8)
    cbar2.set_label('F1 (%)', fontsize=9)
    ax6.set_title('(f) F1 Score Heatmap', fontweight='bold')
    
    # ========== Panel (g): Improvement Bar (Proposed vs Others) ==========
    ax7 = fig.add_subplot(gs[2, 0])
    basins_short = [BASIN_INFO[b]['name'] for b in BASIN_ORDER]
    x = np.arange(len(MODEL_ORDER) - 1)  # Exclude proposed
    width = 0.12
    
    other_models = MODEL_ORDER[1:]
    colors_basin = ['#E64B35', '#00A087', '#4DBBD5', '#3C5488', '#F39B7F', '#8491B4']
    
    for i, basin in enumerate(BASIN_ORDER):
        proposed_mae = all_basin_data['proposed'][basin]['pos_mae']
        improvements = []
        for model in other_models:
            other_mae = all_basin_data[model][basin]['pos_mae']
            imp = (other_mae - proposed_mae) / other_mae * 100 if other_mae > 0 else 0
            improvements.append(imp)
        
        offset = (i - 2.5) * width
        ax7.bar(x + offset, improvements, width, label=basins_short[i], color=colors_basin[i],
               edgecolor='black', linewidth=0.5)
    
    ax7.axhline(0, color='black', linewidth=0.5)
    ax7.set_ylabel('Improvement (%)')
    ax7.set_title('(g) Proposed Position MAE Improvement', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels([MODEL_CONFIG[m]['name'] for m in other_models], rotation=30, ha='right')
    ax7.legend(loc='upper right', ncol=3, fontsize=7)
    ax7.set_ylim(-10, 70)
    
    # ========== Panel (h): Box Plot - All Models Position Error ==========
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Aggregate all basins per model
    box_data = []
    for model in MODEL_ORDER:
        all_dist = []
        for basin in BASIN_ORDER:
            all_dist.extend(all_basin_data[model][basin]['distances'])
        # Filter outliers
        all_dist = [d for d in all_dist if d < 300]
        box_data.append(all_dist)
    
    bp = ax8.boxplot(box_data, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.3))
    
    for patch, model in zip(bp['boxes'], MODEL_ORDER):
        patch.set_facecolor(MODEL_CONFIG[model]['color'])
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
    
    ax8.set_xticklabels([MODEL_CONFIG[m]['name'] for m in MODEL_ORDER], rotation=30, ha='right')
    ax8.set_ylabel('Position Error (km)')
    ax8.set_title('(h) Overall Position Error Distribution', fontweight='bold')
    
    # ========== Panel (i): Summary Statistics Table ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create table data
    col_labels = ['Model', 'Avg MAE\n(km)', 'Avg F1\n(%)', 'Best\nBasin', 'Worst\nBasin']
    table_data = []
    
    for model in MODEL_ORDER:
        mae_vals = [all_basin_data[model][b]['pos_mae'] for b in BASIN_ORDER]
        f1_vals = [all_basin_data[model][b]['f1'] * 100 for b in BASIN_ORDER]
        avg_mae = np.mean(mae_vals)
        avg_f1 = np.mean(f1_vals)
        best_idx = np.argmin(mae_vals)
        worst_idx = np.argmax(mae_vals)
        
        table_data.append([
            MODEL_CONFIG[model]['name'],
            f'{avg_mae:.1f}',
            f'{avg_f1:.1f}',
            BASIN_INFO[BASIN_ORDER[best_idx]]['name'],
            BASIN_INFO[BASIN_ORDER[worst_idx]]['name']
        ])
    
    # Color cells
    cell_colors = []
    for i, model in enumerate(MODEL_ORDER):
        row_colors = [MODEL_CONFIG[model]['color']] + ['white'] * 4
        cell_colors.append(row_colors)
    
    table = ax9.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best row (Proposed)
    for j in range(len(col_labels)):
        table[(1, j)].set_facecolor('#FFE6E6')
    
    ax9.set_title('(i) Summary Statistics', fontweight='bold', y=0.95)
    
    # Main title
    fig.suptitle('Multi-Model Basin-wise Performance Comparison for Tropical Cyclone Detection',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), facecolor='white', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_compact_comparison(all_basin_data, save_path):
    """Create a compact high-impact figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== Panel (a): TC Count MAE Line Chart ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    for model in MODEL_ORDER:
        count_vals = [all_basin_data[model][b]['count_mae'] for b in BASIN_ORDER]
        ax1.plot(range(len(BASIN_ORDER)), count_vals, 'o-', linewidth=2.5, markersize=10,
                label=MODEL_CONFIG[model]['name'], color=MODEL_CONFIG[model]['color'],
                marker=MODEL_CONFIG[model]['marker'], markeredgecolor='black', markeredgewidth=0.5)
    
    ax1.fill_between(range(len(BASIN_ORDER)), 
                     [all_basin_data['proposed'][b]['count_mae'] for b in BASIN_ORDER],
                     alpha=0.2, color=MODEL_CONFIG['proposed']['color'])
    
    ax1.set_xticks(range(len(BASIN_ORDER)))
    ax1.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=45, ha='right')
    ax1.set_ylabel('TC Count MAE')
    ax1.set_title('(a) TC Count Error Across Basins', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # Auto-adjust ylim
    all_count_mae = [all_basin_data[m][b]['count_mae'] for m in MODEL_ORDER for b in BASIN_ORDER]
    ax1.set_ylim(0, max(all_count_mae) * 1.15 if all_count_mae else 1)
    
    # ========== Panel (b): F1 Heatmap ==========
    ax2 = fig.add_subplot(gs[0, 1])
    f1_matrix = np.array([[all_basin_data[m][b]['f1'] * 100 for b in BASIN_ORDER] for m in MODEL_ORDER])
    
    im2 = ax2.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    ax2.set_xticks(range(len(BASIN_ORDER)))
    ax2.set_xticklabels([BASIN_INFO[b]['name'] for b in BASIN_ORDER], rotation=45, ha='right')
    ax2.set_yticks(range(len(MODEL_ORDER)))
    ax2.set_yticklabels([MODEL_CONFIG[m]['name'] for m in MODEL_ORDER])
    
    for i in range(len(MODEL_ORDER)):
        for j in range(len(BASIN_ORDER)):
            val = f1_matrix[i, j]
            color = 'white' if val < 70 else 'black'
            ax2.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')
    
    # Add best marker
    for j in range(len(BASIN_ORDER)):
        best_i = np.argmax(f1_matrix[:, j])
        ax2.add_patch(Rectangle((j-0.5, best_i-0.5), 1, 1, fill=False,
                                edgecolor='gold', linewidth=3))
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label('F1 Score (%)', fontsize=10)
    ax2.set_title('(b) Detection F1', fontweight='bold', pad=10)
    
    # ========== Panel (c): Position MAE Radar Chart ==========
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    
    categories = [BASIN_INFO[b]['name'] for b in BASIN_ORDER]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=9)
    
    # Plot raw Position MAE values (lower is better)
    for model in MODEL_ORDER:
        mae_vals = [all_basin_data[model][b]['pos_mae'] for b in BASIN_ORDER]
        mae_vals += mae_vals[:1]
        ax3.plot(angles, mae_vals, 'o-', linewidth=2, label=MODEL_CONFIG[model]['name'],
                color=MODEL_CONFIG[model]['color'], markersize=6)
        ax3.fill(angles, mae_vals, alpha=0.1, color=MODEL_CONFIG[model]['color'])
    
    # Auto-adjust ylim based on data
    all_pos_mae = [all_basin_data[m][b]['pos_mae'] for m in MODEL_ORDER for b in BASIN_ORDER]
    max_val = max(all_pos_mae) if all_pos_mae else 200
    ax3.set_ylim(0, max_val * 1.1)
    ax3.set_title('(c) Position MAE', fontweight='bold', y=1.08)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=8)
    
    # ========== Panel (d): Basin-wise Improvement (like Panel g in full) ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    basins_short = [BASIN_INFO[b]['name'] for b in BASIN_ORDER]
    other_models = MODEL_ORDER[1:]
    x = np.arange(len(other_models))
    width = 0.12
    colors_basin = ['#E64B35', '#00A087', '#4DBBD5', '#3C5488', '#F39B7F', '#8491B4']
    
    all_improvements = []
    for i, basin in enumerate(BASIN_ORDER):
        proposed_mae = all_basin_data['proposed'][basin]['pos_mae']
        improvements = []
        for model in other_models:
            other_mae = all_basin_data[model][basin]['pos_mae']
            imp = (other_mae - proposed_mae) / other_mae * 100 if other_mae > 0 else 0
            improvements.append(imp)
        all_improvements.extend(improvements)
        
        offset = (i - 2.5) * width
        ax4.bar(x + offset, improvements, width, label=basins_short[i], color=colors_basin[i],
               edgecolor='black', linewidth=0.5)
    
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('(d) Proposed Position MAE Improvement by Basin', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([MODEL_CONFIG[m]['name'] for m in other_models], rotation=30, ha='right')
    ax4.legend(loc='upper right', ncol=3, fontsize=7)
    
    # Auto-adjust ylim
    max_val = max(all_improvements) if all_improvements else 50
    ax4.set_ylim(0, max(50, max_val * 1.1))
    ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for gold boxes (on Panel b - F1 Heatmap)
    legend_elements = [Patch(facecolor='none', edgecolor='gold', linewidth=2, label='Best in Basin')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    fig.suptitle('Comprehensive Basin-wise Performance: Proposed vs Baseline Models',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), facecolor='white', bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ==================== Main ====================
def main():
    base_path = Path(r"D:\Desktop\autodl")
    
    print("Loading all model data...")
    all_basin_data = {}
    for model, config in MODEL_CONFIG.items():
        records = load_jsonl(base_path / config['file'])
        all_basin_data[model] = extract_basin_metrics(records)
        print(f"  {config['name']}: loaded")
    
    print("\nGenerating comprehensive figures...")
    
    # Figure 1: Full 9-panel comprehensive
    plot_comprehensive_basin_comparison(all_basin_data, str(base_path / "fig_all_models_basin_full.png"))
    
    # Figure 2: Compact 4-panel high-impact
    plot_compact_comparison(all_basin_data, str(base_path / "fig_all_models_basin_compact.png"))
    
    print("\n" + "="*60)
    print("All comprehensive figures saved!")
    print("Recommended: fig_all_models_basin_compact.pdf")
    print("="*60)

if __name__ == "__main__":
    main()


