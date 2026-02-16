"""
Tropical Cyclone Detection Error Comparison Visualization
Publication-ready figures with Nature journal color scheme
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# ==================== Configuration ====================
NATURE_COLORS = {
    'proposed': '#E64B35',   # Red - Our method (highlighted)
    'claude': '#4DBBD5',     # Cyan
    'gpt': '#00A087',        # Teal
    'gemini': '#3C5488',     # Navy blue
    'maskrcnn': '#F39B7F',   # Coral/Orange
}

MODEL_NAMES = {
    'proposed': 'TCG-LLM',
    'claude': 'Gemini-3',
    'gpt': 'GPT-5.2',
    'gemini': 'Claude-4.5',
    'maskrcnn': 'Mask-RCNN',
}

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
MATCH_DISTANCE_THRESHOLD = 500  
DISTANCE_ERROR_CAP = None  
# ==================== Data Loading ====================
def load_jsonl(filepath):
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km between two points."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def match_tcs(pred_tcs, gt_tcs):
    """Match predicted TCs to ground truth using Hungarian algorithm."""
    if not pred_tcs or not gt_tcs:
        return []
    
    n_pred = len(pred_tcs)
    n_gt = len(gt_tcs)
    
    # Build cost matrix
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_tcs):
        for j, gt in enumerate(gt_tcs):
            cost_matrix[i, j] = haversine_distance(
                pred['lat'], pred['lon'], gt['lat'], gt['lon']
            )
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_errors = []
    for i, j in zip(row_ind, col_ind):
        # Only count as matched if distance is reasonable
        if cost_matrix[i, j] < MATCH_DISTANCE_THRESHOLD:
            lat_err = abs(pred_tcs[i]['lat'] - gt_tcs[j]['lat'])
            lon_err = abs(pred_tcs[i]['lon'] - gt_tcs[j]['lon'])
            dist_err = cost_matrix[i, j]
            matched_errors.append({
                'lat_error': lat_err,
                'lon_error': lon_err,
                'distance_error': dist_err
            })
    
    return matched_errors

def calculate_metrics(records):
    """Calculate error metrics for a model."""
    count_errors = []
    lat_errors = []
    lon_errors = []
    distance_errors = []
    
    # Detection metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for record in records:
        pred = record['pred']
        gt = record['gt']
        
        # Count error (handle missing fields)
        pred_count = pred.get('current_tc_count', 0)
        gt_count = gt.get('current_tc_count', 0)
        count_errors.append(pred_count - gt_count)
        
        # Position errors (only for matched TCs)
        pred_tcs = pred.get('current_tcs', []) or []
        gt_tcs = gt.get('current_tcs', []) or []
        
        matched_errors = match_tcs(pred_tcs, gt_tcs)
        for err in matched_errors:
            lat_errors.append(err['lat_error'])
            lon_errors.append(err['lon_error'])
            # Apply distance error cap if configured
            dist_err = err['distance_error']
            if DISTANCE_ERROR_CAP is not None:
                dist_err = min(dist_err, DISTANCE_ERROR_CAP)
            distance_errors.append(dist_err)
        
        # Detection counts
        n_matched = len(matched_errors)
        true_positives += n_matched
        false_positives += max(0, len(pred_tcs) - n_matched)
        false_negatives += max(0, len(gt_tcs) - n_matched)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate False Alarm Rate (FAR) and Miss Rate (MR)
    # FAR = FP / (TP + FP) = 1 - Precision
    false_alarm_rate = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    # MR = FN / (TP + FN) = 1 - Recall
    miss_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        'count_mae': np.mean(np.abs(count_errors)),
        'count_rmse': np.sqrt(np.mean(np.array(count_errors)**2)),
        'lat_mae': np.mean(lat_errors) if lat_errors else 0,
        'lon_mae': np.mean(lon_errors) if lon_errors else 0,
        'distance_mae': np.mean(distance_errors) if distance_errors else 0,
        'distance_rmse': np.sqrt(np.mean(np.array(distance_errors)**2)) if distance_errors else 0,
        'lat_errors': lat_errors,
        'lon_errors': lon_errors,
        'distance_errors': distance_errors,
        'count_errors': count_errors,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_alarm_rate': false_alarm_rate,
        'miss_rate': miss_rate,
    }

# ==================== Visualization ====================
def plot_bar_comparison(all_metrics, save_path):
    """Create bar chart comparing key metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    x = np.arange(len(models))
    width = 0.6
    
    # Panel (a): TC Count MAE
    ax1 = axes[0]
    values = [all_metrics[m]['count_mae'] for m in models]
    colors = [NATURE_COLORS[m] for m in models]
    bars1 = ax1.bar(x, values, width, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('MAE (count)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax1.set_ylim(0, max(values) * 1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars1, values):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    # Panel (b): Position Error (Distance MAE)
    ax2 = axes[1]
    values = [all_metrics[m]['distance_mae'] for m in models]
    bars2 = ax2.bar(x, values, width, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('MAE (km)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax2.set_ylim(0, max(values) * 1.2)
    
    for bar, val in zip(bars2, values):
        ax2.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    # Panel (c): F1 Score
    ax3 = axes[2]
    values = [all_metrics[m]['f1'] * 100 for m in models]
    bars3 = ax3.bar(x, values, width, color=colors, edgecolor='black', linewidth=0.8)
    ax3.set_ylabel('F1 Score (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax3.set_ylim(0, 105)
    
    for bar, val in zip(bars3, values):
        ax3.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_boxplot_comparison(all_metrics, save_path):
    """Create boxplot for position error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['proposed', 'gpt', 'gemini', 'claude', 'maskrcnn']
    
    # Panel (a): Latitude Error Distribution
    ax1 = axes[0]
    lat_data = [all_metrics[m]['lat_errors'] for m in models]
    parts1 = ax1.violinplot(lat_data, positions=range(1, len(models)+1), widths=0.7,
                           showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts1['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts1['cmeans'].set_color('black')
    parts1['cmeans'].set_linewidth(1.5)
    parts1['cmedians'].set_color('white')
    parts1['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp1 = ax1.boxplot(lat_data, positions=range(1, len(models)+1), widths=0.15,
                     patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='black', linewidth=1)
    
    ax1.set_ylabel('Latitude Error (°)')
    ax1.set_xticks(range(1, len(models)+1))
    ax1.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    # Panel (b): Longitude Error Distribution
    ax2 = axes[1]
    lon_data = [all_metrics[m]['lon_errors'] for m in models]
    parts2 = ax2.violinplot(lon_data, positions=range(1, len(models)+1), widths=0.7,
                           showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts2['cmeans'].set_color('black')
    parts2['cmeans'].set_linewidth(1.5)
    parts2['cmedians'].set_color('white')
    parts2['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp2 = ax2.boxplot(lon_data, positions=range(1, len(models)+1), widths=0.15,
                     patch_artist=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp2[element], color='black', linewidth=1)
    
    ax2.set_ylabel('Longitude Error (°)')
    ax2.set_xticks(range(1, len(models)+1))
    ax2.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_count_error(all_metrics, save_path):
    """Subplot (a): TC Count MAE & RMSE"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    colors = [NATURE_COLORS[m] for m in models]
    x_grouped = np.arange(len(models))
    width_g = 0.35
    
    mae_vals = [all_metrics[m]['count_mae'] for m in models]
    rmse_vals = [all_metrics[m]['count_rmse'] for m in models]
    
    bars1 = ax.bar(x_grouped - width_g/2, mae_vals, width_g, label='MAE', 
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x_grouped + width_g/2, rmse_vals, width_g, label='RMSE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.5, hatch='///')
    
    ax.set_ylabel('Error (count)')
    ax.set_xticks(x_grouped)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(max(mae_vals), max(rmse_vals)) * 1.25)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_position_error(all_metrics, save_path):
    """Subplot (b): Position Distance Error"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    colors = [NATURE_COLORS[m] for m in models]
    x_grouped = np.arange(len(models))
    width_g = 0.35
    
    dist_mae = [all_metrics[m]['distance_mae'] for m in models]
    dist_rmse = [all_metrics[m]['distance_rmse'] for m in models]
    
    bars3 = ax.bar(x_grouped - width_g/2, dist_mae, width_g, label='MAE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    bars4 = ax.bar(x_grouped + width_g/2, dist_rmse, width_g, label='RMSE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.5, hatch='///')
    
    ax.set_ylabel('Distance Error (km)')
    ax.set_xticks(x_grouped)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, max(max(dist_mae), max(dist_rmse)) * 1.25)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_detection_metrics(all_metrics, save_path):
    """Subplot (c): Precision, Recall, F1"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    x = np.arange(len(models))
    width_t = 0.25
    
    precision_vals = [all_metrics[m]['precision'] * 100 for m in models]
    recall_vals = [all_metrics[m]['recall'] * 100 for m in models]
    f1_vals = [all_metrics[m]['f1'] * 100 for m in models]
    
    bars5 = ax.bar(x - width_t, precision_vals, width_t, label='Precision',
                    color='#E64B35', edgecolor='black', linewidth=0.8)
    bars6 = ax.bar(x, recall_vals, width_t, label='Recall',
                    color='#4DBBD5', edgecolor='black', linewidth=0.8)
    bars7 = ax.bar(x + width_t, f1_vals, width_t, label='F1 Score',
                    color='#00A087', edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_latitude_error(all_metrics, save_path):
    """Subplot (d): Latitude Error Boxplot"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    
    lat_data = [all_metrics[m]['lat_errors'] for m in models]
    parts = ax.violinplot(lat_data, positions=range(1, len(models)+1), widths=0.7,
                         showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp = ax.boxplot(lat_data, positions=range(1, len(models)+1), widths=0.15,
                   patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    ax.set_ylabel('Latitude Error (°)')
    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_longitude_error(all_metrics, save_path):
    """Subplot (e): Longitude Error Boxplot"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    
    lon_data = [all_metrics[m]['lon_errors'] for m in models]
    parts = ax.violinplot(lon_data, positions=range(1, len(models)+1), widths=0.7,
                         showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp = ax.boxplot(lon_data, positions=range(1, len(models)+1), widths=0.15,
                   patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1)
    
    ax.set_ylabel('Longitude Error (°)')
    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_subplot_cdf(all_metrics, save_path):
    """Subplot (f): Distance Error CDF"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    
    for model in models:
        errors = sorted(all_metrics[model]['distance_errors'])
        if errors:
            cdf = np.arange(1, len(errors) + 1) / len(errors)
            if model == 'proposed':
                ax.plot(errors, cdf, label=MODEL_NAMES[model], color=NATURE_COLORS[model],
                        linewidth=3.2, alpha=1.0, zorder=5)
            else:
                ax.plot(errors, cdf, label=MODEL_NAMES[model], color=NATURE_COLORS[model], 
                        linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Position Error (km)')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_comprehensive_comparison(all_metrics, save_path):
    """Create comprehensive comparison figure."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    colors = [NATURE_COLORS[m] for m in models]
    x = np.arange(len(models))
    width = 0.6
    
    # (a) TC Count MAE & RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    mae_vals = [all_metrics[m]['count_mae'] for m in models]
    rmse_vals = [all_metrics[m]['count_rmse'] for m in models]
    
    x_grouped = np.arange(len(models))
    width_g = 0.35
    bars1 = ax1.bar(x_grouped - width_g/2, mae_vals, width_g, label='MAE', 
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax1.bar(x_grouped + width_g/2, rmse_vals, width_g, label='RMSE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.5, hatch='///')
    
    ax1.set_ylabel('Error (count)')
    ax1.set_xticks(x_grouped)
    ax1.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, max(max(mae_vals), max(rmse_vals)) * 1.25)
    
    # (b) Position Distance Error
    ax2 = fig.add_subplot(gs[0, 1])
    dist_mae = [all_metrics[m]['distance_mae'] for m in models]
    dist_rmse = [all_metrics[m]['distance_rmse'] for m in models]
    
    bars3 = ax2.bar(x_grouped - width_g/2, dist_mae, width_g, label='MAE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.9)
    bars4 = ax2.bar(x_grouped + width_g/2, dist_rmse, width_g, label='RMSE',
                    color=colors, edgecolor='black', linewidth=0.8, alpha=0.5, hatch='///')
    
    ax2.set_ylabel('Distance Error (km)')
    ax2.set_xticks(x_grouped)
    ax2.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, max(max(dist_mae), max(dist_rmse)) * 1.25)
    
    # (c) Precision, Recall, F1
    ax3 = fig.add_subplot(gs[0, 2])
    precision_vals = [all_metrics[m]['precision'] * 100 for m in models]
    recall_vals = [all_metrics[m]['recall'] * 100 for m in models]
    f1_vals = [all_metrics[m]['f1'] * 100 for m in models]
    
    width_t = 0.25
    bars5 = ax3.bar(x - width_t, precision_vals, width_t, label='Precision',
                    color='#E64B35', edgecolor='black', linewidth=0.8)
    bars6 = ax3.bar(x, recall_vals, width_t, label='Recall',
                    color='#4DBBD5', edgecolor='black', linewidth=0.8)
    bars7 = ax3.bar(x + width_t, f1_vals, width_t, label='F1 Score',
                    color='#00A087', edgecolor='black', linewidth=0.8)
    
    ax3.set_ylabel('Score (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.set_ylim(0, 105)
    
    # (d) Latitude Error Violin Plot
    ax4 = fig.add_subplot(gs[1, 0])
    lat_data = [all_metrics[m]['lat_errors'] for m in models]
    parts1 = ax4.violinplot(lat_data, positions=range(1, len(models)+1), widths=0.7,
                           showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts1['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts1['cmeans'].set_color('black')
    parts1['cmeans'].set_linewidth(1.5)
    parts1['cmedians'].set_color('white')
    parts1['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp1 = ax4.boxplot(lat_data, positions=range(1, len(models)+1), widths=0.15,
                     patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='black', linewidth=1)
    
    ax4.set_ylabel('Latitude Error (°)')
    ax4.set_xticks(range(1, len(models)+1))
    ax4.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    # (e) Longitude Error Violin Plot
    ax5 = fig.add_subplot(gs[1, 1])
    lon_data = [all_metrics[m]['lon_errors'] for m in models]
    parts2 = ax5.violinplot(lon_data, positions=range(1, len(models)+1), widths=0.7,
                           showmeans=True, showmedians=True, showextrema=False)
    
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(NATURE_COLORS[models[i]])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(0.8)
    
    parts2['cmeans'].set_color('black')
    parts2['cmeans'].set_linewidth(1.5)
    parts2['cmedians'].set_color('white')
    parts2['cmedians'].set_linewidth(1.5)
    
    # Add box plot on top for quartiles
    bp2 = ax5.boxplot(lon_data, positions=range(1, len(models)+1), widths=0.15,
                     patch_artist=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp2[element], color='black', linewidth=1)
    
    ax5.set_ylabel('Longitude Error (°)')
    ax5.set_xticks(range(1, len(models)+1))
    ax5.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=30, ha='right')
    
    # (f) Distance Error CDF
    ax6 = fig.add_subplot(gs[1, 2])
    for model in models:
        errors = sorted(all_metrics[model]['distance_errors'])
        if errors:
            cdf = np.arange(1, len(errors) + 1) / len(errors)
            if model == 'proposed':
                ax6.plot(errors, cdf, label=MODEL_NAMES[model], color=NATURE_COLORS[model],
                        linewidth=3.2, alpha=1.0, zorder=5)
            else:
                ax6.plot(errors, cdf, label=MODEL_NAMES[model], color=NATURE_COLORS[model], 
                        linewidth=2, alpha=0.9)
    
    ax6.set_xlabel('Position Error (km)')
    ax6.set_ylabel('Cumulative Probability')
    ax6.legend(loc='lower right', framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_xlim(0, None)
    ax6.set_ylim(0, 1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def plot_radar_comparison(all_metrics, save_path):
    """Create radar chart for multi-dimensional comparison."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    
    # Metrics to compare (normalized, higher is better)
    categories = ['F1 Score', 'Precision', 'Recall', 'Count Acc.', 'Position Acc.']
    
    # Calculate normalized values (0-1, higher is better)
    max_count_mae = max(all_metrics[m]['count_mae'] for m in models)
    max_dist_mae = max(all_metrics[m]['distance_mae'] for m in models)
    
    values_dict = {}
    for model in models:
        m = all_metrics[model]
        values_dict[model] = [
            m['f1'],
            m['precision'],
            m['recall'],
            1 - m['count_mae'] / max_count_mae if max_count_mae > 0 else 1,
            1 - m['distance_mae'] / max_dist_mae if max_dist_mae > 0 else 1,
        ]
    
    # Setup radar
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Draw each model
    for model in models:
        values = values_dict[model] + values_dict[model][:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=MODEL_NAMES[model], 
                color=NATURE_COLORS[model], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=NATURE_COLORS[model])
    
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

def print_metrics_table(all_metrics):
    """Print metrics in table format for paper."""
    models = ['proposed', 'gpt', 'claude', 'gemini', 'maskrcnn']
    
    print("\n" + "="*90)
    print("TABLE: Quantitative Comparison of TC Detection and Localization Performance")
    print("="*90)
    print(f"{'Model':<12} {'Count MAE':>10} {'Count RMSE':>12} {'Dist MAE':>10} {'Dist RMSE':>11} {'FAR':>8} {'MR':>8} {'F1':>8}")
    print("-"*90)
    
    for model in models:
        m = all_metrics[model]
        print(f"{MODEL_NAMES[model]:<12} {m['count_mae']:>10.4f} {m['count_rmse']:>12.4f} "
              f"{m['distance_mae']:>10.2f} {m['distance_rmse']:>11.2f} "
              f"{m['false_alarm_rate']*100:>7.2f}% {m['miss_rate']*100:>7.2f}% {m['f1']*100:>7.2f}%")
    
    print("="*90)
    print("Note: Count MAE/RMSE in TC count; Distance MAE/RMSE in km; FAR = False Alarm Rate; MR = Miss Rate")
    print()

# ==================== Main ====================
def main():
    base_path = Path(r"figures")
    
    # Load all model results
    print("Loading data...")
    model_files = {
        'proposed': 'TCG-LLM.jsonl',
        'claude': 'claude.jsonl',
        'gpt': 'gpt.jsonl',
        'gemini': 'gemini.jsonl',
        'maskrcnn': 'maskrcnn.jsonl',
    }
    
    all_data = {}
    for model, filename in model_files.items():
        filepath = base_path / filename
        all_data[model] = load_jsonl(filepath)
        print(f"  Loaded {model}: {len(all_data[model])} records")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    all_metrics = {}
    for model, records in all_data.items():
        all_metrics[model] = calculate_metrics(records)
        print(f"  {model}: Count MAE={all_metrics[model]['count_mae']:.4f}, "
              f"Distance MAE={all_metrics[model]['distance_mae']:.2f} km, "
              f"F1={all_metrics[model]['f1']*100:.2f}%")
    
    # Print table
    print_metrics_table(all_metrics)
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: Simple bar comparison
    # plot_bar_comparison(all_metrics, str(base_path / "fig_error_bars.png"))
    
    # # Figure 2: Boxplot comparison
    # plot_boxplot_comparison(all_metrics, str(base_path / "fig_error_boxplot.png"))
    
    # Figure 3: Comprehensive comparison 
    plot_comprehensive_comparison(all_metrics, str(base_path / "fig_comprehensive_comparison.png"))
    
    # Figure 4: Radar chart
    # plot_radar_comparison(all_metrics, str(base_path / "fig_radar_comparison.png"))
    
    # Individual subplots from comprehensive comparison
    # print("\nGenerating individual subplots...")
    # plot_subplot_count_error(all_metrics, str(base_path / "fig_subplot_count_error.png"))
    # plot_subplot_position_error(all_metrics, str(base_path / "fig_subplot_position_error.png"))
    # plot_subplot_detection_metrics(all_metrics, str(base_path / "fig_subplot_detection_metrics.png"))
    # plot_subplot_latitude_error(all_metrics, str(base_path / "fig_subplot_latitude_error.png"))
    # plot_subplot_longitude_error(all_metrics, str(base_path / "fig_subplot_longitude_error.png"))
    # plot_subplot_cdf(all_metrics, str(base_path / "fig_subplot_cdf.png"))
    
    print("\nAll figures saved successfully!")


if __name__ == "__main__":
    main()

