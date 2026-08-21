import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def clarke_zone(ref, pred):
    """
    Computes Clarke Error Grid zone ('A', 'B', 'C', 'D', 'E')
    for a reference (actual) and predicted glucose pair in mg/dL.
    """
    ref = float(ref)
    pred = float(pred)
    
    # Zone A: within 20% of reference or both <= 70
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return 'A'
    
    # Zone E: opposite treatment (erroneous extreme)
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return 'E'
    
    # Zone C: overcorrecting
    if (70 <= ref <= 290) and (pred >= ref + 110):
        return 'C'
    if (130 <= ref <= 180) and (pred <= (7.0 / 5.0) * ref - 182):
        return 'C'
    
    # Zone D: failure to detect hypo/hyper
    if (ref >= 240) and (70 <= pred <= 180):
        return 'D'
    if (ref <= 70) and (70 < pred < 180):
        return 'D'
    
    # Zone B: benign errors (everything else)
    return 'B'

def evaluate_clarke_grid(y_true, y_pred):
    """
    Evaluates Clarke Error Grid distributions.
    Returns dictionary with counts and percentages for each zone.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[valid], y_p[valid]
    
    n_total = len(y_t)
    if n_total == 0:
        return {z: 0.0 for z in ['A', 'B', 'C', 'D', 'E', 'A+B']}
    
    zones = [clarke_zone(r, p) for r, p in zip(y_t, y_p)]
    counts = pd.Series(zones).value_counts().reindex(['A', 'B', 'C', 'D', 'E'], fill_value=0)
    pcts = (counts / n_total) * 100.0
    
    return {
        'A': pcts['A'],
        'B': pcts['B'],
        'C': pcts['C'],
        'D': pcts['D'],
        'E': pcts['E'],
        'A+B': pcts['A'] + pcts['B'],
        'n_total': n_total,
        'zones_list': zones
    }

def plot_clarke_error_grid(y_true, y_pred, title="Clarke Error Grid Analysis", ax=None, save_path=None):
    """
    Draws a standard Clarke Error Grid diagram with data points and zone percentages.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[valid], y_p[valid]
    
    stats = evaluate_clarke_grid(y_t, y_p)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    else:
        fig = ax.figure
        
    ax.set_facecolor("#FAFAFA")
    
    # Plot reference line y = x
    ax.plot([0, 450], [0, 450], 'k--', linewidth=1.2, alpha=0.7, label="Ideal (y = x)")
    
    # 20% error boundary lines (Zone A)
    ax.plot([0, 450], [0, 450 * 1.2], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([0, 450], [0, 450 * 0.8], 'k-', linewidth=0.8, alpha=0.6)
    
    # Boundary lines for Zone C & D
    # Upper C line: pred = ref + 110 for ref in [70, 290]
    ax.plot([70, 290], [180, 400], 'k-', linewidth=0.8, alpha=0.6)
    # Lower C line: pred = (7/5)ref - 182 for ref in [130, 180]
    ax.plot([130, 180], [0, 70], 'k-', linewidth=0.8, alpha=0.6)
    
    # Zone D boundaries
    ax.plot([240, 450], [70, 70], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([240, 450], [180, 180], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([70, 70], [180, 450], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([0, 70], [70, 70], 'k-', linewidth=0.8, alpha=0.6)
    ax.plot([70, 70], [0, 70], 'k-', linewidth=0.8, alpha=0.6)
    
    # Scatter points with slight alpha
    ax.scatter(y_t, y_p, s=14, color="#1f77b4", alpha=0.45, edgecolors='none', zorder=3)
    
    # Zone Labels
    ax.text(30, 15, "A", fontsize=14, fontweight='bold', color="#2ca02c")
    ax.text(380, 370, "A", fontsize=14, fontweight='bold', color="#2ca02c")
    ax.text(280, 390, "B", fontsize=13, fontweight='bold', color="#1f77b4")
    ax.text(390, 260, "B", fontsize=13, fontweight='bold', color="#1f77b4")
    ax.text(160, 380, "C", fontsize=13, fontweight='bold', color="#ff7f0e")
    ax.text(160, 20, "C", fontsize=13, fontweight='bold', color="#ff7f0e")
    ax.text(35, 125, "D", fontsize=13, fontweight='bold', color="#d62728")
    ax.text(380, 125, "D", fontsize=13, fontweight='bold', color="#d62728")
    ax.text(35, 380, "E", fontsize=13, fontweight='bold', color="#9467bd")
    ax.text(380, 35, "E", fontsize=13, fontweight='bold', color="#9467bd")
    
    # Summary Box
    stats_text = (
        f"Zone A: {stats['A']:.1f}%\n"
        f"Zone B: {stats['B']:.1f}%\n"
        f"Zone C: {stats['C']:.1f}%\n"
        f"Zone D: {stats['D']:.1f}%\n"
        f"Zone E: {stats['E']:.1f}%\n"
        f"Clinical (A+B): {stats['A+B']:.1f}%\n"
        f"N = {stats['n_total']}"
    )
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9),
            fontsize=10, family='monospace')
    
    ax.set_xlim(0, 450)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    return fig, ax
