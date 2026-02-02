"""
Combined Kalshi vs Polymarket Comparison Chart
Fee comparison (left) and Latency boxplots (right)
Matching Galaxy Research style
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# Setup Style
# ============================================================================

sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# Original brand colors
KALSHI_COLOR = '#00cb90'      # Kalshi green
POLYMARKET_COLOR = '#2e59f7'  # Polymarket blue

# ============================================================================
# Load Latency Data
# ============================================================================

kalshi_maker = pd.read_csv(r'D:\PROJECTS\poly\latency\kalshi\kalshi_maker_delay.csv')
kalshi_taker = pd.read_csv(r'D:\PROJECTS\poly\latency\kalshi\kalshi_taker_delay.csv')
poly_maker = pd.read_csv(r'D:\PROJECTS\poly\latency\polymarket\polymarket_maker_delay.csv')
poly_taker = pd.read_csv(r'D:\PROJECTS\poly\latency\polymarket\polymarket_taker_delay.csv')

# Extract delay values and cap at 100ms to exclude extremes
cap = 100
kalshi_maker_clean = kalshi_maker['maker_delay_ms'].dropna().values
kalshi_maker_clean = kalshi_maker_clean[kalshi_maker_clean <= cap]

kalshi_taker_clean = kalshi_taker['taker_delay_ms'].dropna().values
kalshi_taker_clean = kalshi_taker_clean[kalshi_taker_clean <= cap]

poly_maker_clean = poly_maker['maker_delay_ms'].dropna().values
poly_maker_clean = poly_maker_clean[poly_maker_clean <= cap]

poly_taker_clean = poly_taker['taker_delay_ms'].dropna().values
poly_taker_clean = poly_taker_clean[poly_taker_clean <= cap]

# Build DataFrames for seaborn
taker_df = pd.DataFrame({
    'Exchange': ['Kalshi'] * len(kalshi_taker_clean) + ['Polymarket'] * len(poly_taker_clean),
    'Delay': np.concatenate([kalshi_taker_clean, poly_taker_clean])
})

maker_df = pd.DataFrame({
    'Exchange': ['Kalshi'] * len(kalshi_maker_clean) + ['Polymarket'] * len(poly_maker_clean),
    'Delay': np.concatenate([kalshi_maker_clean, poly_maker_clean])
})

# ============================================================================
# Create Combined Figure
# ============================================================================

fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor('white')

# Create grid: 3 subplots
ax1 = fig.add_subplot(1, 3, 1)  # Fee comparison
ax2 = fig.add_subplot(1, 3, 2)  # Taker latency
ax3 = fig.add_subplot(1, 3, 3)  # Maker latency

palette = {'Kalshi': KALSHI_COLOR, 'Polymarket': POLYMARKET_COLOR}

# ============================================================================
# Plot 1: Fee Comparison
# ============================================================================

prices = np.linspace(0.01, 0.99, 200)

# Kalshi fees in bps
kalshi_taker_bps = 700 * prices * (1 - prices)
kalshi_maker_bps = 175 * prices * (1 - prices)

# Polymarket = 0 fees (show slightly above axis for visibility)
polymarket_bps = np.full_like(prices, 3)

ax1.plot(prices, kalshi_taker_bps, color=KALSHI_COLOR, linewidth=2, label='Kalshi Taker', linestyle='-')
ax1.plot(prices, kalshi_maker_bps, color=KALSHI_COLOR, linewidth=2, label='Kalshi Maker', linestyle='--')
ax1.plot(prices, polymarket_bps, color=POLYMARKET_COLOR, linewidth=2.5, label='Polymarket (0 bps)')

ax1.set_xlabel('Contract Price', fontsize=16, color='#666666')
ax1.set_ylabel('bps per contract', fontsize=16, color='#666666')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, None)
ax1.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
ax1.set_xticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=15)
ax1.tick_params(axis='y', labelsize=15)

# Styling
sns.despine(ax=ax1, top=True, right=True)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['left'].set_color('#666666')
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['bottom'].set_color('#666666')
ax1.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')

# Legend center top - positioned above the curve but below title
ax1.legend(frameon=False, fontsize=14, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))

# Title in top left
ax1.text(0.0, 1.15, 'Trading Fees', transform=ax1.transAxes, fontsize=18, fontweight='semibold', verticalalignment='top')

# ============================================================================
# Plot 2: Taker Latency
# ============================================================================

sns.boxplot(x='Exchange', y='Delay', data=taker_df, ax=ax2,
            hue='Exchange', palette=palette, width=0.5, linewidth=0.8,
            showfliers=False, legend=False,
            boxprops=dict(edgecolor='#333333'),
            whiskerprops=dict(color='#333333', linewidth=0.8),
            capprops=dict(color='#333333', linewidth=0.8),
            medianprops=dict(color='#333333', linewidth=0.8))

ax2.set_xlabel('')
ax2.set_ylabel('Order Delay (ms)', fontsize=16, color='#666666')
ax2.set_ylim(0, None)
ax2.tick_params(axis='both', labelsize=15)

# Styling
sns.despine(ax=ax2, top=True, right=True)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['left'].set_color('#666666')
ax2.spines['bottom'].set_linewidth(0.5)
ax2.spines['bottom'].set_color('#666666')
ax2.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')

# Title
ax2.text(0.0, 1.15, 'Taker Latency', transform=ax2.transAxes, fontsize=18, fontweight='semibold', verticalalignment='top')

# ============================================================================
# Plot 3: Maker Latency
# ============================================================================

sns.boxplot(x='Exchange', y='Delay', data=maker_df, ax=ax3,
            hue='Exchange', palette=palette, width=0.5, linewidth=0.8,
            showfliers=False, legend=False,
            boxprops=dict(edgecolor='#333333'),
            whiskerprops=dict(color='#333333', linewidth=0.8),
            capprops=dict(color='#333333', linewidth=0.8),
            medianprops=dict(color='#333333', linewidth=0.8))

ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_ylim(0, None)
ax3.tick_params(axis='both', labelsize=15)

# Styling
sns.despine(ax=ax3, top=True, right=True)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['left'].set_color('#666666')
ax3.spines['bottom'].set_linewidth(0.5)
ax3.spines['bottom'].set_color('#666666')
ax3.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')

# Title
ax3.text(0.0, 1.15, 'Maker Latency', transform=ax3.transAxes, fontsize=18, fontweight='semibold', verticalalignment='top')

# ============================================================================
# Overall Title and Footer
# ============================================================================

# Main title
fig.text(0.0, 1.02, 'Kalshi vs Polymarket: Fees and Latency Comparison', 
         fontsize=20, fontweight='bold', verticalalignment='top')

# Footer with methodology note - italic
fig.text(0.5, -0.02, 
         'Data generated through co-located AWS servers: us-east-1 for Kalshi and eu-west-1 for Polymarket. '
         '30 samples for end-to-end latency (taker), 100 samples for end-to-end order creation (maker).',
         fontsize=15, fontstyle='italic', ha='center', color='#555555')

plt.tight_layout()
plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.22, left=0.05, right=0.98)

# Save
plt.savefig(r'D:\PROJECTS\poly\latency\kalshi_polymarket_comparison.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("Combined chart saved: kalshi_polymarket_comparison.png")
print(f"\nTaker - Kalshi: median={np.median(kalshi_taker_clean):.1f}ms, Polymarket: median={np.median(poly_taker_clean):.1f}ms")
print(f"Maker - Kalshi: median={np.median(kalshi_maker_clean):.1f}ms, Polymarket: median={np.median(poly_maker_clean):.1f}ms")
print(f"\nFees at P = 0.50 (bps per contract):")
print(f"  Kalshi Taker: {700 * 0.5 * 0.5:.1f} bps")
print(f"  Kalshi Maker: {175 * 0.5 * 0.5:.1f} bps")
print(f"  Polymarket:   0 bps")
