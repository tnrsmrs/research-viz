"""
Kalshi vs Polymarket Daily Trading Volume Comparison
Last 1 year of volume data
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
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

# Brand colors
KALSHI_COLOR = '#00cb90'      # Kalshi green
POLYMARKET_COLOR = '#2e59f7'  # Polymarket blue

# Time range: last 365 days
DAYS_LOOKBACK = 365

# ============================================================================
# Data Fetching Functions
# ============================================================================

def fetch_polymarket_volume():
    """
    Fetch Polymarket volume data from DefiLlama.
    """
    print("Fetching Polymarket volume data...")
    url = "https://api.llama.fi/summary/dexs/polymarket"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        chart_data = data.get('totalDataChart', [])
        if chart_data:
            df = pd.DataFrame(chart_data, columns=['timestamp', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('date')
            return df
    except Exception as e:
        print(f"  Error fetching Polymarket: {e}")
    
    return pd.DataFrame()


def fetch_kalshi_volume():
    """
    Fetch Kalshi volume data.
    Note: Kalshi doesn't have a public free API for historical volume.
    We'll try to get data from available sources or use placeholder.
    """
    print("Fetching Kalshi volume data...")
    
    # Try DefiLlama first (they might track Kalshi)
    url = "https://api.llama.fi/summary/dexs/kalshi"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        chart_data = data.get('totalDataChart', [])
        if chart_data:
            df = pd.DataFrame(chart_data, columns=['timestamp', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('date')
            return df
    except Exception as e:
        print(f"  DefiLlama doesn't have Kalshi data: {e}")
    
    # Try prediction markets endpoint
    url = "https://api.llama.fi/overview/prediction-markets"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        breakdown = data.get('totalDataChartBreakdown', [])
        if breakdown:
            records = []
            for entry in breakdown:
                timestamp = entry[0]
                volumes = entry[1]
                date = datetime.fromtimestamp(timestamp)
                
                # Find Kalshi volume
                kalshi_vol = 0
                for key, value in volumes.items():
                    if 'kalshi' in key.lower():
                        if isinstance(value, dict):
                            kalshi_vol = sum(value.values())
                        else:
                            kalshi_vol = value
                        break
                
                records.append({'date': date, 'volume': kalshi_vol})
            
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values('date')
                return df
    except Exception as e:
        print(f"  Error with prediction markets endpoint: {e}")
    
    return pd.DataFrame()


def format_millions(x, pos):
    """Format y-axis values as millions."""
    return f'${x:.0f}M'


def format_billions(x, pos):
    """Format y-axis values as billions."""
    if x >= 1:
        return f'${x:.1f}B'
    else:
        return f'${x*1000:.0f}M'


# ============================================================================
# Chart Generation
# ============================================================================

def create_volume_comparison_chart(kalshi_df, polymarket_df, output_path):
    """Create Kalshi vs Polymarket volume comparison chart."""
    print("Creating volume comparison chart...")
    
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('white')
    
    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=DAYS_LOOKBACK)
    
    # Filter and plot Polymarket
    if not polymarket_df.empty:
        poly_df = polymarket_df[polymarket_df['date'] >= cutoff_date].copy()
        poly_df['volume_billions'] = poly_df['volume'] / 1e9
        poly_df['volume_smooth'] = poly_df['volume_billions'].rolling(window=7, min_periods=1).mean()
        
        ax.plot(poly_df['date'], poly_df['volume_smooth'],
               label='Polymarket',
               color=POLYMARKET_COLOR,
               linewidth=2)
        print(f"  Polymarket: {len(poly_df)} days")
    
    # Filter and plot Kalshi
    if not kalshi_df.empty:
        kal_df = kalshi_df[kalshi_df['date'] >= cutoff_date].copy()
        kal_df['volume_billions'] = kal_df['volume'] / 1e9
        kal_df['volume_smooth'] = kal_df['volume_billions'].rolling(window=7, min_periods=1).mean()
        
        ax.plot(kal_df['date'], kal_df['volume_smooth'],
               label='Kalshi',
               color=KALSHI_COLOR,
               linewidth=2)
        print(f"  Kalshi: {len(kal_df)} days")
    
    # Title and source
    ax.text(-0.05, 1.08, 'Daily Trading Volume: Kalshi vs Polymarket (7-day avg)', 
            transform=ax.transAxes, fontsize=20, fontweight='bold', 
            verticalalignment='top')
    ax.text(-0.05, 1.03, 'Source: DefiLlama API', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', 
            verticalalignment='top', color='#666666')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_billions))
    
    # Format x-axis with dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    ax.tick_params(axis='both', which='major', length=3, labelsize=13, width=0.5, colors='#666666')
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=14, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#666666')
    ax.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Set x-axis to data range
    all_dates = []
    if not polymarket_df.empty:
        poly_filtered = polymarket_df[polymarket_df['date'] >= cutoff_date]
        if not poly_filtered.empty:
            all_dates.extend([poly_filtered['date'].min(), poly_filtered['date'].max()])
    if not kalshi_df.empty:
        kal_filtered = kalshi_df[kalshi_df['date'] >= cutoff_date]
        if not kal_filtered.empty:
            all_dates.extend([kal_filtered['date'].min(), kal_filtered['date'].max()])
    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("Fetching prediction market volume data...")
    print("=" * 60)
    
    # Fetch data
    polymarket_df = fetch_polymarket_volume()
    kalshi_df = fetch_kalshi_volume()
    
    if polymarket_df.empty and kalshi_df.empty:
        print("No data available for either platform.")
        return
    
    # Create chart
    output_path = r'D:\PROJECTS\poly\latency\kalshi_polymarket_volume.png'
    create_volume_comparison_chart(kalshi_df, polymarket_df, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    cutoff_date = datetime.now() - timedelta(days=30)
    
    if not polymarket_df.empty:
        recent_poly = polymarket_df[polymarket_df['date'] >= cutoff_date]
        if not recent_poly.empty:
            avg_vol = recent_poly['volume'].mean() / 1e9
            print(f"Polymarket (Avg Daily, Last 30 days): ${avg_vol:.2f}B")
    
    if not kalshi_df.empty:
        recent_kal = kalshi_df[kalshi_df['date'] >= cutoff_date]
        if not recent_kal.empty:
            avg_vol = recent_kal['volume'].mean() / 1e9
            print(f"Kalshi (Avg Daily, Last 30 days): ${avg_vol:.2f}B")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
