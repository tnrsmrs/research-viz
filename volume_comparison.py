"""
DEX Volume Comparison Charts
Comparing Perpetual and Spot trading volumes across major DEXs
Data source: DefiLlama API, Binance, OKX, Bybit APIs
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import time

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = r'D:\PROJECTS\poly\latency\research'

# DefiLlama API endpoints
DERIVATIVES_URL = "https://api.llama.fi/overview/derivatives"
DEX_SUMMARY_URL = "https://api.llama.fi/summary/dexs/{protocol}"

# CEX API endpoints for perpetual futures klines
CEX_CONFIGS = {
    "Binance": {
        "base_url": "https://fapi.binance.com/fapi/v1/klines",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "interval": "1d",
        "limit": 365,
    },
    "OKX": {
        "base_url": "https://www.okx.com/api/v5/market/candles",
        "symbols": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        "bar": "1D",
        "limit": 300,  # OKX max is 300
    },
    "Bybit": {
        "base_url": "https://api.bybit.com/v5/market/kline",
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "interval": "D",
        "category": "linear",
        "limit": 200,
    },
}

# Protocols to compare
# Note: Aster perps was delisted from DefiLlama due to data integrity concerns, but spot is still available
PERPS_PROTOCOLS = ["Hyperliquid", "dYdX", "Drift", "Paradex", "GMX"]  # Names as they appear in DefiLlama

# Spot protocols: Orderbook DEXs vs AMM DEXs (V3 versions)
SPOT_PROTOCOLS = {
    "hyperliquid": "Hyperliquid",
    "aster": "Aster",
    "uniswap-v3": "Uniswap V3", 
    "pancakeswap-amm-v3": "PancakeSwap V3"
}

# Perp DEX Architecture Comparison
HYBRID_PERPS = ["Hyperliquid", "Lighter", "Aster"]  # Hybrid (off-chain matching, on-chain settlement)
DECENTRALIZED_PERPS = ["dYdX", "GMX", "Jupiter"]  # Fully decentralized

# Brand colors for each protocol/exchange
COLORS = {
    # Perp DEXs
    "Hyperliquid": "#84cc16",  # Lime green
    "dYdX": "#6366f1",         # Indigo
    "Drift": "#f59e0b",        # Amber
    "Paradex": "#e11d48",      # Rose/Red
    "GMX": "#2d42fc",          # GMX blue
    "Lighter": "#9333ea",      # Purple
    "Aster": "#f97316",        # Orange
    "Jupiter": "#22c55e",      # Green
    # Spot DEXs
    "Uniswap V3": "#ff007a",   # Pink
    "PancakeSwap V3": "#d4a017",  # Gold
    "Aster": "#f97316",        # Orange
    # CEXes
    "Binance": "#f0b90b",      # Binance yellow
    "OKX": "#000000",          # OKX black
    "Bybit": "#f7a600",        # Bybit orange
}

# Time ranges
DAYS_LOOKBACK = 730      # 2 years for perps architecture chart
DAYS_LOOKBACK_SPOT = 365  # 1 year for spot chart

# ============================================================================
# Data Fetching Functions
# ============================================================================

def fetch_derivatives_data():
    """Fetch derivatives/perps volume data from DefiLlama."""
    print("Fetching derivatives data...")
    response = requests.get(DERIVATIVES_URL, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_dex_data(protocol_slug):
    """Fetch DEX spot volume data for a specific protocol."""
    print(f"Fetching DEX data for {protocol_slug}...")
    url = DEX_SUMMARY_URL.format(protocol=protocol_slug)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_chart_data(data_list, cutoff_timestamp):
    """
    Parse [timestamp, volume] pairs into a DataFrame.
    Filter to entries after cutoff_timestamp.
    """
    df = pd.DataFrame(data_list, columns=['timestamp', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[df['timestamp'] >= cutoff_timestamp]
    df = df.sort_values('date')
    return df


def extract_protocol_from_breakdown(breakdown_data, protocol_names, cutoff_timestamp):
    """
    Extract daily volumes for specific protocols from breakdown data.
    breakdown_data format: [[timestamp, {"Protocol1": vol1, "Protocol2": vol2, ...}], ...]
    """
    records = []
    
    for entry in breakdown_data:
        timestamp = entry[0]
        if timestamp < cutoff_timestamp:
            continue
            
        volumes = entry[1]
        date = datetime.fromtimestamp(timestamp)
        
        record = {'date': date, 'timestamp': timestamp}
        
        for protocol in protocol_names:
            # Handle variations in protocol names
            vol = 0
            for key, value in volumes.items():
                # Check if protocol name is in the key (case insensitive)
                if protocol.lower() in key.lower():
                    if isinstance(value, dict):
                        # Nested structure: sum all sub-values
                        vol += sum(value.values())
                    else:
                        vol += value
                    break
            record[protocol] = vol
        
        records.append(record)
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('date')
    return df


# ============================================================================
# CEX Data Fetching Functions
# ============================================================================

def fetch_binance_klines(symbol, limit=365):
    """
    Fetch daily klines from Binance Futures.
    Returns list of [timestamp, open, high, low, close, volume, ...].
    Volume is in base asset (e.g., BTC), quote volume is in USDT.
    Only uses actual futures endpoints (not spot as fallback).
    """
    # Only use actual futures endpoints - we need perp data, not spot
    endpoints = [
        "https://fapi.binance.com/fapi/v1/klines",    # USDT-M Futures
        "https://dapi.binance.com/dapi/v1/klines",    # COIN-M Futures (different symbols)
    ]
    
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit,
    }
    
    for url in endpoints:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [451, 403]:
                continue  # Try next endpoint
            raise
    
    raise Exception("Binance futures endpoints blocked (regional restriction)")


def fetch_okx_klines(symbol, limit=300):
    """
    Fetch daily klines from OKX perpetual swaps.
    Returns list of [timestamp, open, high, low, close, vol, volCcy, volCcyQuote, confirm].
    """
    url = CEX_CONFIGS["OKX"]["base_url"]
    params = {
        "instId": symbol,
        "bar": "1D",
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


def fetch_bybit_klines(symbol, limit=200):
    """
    Fetch daily klines from Bybit perpetual.
    Returns list of [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover].
    Tries multiple endpoints for regional availability.
    """
    # Try multiple Bybit endpoints
    endpoints = [
        "https://api.bybit.com/v5/market/kline",
        "https://api.bytick.com/v5/market/kline",  # Alternative domain
    ]
    
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": "D",
        "limit": limit,
    }
    
    for url in endpoints:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {}).get("list", [])
            if result:
                return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [451, 403]:
                continue
            raise
    
    raise Exception("All Bybit endpoints blocked")


def fetch_cex_perps_volume():
    """
    Fetch perpetual futures volume from CEXes (Binance, OKX, Bybit).
    Uses BTC and ETH pairs and sums their quote volumes (USDT).
    Returns dict of {exchange_name: DataFrame with date and volume columns}.
    """
    cex_data = {}
    
    # Binance
    print("Fetching Binance perpetual klines...")
    try:
        binance_records = {}
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            klines = fetch_binance_klines(symbol)
            for k in klines:
                # k = [open_time, open, high, low, close, volume, close_time, quote_volume, ...]
                ts = k[0] // 1000  # Convert ms to seconds
                date = datetime.fromtimestamp(ts).date()
                quote_vol = float(k[7])  # Quote asset volume (USDT)
                if date not in binance_records:
                    binance_records[date] = 0
                binance_records[date] += quote_vol
            time.sleep(0.2)  # Rate limiting
        
        if binance_records:
            df = pd.DataFrame([
                {"date": pd.Timestamp(d), "volume": v}
                for d, v in binance_records.items()
            ]).sort_values("date")
            cex_data["Binance"] = df
            print(f"  Binance: {len(df)} days")
    except Exception as e:
        print(f"  Binance error: {e}")
    
    # OKX
    print("Fetching OKX perpetual klines...")
    try:
        okx_records = {}
        for symbol in ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]:
            klines = fetch_okx_klines(symbol)
            for k in klines:
                # k = [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                ts = int(k[0]) // 1000  # Convert ms to seconds
                date = datetime.fromtimestamp(ts).date()
                # volCcyQuote is the quote volume in USDT
                quote_vol = float(k[7]) if len(k) > 7 else float(k[5])
                if date not in okx_records:
                    okx_records[date] = 0
                okx_records[date] += quote_vol
            time.sleep(0.2)
        
        if okx_records:
            df = pd.DataFrame([
                {"date": pd.Timestamp(d), "volume": v}
                for d, v in okx_records.items()
            ]).sort_values("date")
            cex_data["OKX"] = df
            print(f"  OKX: {len(df)} days")
    except Exception as e:
        print(f"  OKX error: {e}")
    
    # Bybit
    print("Fetching Bybit perpetual klines...")
    try:
        bybit_records = {}
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            klines = fetch_bybit_klines(symbol)
            for k in klines:
                # k = [startTime, open, high, low, close, volume, turnover]
                # turnover is in quote currency (USDT)
                ts = int(k[0]) // 1000  # Convert ms to seconds
                date = datetime.fromtimestamp(ts).date()
                quote_vol = float(k[6])  # Turnover in USDT
                if date not in bybit_records:
                    bybit_records[date] = 0
                bybit_records[date] += quote_vol
            time.sleep(0.2)
        
        if bybit_records:
            df = pd.DataFrame([
                {"date": pd.Timestamp(d), "volume": v}
                for d, v in bybit_records.items()
            ]).sort_values("date")
            cex_data["Bybit"] = df
            print(f"  Bybit: {len(df)} days")
    except Exception as e:
        print(f"  Bybit error: {e}")
    
    return cex_data


def get_hyperliquid_perps_volume(cutoff_timestamp):
    """
    Extract Hyperliquid's total perpetual volume from DefiLlama.
    Returns DataFrame with date and volume columns.
    """
    derivatives_data = fetch_derivatives_data()
    breakdown = derivatives_data.get('totalDataChartBreakdown', [])
    
    records = []
    for entry in breakdown:
        timestamp = entry[0]
        if timestamp < cutoff_timestamp:
            continue
        
        volumes = entry[1]
        date = datetime.fromtimestamp(timestamp)
        
        # Find Hyperliquid volume
        vol = 0
        for key, value in volumes.items():
            if "hyperliquid" in key.lower():
                if isinstance(value, dict):
                    vol = sum(value.values())
                else:
                    vol = value
                break
        
        records.append({"date": date, "volume": vol})
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("date")
    return df


# ============================================================================
# Chart Styling (Galaxy Research style)
# ============================================================================

def setup_style():
    """Set up matplotlib style matching Galaxy Research charts."""
    sns.set_style("white")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5


def style_axis(ax):
    """Apply minimal axis styling."""
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(axis='both', which='major', length=3, labelsize=13, width=0.5, colors='#666666')
    ax.grid(axis='y', linestyle='-', alpha=0.3, linewidth=0.5, color='#cccccc')


def format_billions(x, pos):
    """Format y-axis values as billions."""
    return f'${x:.0f}B'


# ============================================================================
# Chart Generation Functions
# ============================================================================

def create_perps_chart(df, output_path):
    """Create DEX Perpetuals volume comparison chart."""
    print("Creating perps volume chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Contrasting colors for better visibility
    perps_colors = {
        "Hyperliquid": "#000000",  # Black
        "dYdX": "#5dade2",         # Light blue
        "Drift": "#f39c12",        # Orange
        "Paradex": "#e74c3c",      # Red
        "GMX": "#9b59b6",          # Purple
    }
    
    # Convert volumes to billions
    for protocol in PERPS_PROTOCOLS:
        if protocol in df.columns:
            df[f'{protocol}_billions'] = df[protocol] / 1e9
    
    # Apply 7-day rolling average for smoother visualization
    for protocol in PERPS_PROTOCOLS:
        col = f'{protocol}_billions'
        if col in df.columns:
            df[f'{protocol}_smooth'] = df[col].rolling(window=7, min_periods=1).mean()
    
    # Plot each protocol with thinner lines
    for protocol in PERPS_PROTOCOLS:
        smooth_col = f'{protocol}_smooth'
        if smooth_col in df.columns:
            ax.plot(df['date'], df[smooth_col], 
                   label=protocol, 
                   color=perps_colors.get(protocol, '#888888'),
                   linewidth=1.2)
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'DEX Perpetuals Daily Volume (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
    ax.text(-0.05, 1.03, 'Source: DefiLlama API', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', 
            verticalalignment='top', color='#666666')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_billions))
    
    # Format x-axis with dates - more ticks for detail
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    ax.set_xlim(df['date'].min(), df['date'].max())
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


def create_spot_chart(protocol_data, output_path):
    """Create DEX Spot volume comparison chart."""
    print("Creating spot volume chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Contrasting colors
    spot_colors = {
        "Hyperliquid": "#000000",    # Black
        "Aster": "#5dade2",          # Light blue
        "Uniswap V3": "#e74c3c",     # Red
        "PancakeSwap V3": "#f39c12", # Orange
    }
    
    # Plot each protocol
    for protocol_name, df in protocol_data.items():
        if df.empty:
            continue
            
        # Convert to billions and smooth
        df = df.copy()
        df['volume_billions'] = df['volume'] / 1e9
        df['volume_smooth'] = df['volume_billions'].rolling(window=7, min_periods=1).mean()
        
        ax.plot(df['date'], df['volume_smooth'],
               label=protocol_name,
               color=spot_colors.get(protocol_name, '#888888'),
               linewidth=1.2)
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'DEX Spot Daily Volume (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
    ax.text(-0.05, 1.03, 'Source: DefiLlama API', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', 
            verticalalignment='top', color='#666666')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_billions))
    
    # Format x-axis with dates - more ticks for detail
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    all_dates = []
    for name, df in protocol_data.items():
        if not df.empty:
            all_dates.extend([df['date'].min(), df['date'].max()])
    if all_dates:
        ax.set_xlim(min(all_dates), max(all_dates))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


def create_cex_vs_hyperliquid_chart(cex_data, hyperliquid_df, output_path):
    """Create CEX vs Hyperliquid perpetual volume comparison chart."""
    print("Creating CEX vs Hyperliquid perps chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Contrasting colors
    cex_colors = {
        "Hyperliquid": "#000000",  # Black
        "Binance": "#5dade2",      # Light blue
        "OKX": "#e74c3c",          # Red
        "Bybit": "#f39c12",        # Orange
    }
    
    # Find common date range
    all_dates = set()
    for name, df in cex_data.items():
        all_dates.update(df['date'].dt.date)
    if not hyperliquid_df.empty:
        all_dates.update(hyperliquid_df['date'].dt.date)
    
    min_date = max(df['date'].min() for df in cex_data.values())
    
    # Plot Hyperliquid first
    if not hyperliquid_df.empty:
        df = hyperliquid_df[hyperliquid_df['date'] >= min_date].copy()
        df['volume_billions'] = df['volume'] / 1e9
        df['volume_smooth'] = df['volume_billions'].rolling(window=7, min_periods=1).mean()
        
        ax.plot(df['date'], df['volume_smooth'],
               label='Hyperliquid',
               color=cex_colors['Hyperliquid'],
               linewidth=1.2)
    
    # Plot CEXes
    for name, df in cex_data.items():
        df = df[df['date'] >= min_date].copy()
        df['volume_billions'] = df['volume'] / 1e9
        df['volume_smooth'] = df['volume_billions'].rolling(window=7, min_periods=1).mean()
        
        ax.plot(df['date'], df['volume_smooth'],
               label=f'{name} (BTC+ETH)',
               color=cex_colors.get(name, '#888888'),
               linewidth=1.2)
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'Perpetuals Daily Volume: Hyperliquid vs CEX (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
    ax.text(-0.05, 1.03, 'Source: DefiLlama API, OKX API', 
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
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    all_dates = [hyperliquid_df['date'].min(), hyperliquid_df['date'].max()]
    for name, cex_df in cex_data.items():
        all_dates.extend([cex_df['date'].min(), cex_df['date'].max()])
    ax.set_xlim(min(all_dates), max(all_dates))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


def create_architecture_perps_chart(df, output_path):
    """
    Create Perp DEX comparison chart by architecture type.
    Hybrid DEXs (solid lines) vs Fully Decentralized (dashed lines).
    """
    print("Creating architecture comparison perps chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Contrasting colors - solid for hybrid, lighter shades for decentralized
    arch_colors = {
        # Hybrid (solid darker colors)
        "Hyperliquid": "#000000",  # Black
        "Lighter": "#5dade2",      # Light blue
        "Aster": "#e74c3c",        # Red
        # Decentralized (lighter shades)
        "dYdX": "#9b59b6",         # Purple
        "GMX": "#f39c12",          # Orange
        "Jupiter": "#27ae60",      # Green
    }
    
    all_protocols = HYBRID_PERPS + DECENTRALIZED_PERPS
    
    # Convert volumes to billions
    for protocol in all_protocols:
        if protocol in df.columns:
            df[f'{protocol}_billions'] = df[protocol] / 1e9
    
    # Apply 7-day rolling average for smoother visualization
    for protocol in all_protocols:
        col = f'{protocol}_billions'
        if col in df.columns:
            df[f'{protocol}_smooth'] = df[col].rolling(window=7, min_periods=1).mean()
    
    # Plot Hybrid DEXs (solid lines)
    for protocol in HYBRID_PERPS:
        smooth_col = f'{protocol}_smooth'
        if smooth_col in df.columns:
            ax.plot(df['date'], df[smooth_col], 
                   label=f'{protocol} (Hybrid)', 
                   color=arch_colors.get(protocol, '#888888'),
                   linewidth=1.2,
                   linestyle='-')
    
    # Plot Fully Decentralized DEXs (dashed lines)
    for protocol in DECENTRALIZED_PERPS:
        smooth_col = f'{protocol}_smooth'
        if smooth_col in df.columns:
            ax.plot(df['date'], df[smooth_col], 
                   label=f'{protocol} (Decentralized)', 
                   color=arch_colors.get(protocol, '#888888'),
                   linewidth=1.2,
                   linestyle='--')
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'Perp DEX Volume: Hybrid vs Fully Decentralized (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
    ax.text(-0.05, 1.03, 'Source: DefiLlama API', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', 
            verticalalignment='top', color='#666666')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_billions))
    
    # Format x-axis with dates - more ticks for detail
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    # Legend in center top with two rows
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    ax.set_xlim(df['date'].min(), df['date'].max())
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


# Spot DEX categories for aggregate chart
ORDERBOOK_SPOT = ["Hyperliquid", "Aster"]
AMM_SPOT = ["Uniswap V3", "PancakeSwap V3"]


def create_architecture_aggregate_chart(df, output_path):
    """
    Create aggregate comparison chart: Total Hybrid vs Total Decentralized volume.
    """
    print("Creating aggregate architecture comparison chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Sum volumes for each architecture type
    df = df.copy()
    
    # Hybrid total
    hybrid_cols = [p for p in HYBRID_PERPS if p in df.columns]
    df['Hybrid_total'] = df[hybrid_cols].sum(axis=1) / 1e9
    df['Hybrid_smooth'] = df['Hybrid_total'].rolling(window=7, min_periods=1).mean()
    
    # Decentralized total
    decentral_cols = [p for p in DECENTRALIZED_PERPS if p in df.columns]
    df['Decentralized_total'] = df[decentral_cols].sum(axis=1) / 1e9
    df['Decentralized_smooth'] = df['Decentralized_total'].rolling(window=7, min_periods=1).mean()
    
    # Plot with contrasting colors
    ax.plot(df['date'], df['Hybrid_smooth'], 
           label='Hybrid DEXs (Hyperliquid, Lighter, Aster)', 
           color='#000000',  # Black
           linewidth=1.2)
    
    ax.plot(df['date'], df['Decentralized_smooth'], 
           label='Decentralized DEXs (dYdX, GMX, Jupiter)', 
           color='#5dade2',  # Light blue
           linewidth=1.2)
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'Perp DEX Total Volume: Hybrid vs Decentralized (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
    ax.text(-0.05, 1.03, 'Source: DefiLlama API', 
            transform=ax.transAxes, fontsize=12, fontweight='normal', 
            verticalalignment='top', color='#666666')
    
    # Styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_billions))
    
    # Format x-axis with dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=0, ha='center', fontsize=12)
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    ax.set_xlim(df['date'].min(), df['date'].max())
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


def create_spot_aggregate_chart(protocol_data, output_path):
    """
    Create aggregate comparison chart: Total Orderbook vs Total AMM spot volume.
    """
    print("Creating aggregate spot comparison chart...")
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # Find common date range
    all_dates = set()
    for name, df in protocol_data.items():
        all_dates.update(df['date'].dt.date)
    
    # Create aligned dataframe
    date_range = sorted(all_dates)
    aligned_data = {'date': [pd.Timestamp(d) for d in date_range]}
    
    for name, df in protocol_data.items():
        df_indexed = df.set_index(df['date'].dt.date)['volume']
        aligned_data[name] = [df_indexed.get(d, 0) for d in date_range]
    
    combined_df = pd.DataFrame(aligned_data)
    
    # Sum volumes for each category
    orderbook_cols = [p for p in ORDERBOOK_SPOT if p in combined_df.columns]
    amm_cols = [p for p in AMM_SPOT if p in combined_df.columns]
    
    combined_df['Orderbook_total'] = combined_df[orderbook_cols].sum(axis=1) / 1e9
    combined_df['Orderbook_smooth'] = combined_df['Orderbook_total'].rolling(window=7, min_periods=1).mean()
    
    combined_df['AMM_total'] = combined_df[amm_cols].sum(axis=1) / 1e9
    combined_df['AMM_smooth'] = combined_df['AMM_total'].rolling(window=7, min_periods=1).mean()
    
    # Plot with contrasting colors
    ax.plot(combined_df['date'], combined_df['Orderbook_smooth'], 
           label='Orderbook DEXs (Hyperliquid, Aster)', 
           color='#000000',  # Black
           linewidth=1.2)
    
    ax.plot(combined_df['date'], combined_df['AMM_smooth'], 
           label='AMM DEXs (Uniswap V3, PancakeSwap V3)', 
           color='#5dade2',  # Light blue
           linewidth=1.2)
    
    # Title in top left, bolded
    ax.text(-0.05, 1.08, 'Spot DEX Total Volume: Orderbook vs AMM (7-day avg)', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='top')
    # Source under title, unbolded
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
    
    # Legend in center top
    ax.legend(frameon=False, fontsize=12, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    
    # Apply styling
    style_axis(ax)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    # Set x-axis to data range (no whitespace)
    ax.set_xlim(combined_df['date'].min(), combined_df['date'].max())
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    # Calculate cutoff timestamp (365 days ago)
    cutoff_date = datetime.now() - timedelta(days=DAYS_LOOKBACK)
    cutoff_timestamp = int(cutoff_date.timestamp())
    
    print(f"Fetching data from {cutoff_date.strftime('%Y-%m-%d')} to now...")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Chart 1: DEX Perpetuals Volume
    # -------------------------------------------------------------------------
    try:
        derivatives_data = fetch_derivatives_data()
        breakdown = derivatives_data.get('totalDataChartBreakdown', [])
        
        if breakdown:
            perps_df = extract_protocol_from_breakdown(breakdown, PERPS_PROTOCOLS, cutoff_timestamp)
            
            if not perps_df.empty:
                # Skip individual perps chart - only generate aggregate
                pass
                
                # Print summary stats
                print("\nPerps Volume Summary (Avg Daily, Last 30 days):")
                recent = perps_df.tail(30)
                for protocol in PERPS_PROTOCOLS:
                    if protocol in recent.columns:
                        avg_vol = recent[protocol].mean() / 1e9
                        print(f"  {protocol}: ${avg_vol:.2f}B")
            else:
                print("No perps data found for the specified protocols.")
        
        # -------------------------------------------------------------------------
        # Chart 1b: Perp DEX Architecture Comparison (Hybrid vs Decentralized)
        # -------------------------------------------------------------------------
        print("\n" + "-" * 40)
        print("Creating architecture comparison chart...")
        
        all_arch_protocols = HYBRID_PERPS + DECENTRALIZED_PERPS
        arch_df = extract_protocol_from_breakdown(breakdown, all_arch_protocols, cutoff_timestamp)
        
        if not arch_df.empty:
            # Skip individual architecture chart - only generate aggregate
            pass
            
            # Print summary stats
            print("\nHybrid Perp DEXs (Avg Daily, Last 30 days):")
            recent = arch_df.tail(30)
            for protocol in HYBRID_PERPS:
                if protocol in recent.columns:
                    avg_vol = recent[protocol].mean() / 1e9
                    print(f"  {protocol}: ${avg_vol:.2f}B")
            
            print("\nFully Decentralized Perp DEXs (Avg Daily, Last 30 days):")
            for protocol in DECENTRALIZED_PERPS:
                if protocol in recent.columns:
                    avg_vol = recent[protocol].mean() / 1e9
                    print(f"  {protocol}: ${avg_vol:.2f}B")
            
            # Also create aggregate chart
            agg_output = f"{OUTPUT_DIR}\\perps_aggregate.png"
            create_architecture_aggregate_chart(arch_df, agg_output)
            
            # Print aggregate summary
            hybrid_cols = [p for p in HYBRID_PERPS if p in recent.columns]
            decentral_cols = [p for p in DECENTRALIZED_PERPS if p in recent.columns]
            hybrid_total = recent[hybrid_cols].sum(axis=1).mean() / 1e9
            decentral_total = recent[decentral_cols].sum(axis=1).mean() / 1e9
            print(f"\nAggregate (Avg Daily, Last 30 days):")
            print(f"  Total Hybrid: ${hybrid_total:.2f}B")
            print(f"  Total Decentralized: ${decentral_total:.2f}B")
        else:
            print("No architecture comparison data available.")
            
    except Exception as e:
        print(f"Error fetching derivatives data: {e}")
    
    print("\n" + "=" * 60)
    
    # -------------------------------------------------------------------------
    # Chart 2: DEX Spot Volume (1 year lookback)
    # -------------------------------------------------------------------------
    spot_cutoff_date = datetime.now() - timedelta(days=DAYS_LOOKBACK_SPOT)
    spot_cutoff_timestamp = int(spot_cutoff_date.timestamp())
    print(f"Spot data from {spot_cutoff_date.strftime('%Y-%m-%d')} to now...")
    
    spot_data = {}
    
    for slug, name in SPOT_PROTOCOLS.items():
        try:
            data = fetch_dex_data(slug)
            chart_data = data.get('totalDataChart', [])
            
            if chart_data:
                df = parse_chart_data(chart_data, spot_cutoff_timestamp)
                if not df.empty:
                    spot_data[name] = df
                    print(f"  {name}: {len(df)} data points")
                else:
                    print(f"  {name}: No data in time range")
            else:
                print(f"  {name}: No chart data available")
                
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    if spot_data:
        # Skip individual spot chart - only generate aggregate
        pass
        
        # Print summary stats
        print("\nSpot Volume Summary (Avg Daily, Last 30 days):")
        for name, df in spot_data.items():
            recent = df.tail(30)
            avg_vol = recent['volume'].mean() / 1e9
            print(f"  {name}: ${avg_vol:.2f}B")
        
        # Create aggregate spot chart
        spot_agg_output = f"{OUTPUT_DIR}\\spot_aggregate.png"
        create_spot_aggregate_chart(spot_data, spot_agg_output)
        
        # Print aggregate summary
        orderbook_total = sum(
            spot_data[p].tail(30)['volume'].mean() / 1e9 
            for p in ORDERBOOK_SPOT if p in spot_data
        )
        amm_total = sum(
            spot_data[p].tail(30)['volume'].mean() / 1e9 
            for p in AMM_SPOT if p in spot_data
        )
        print(f"\nAggregate Spot (Avg Daily, Last 30 days):")
        print(f"  Total Orderbook: ${orderbook_total:.2f}B")
        print(f"  Total AMM: ${amm_total:.2f}B")
    else:
        print("No spot volume data available.")
    
    print("\n" + "=" * 60)
    
    # -------------------------------------------------------------------------
    # Chart 3: CEX vs Hyperliquid Perpetual Volume
    # -------------------------------------------------------------------------
    print("\nFetching CEX perpetual volume data (BTC + ETH pairs)...")
    
    try:
        # Fetch CEX data
        cex_data = fetch_cex_perps_volume()
        
        # Get Hyperliquid data
        print("\nGetting Hyperliquid perps volume...")
        hyperliquid_df = get_hyperliquid_perps_volume(cutoff_timestamp)
        print(f"  Hyperliquid: {len(hyperliquid_df)} days")
        
        if cex_data and not hyperliquid_df.empty:
            # Skip CEX vs Hyperliquid chart
            pass
            
            # Print summary stats
            print("\nCEX vs Hyperliquid Volume Summary (Avg Daily, Last 30 days):")
            
            # Hyperliquid
            recent_hl = hyperliquid_df.tail(30)
            avg_hl = recent_hl['volume'].mean() / 1e9
            print(f"  Hyperliquid (all pairs): ${avg_hl:.2f}B")
            
            # CEXes
            for name, df in cex_data.items():
                recent = df.tail(30)
                avg_vol = recent['volume'].mean() / 1e9
                print(f"  {name} (BTC+ETH only): ${avg_vol:.2f}B")
        else:
            print("Insufficient data for CEX vs Hyperliquid chart.")
            
    except Exception as e:
        print(f"Error creating CEX vs Hyperliquid chart: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
