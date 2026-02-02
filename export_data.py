"""
Export Kalshi and Polymarket data to CSV files
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

DAYS_LOOKBACK = 365

def fetch_polymarket_volume():
    """Fetch Polymarket volume data from DefiLlama."""
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
            return df[['date', 'volume']]
    except Exception as e:
        print(f"  Error fetching Polymarket: {e}")
    
    return pd.DataFrame()


def fetch_kalshi_volume():
    """Fetch Kalshi volume data from DefiLlama."""
    print("Fetching Kalshi volume data...")
    
    # Try direct endpoint first
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
            return df[['date', 'volume']]
    except Exception as e:
        print(f"  Direct endpoint: {e}")
    
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
                return df[['date', 'volume']]
    except Exception as e:
        print(f"  Prediction markets endpoint: {e}")
    
    return pd.DataFrame()


def main():
    print("Exporting Kalshi and Polymarket data to CSV...")
    print("=" * 60)
    
    cutoff_date = datetime.now() - timedelta(days=DAYS_LOOKBACK)
    
    # Fetch and export Polymarket data
    polymarket_df = fetch_polymarket_volume()
    if not polymarket_df.empty:
        poly_filtered = polymarket_df[polymarket_df['date'] >= cutoff_date].copy()
        poly_filtered['volume_usd'] = poly_filtered['volume']
        poly_filtered['date'] = poly_filtered['date'].dt.strftime('%Y-%m-%d')
        poly_filtered[['date', 'volume_usd']].to_csv('polymarket_daily_volume.csv', index=False)
        print(f"Saved: polymarket_daily_volume.csv ({len(poly_filtered)} rows)")
    
    # Fetch and export Kalshi data
    kalshi_df = fetch_kalshi_volume()
    if not kalshi_df.empty:
        kal_filtered = kalshi_df[kalshi_df['date'] >= cutoff_date].copy()
        kal_filtered['volume_usd'] = kal_filtered['volume']
        kal_filtered['date'] = kal_filtered['date'].dt.strftime('%Y-%m-%d')
        kal_filtered[['date', 'volume_usd']].to_csv('kalshi_daily_volume.csv', index=False)
        print(f"Saved: kalshi_daily_volume.csv ({len(kal_filtered)} rows)")
    
    # Create combined CSV
    if not polymarket_df.empty and not kalshi_df.empty:
        poly_filtered = polymarket_df[polymarket_df['date'] >= cutoff_date].copy()
        kal_filtered = kalshi_df[kalshi_df['date'] >= cutoff_date].copy()
        
        poly_filtered = poly_filtered.rename(columns={'volume': 'polymarket_volume_usd'})
        kal_filtered = kal_filtered.rename(columns={'volume': 'kalshi_volume_usd'})
        
        combined = pd.merge(poly_filtered, kal_filtered, on='date', how='outer')
        combined = combined.sort_values('date')
        combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
        combined.to_csv('kalshi_polymarket_combined_volume.csv', index=False)
        print(f"Saved: kalshi_polymarket_combined_volume.csv ({len(combined)} rows)")
    
    # Export latency data if available
    try:
        # Latency data from the comparison script
        latency_data = {
            'platform': ['Kalshi', 'Kalshi', 'Polymarket', 'Polymarket'],
            'metric': ['taker_latency_ms', 'maker_latency_ms', 'taker_latency_ms', 'maker_latency_ms'],
            'description': ['End-to-end taker latency (30 samples)', 'End-to-end maker latency (100 samples)',
                           'End-to-end taker latency (30 samples)', 'End-to-end maker latency (100 samples)'],
            'source': ['AWS us-east-1 co-located', 'AWS us-east-1 co-located',
                      'AWS eu-west-1 co-located', 'AWS eu-west-1 co-located']
        }
        latency_df = pd.DataFrame(latency_data)
        latency_df.to_csv('latency_metadata.csv', index=False)
        print("Saved: latency_metadata.csv")
        
        # Fee comparison data
        fee_data = {
            'platform': ['Kalshi', 'Kalshi', 'Polymarket'],
            'fee_type': ['taker', 'maker', 'all'],
            'fee_bps': [7, 0, 0],
            'notes': ['7 bps on contracts $0.02-$0.98', '0 bps for makers', '0 bps for all trades']
        }
        fee_df = pd.DataFrame(fee_data)
        fee_df.to_csv('fee_comparison.csv', index=False)
        print("Saved: fee_comparison.csv")
    except Exception as e:
        print(f"  Note: Could not export latency/fee metadata: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
