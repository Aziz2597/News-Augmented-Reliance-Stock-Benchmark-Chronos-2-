import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from urllib.parse import urlparse, urlunparse
import os

def conservative_url_normalize(url: str) -> str:
    """Normalizes URLs conservatively while keeping query parameters intact."""
    if pd.isna(url):
        return ""
    parsed = urlparse(str(url).strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    if scheme == 'http' and netloc.endswith(':80'):
        netloc = netloc[:-3]
    elif scheme == 'https' and netloc.endswith(':443'):
        netloc = netloc[:-4]
        
    path = parsed.path.rstrip('/')
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))

def build_features():
    print("--- Starting Phase 2: Production Feature Engineering & Naive Baseline ---")

    # ---------------------------------------------------------
    # 1. Load and Validate NSE Price Data
    # ---------------------------------------------------------
    nse_filename = "RELIANCE_EQ_01-01-2025-TO-30-09-2026.csv"
    print(f"Loading NSE prices from {nse_filename}...")
    df_price = pd.read_csv(nse_filename)
    df_price.columns = df_price.columns.str.strip()
    
    # Strict Validations
    if 'Symbol' in df_price.columns and not (df_price['Symbol'] == 'RELIANCE').all():
        raise ValueError("NSE Data contains symbols other than 'RELIANCE'.")
        
    if 'Series' in df_price.columns:
        invalid_series_count = (~df_price['Series'].isin(['EQ', 'EQUITY'])).sum()
        if invalid_series_count > 0:
            print(f"Notice: Filtering out {invalid_series_count} non-standard trading segment rows (e.g., Block deals) to isolate the 'EQ' series.")
            df_price = df_price[df_price['Series'].isin(['EQ', 'EQUITY'])].copy()
        
    date_col = next((col for col in ['Date', 'TIMESTAMP', 'CH_TIMESTAMP'] if col in df_price.columns), df_price.columns[0])
    close_col = next((col for col in ['CLOSE', 'Close Price', 'CH_CLOSING_PRICE', 'Close', 'close'] if col in df_price.columns), None)
    
    if not close_col:
        raise ValueError(f"Could not find closing price column. Available: {list(df_price.columns)}")

    # Explicitly parse dates and catch unparseable formats rather than dropping
    try:
        # Usually NSE formats are like 01-Jan-2025, but we let pandas infer strictly first
        df_price['date'] = pd.to_datetime(df_price[date_col], format='mixed')
    except Exception as e:
        raise ValueError(f"Failed to parse dates in column '{date_col}': {e}")
        
    # Validate close prices are numeric and positive
    df_price['close'] = pd.to_numeric(df_price[close_col].astype(str).str.replace(',', ''), errors='coerce')
    if df_price['close'].isna().any() or (df_price['close'] <= 0).any():
        raise ValueError("Found missing, non-numeric, or non-positive closing prices.")
    
    df_price = df_price.sort_values('date').reset_index(drop=True)
    df_price = df_price[(df_price['date'] >= '2025-01-01') & (df_price['date'] <= '2026-09-30')].reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Load and Clean GDELT News Data
    # ---------------------------------------------------------
    print("Loading GDELT news data...")
    df_news = pd.read_csv("gdelt_reliance_raw.csv")
    df_news.columns = df_news.columns.str.strip()
    
    col_map = {c.lower(): c for c in df_news.columns}
    gkg_col = col_map.get('gkgrecordid')
    doc_col = col_map.get('article_url_raw') or col_map.get('documentidentifier')
    tone_col = col_map.get('first_tone') or col_map.get('raw_v2tone')
    
    if not gkg_col or not doc_col:
        raise ValueError(f"Could not find required GDELT columns. Available: {list(df_news.columns)}")

    # URL Normalization
    df_news['normalized_article_url'] = df_news[doc_col].apply(conservative_url_normalize)
    
    # Parse GDELT UTC timestamps from GKGRECORDID first 14 chars safely
    df_news['capture_str'] = df_news[gkg_col].astype(str).str.slice(0, 14)
    df_news['utc_time'] = pd.to_datetime(df_news['capture_str'], format='%Y%m%d%H%M%S', errors='coerce')
    if df_news['utc_time'].isna().any():
        raise ValueError("Invalid GKGRECORDID formats caused unparseable timestamps.")
    
    utc_zone = pytz.utc
    ist_zone = pytz.timezone('Asia/Kolkata')
    df_news['utc_time'] = df_news['utc_time'].dt.tz_localize(utc_zone)
    df_news['ist_time'] = df_news['utc_time'].dt.tz_convert(ist_zone)
    
    # Tone Parsing (Handling raw comma-separated lists if needed)
    if tone_col == "raw_v2tone" or tone_col == "v2tone":
        df_news['parsed_tone'] = pd.to_numeric(df_news[tone_col].astype("string").str.split(",").str[0], errors="coerce")
    else:
        df_news['parsed_tone'] = pd.to_numeric(df_news[tone_col], errors="coerce")
        
    df_news = df_news.sort_values(by=['normalized_article_url', 'utc_time', gkg_col])
    df_news_deduped = df_news.drop_duplicates(subset=['normalized_article_url'], keep='first').reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Vectorized Trading-Session Window Alignment
    # ---------------------------------------------------------
    print("Aligning articles to continuous NSE session intervals...")
    
    # Establish 18:00 IST origins for every trading session
    df_price['session_origin'] = df_price['date'].apply(
        lambda d: ist_zone.localize(datetime.combine(d.date(), time(18, 0, 0)))
    )
    
    # The bootstrap origin is exactly 1 day before the first trading session in our dataset
    bootstrap_origin = df_price['session_origin'].iloc[0] - timedelta(days=1)
    
    # Create contiguous boundaries: [bootstrap, origin_1, origin_2, ... origin_n]
    origin_bounds = [bootstrap_origin] + df_price['session_origin'].tolist()
    
    # Using pd.cut maps every article into the (previous_origin, current_origin] interval
    df_news_deduped['session_idx'] = pd.cut(
        df_news_deduped['ist_time'], 
        bins=origin_bounds, 
        labels=False, # Returns 0 for the first interval, 1 for the second, etc.
        right=True    # Interval is inclusive on the right (]
    )
    
    # Filter to articles that actually fell within our overall project window
    valid_news = df_news_deduped.dropna(subset=['session_idx']).copy()
    valid_news['session_idx'] = valid_news['session_idx'].astype(int)
    
    # Aggregate news per session index
    aggregated = valid_news.groupby('session_idx').agg(
        article_count=('normalized_article_url', 'size'),
        valid_tone_count=('parsed_tone', 'count'),
        mean_tone=('parsed_tone', 'mean')
    )
    
    # Reindex to ensure every trading session exists, even if 0 articles were found
    aggregated = aggregated.reindex(range(len(df_price)), fill_value=0)
    aggregated['mean_tone'] = aggregated['mean_tone'].fillna(0.0)
    aggregated['log_article_count'] = np.log1p(aggregated['article_count'])
    
    # Merge aggregated features back into the price dataframe
    df_price['article_count'] = aggregated['article_count'].values
    df_price['valid_tone_count'] = aggregated['valid_tone_count'].values
    df_price['mean_tone'] = aggregated['mean_tone'].values
    df_price['log_article_count'] = aggregated['log_article_count'].values
    
    # Add audit interval columns
    df_price['window_start_ist'] = [bootstrap_origin] + df_price['session_origin'].tolist()[:-1]
    df_price['window_end_ist'] = df_price['session_origin']

    # ---------------------------------------------------------
    # 4. Prepare Naive Baseline Prediction (P_{t+1} = P_t)
    # ---------------------------------------------------------
    df_price['naive_prediction'] = df_price['close'].shift(1)

    # ---------------------------------------------------------
    # 5. Clean up output columns and save
    # ---------------------------------------------------------
    output_cols = [
        'date', 'close', 'naive_prediction', 
        'window_start_ist', 'window_end_ist', 
        'article_count', 'valid_tone_count', 'mean_tone', 'log_article_count'
    ]
    df_final = df_price[output_cols]
    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/forecast_features_ready.csv"
    df_final.to_csv(output_path, index=False)
    print(f"Feature engineering successful! Processed {len(df_final)} trading sessions and saved to {output_path}.")
    
    # Print the first two rows to verify the bootstrap origin
    print("\n--- First 2 Intervals for Audit ---")
    print(df_final[['date', 'window_start_ist', 'window_end_ist']].head(2))
    
    return df_final

if __name__ == "__main__":
    build_features()