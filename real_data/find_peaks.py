import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import paths

from analyze import load_data, clean_data, get_topics
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_peaks():
    base_dir = paths.DATA_ANALYZE_DIR / 'raw_data'
    topics = get_topics(str(base_dir))
    
    results = {}
    
    for topic in topics:
        print(f"Analyzing {topic}...")
        df = load_data(base_dir, topic)
        if df is None or df.empty:
            continue
            
        df = clean_data(df)
        df['date'] = df['publish_time'].dt.date
        daily_active_users = df.groupby('date')['author_screen_name'].nunique().sort_index()
        
        # Convert to series with datetime index for easier handling
        ts = pd.Series(daily_active_users.values, index=pd.to_datetime(daily_active_users.index))
        
        current_prominence = ts.max() * 0.5
        current_height = ts.mean() * 1
        
        peaks, properties = find_peaks(ts.values, height=current_height, distance=3, prominence=current_prominence)
        
        peak_dates = ts.index[peaks]
        peak_values = ts.values[peaks]
        
        print(f"Topic: {topic}")
        print(f"Active User Peaks found at: {peak_dates.strftime('%Y-%m-%d').tolist()} with values {peak_values}")
        
        if len(peak_dates) >= 2:
            results[topic] = {
                'peaks': list(zip(peak_dates, peak_values)),
                'df': df  # Keep df in memory for analysis? might be too big. 
                          # Better to reload or filter now.
            }
            
            # Sort peaks by date
            sorted_peaks = sorted(list(zip(peak_dates, peak_values)), key=lambda x: x[0])
            
            # Check all peaks
            for i, (peak_date, peak_val) in enumerate(sorted_peaks):
                wave_name = "First wave" if i == 0 else f"Wave {i+1}"
                print(f"Analyzing {wave_name} peak (Active Users): {peak_date.strftime('%Y-%m-%d')}")
                
                # Get data for this date
                day_df = df[df['date'] == peak_date.date()]
                
                if day_df.empty:
                    print("No data found for this date (timezone mismatch?). Skipping.")
                    continue

                # 1. Top posts by reposts
                top_reposts = day_df.sort_values('reposts_count', ascending=False).head(5)
                print("Top reposted content:")
                for _, row in top_reposts.iterrows():
                    print(f"- [{row['reposts_count']} reposts] {row['content'][:50]}...")
                
                # 2. Top posts by comments
                top_comments = day_df.sort_values('comments_count', ascending=False).head(5)
                print("Top commented content:")
                for _, row in top_comments.iterrows():
                    print(f"- [{row['comments_count']} comments] {row['content'][:50]}...")
                    
                # 3. Frequent content (if many users post the same thing)
                top_content = day_df['content'].value_counts().head(5)
                print("Most frequent content:")
                for content, count in top_content.items():
                    print(f"- [{count} times] {content[:50]}...")

if __name__ == "__main__":
    analyze_peaks()
