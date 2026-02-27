import os
import pandas as pd
import glob
import re
from datetime import datetime
import sys

# Add parent directory to path to import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import paths

raw_data_dir = paths.DATA_ANALYZE_DIR / 'raw_data'

def get_movie_stats():
    movie_files = {}
    
    # Traverse directories
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~'):
                # Extract movie name (assume before first underscore or special handling)
                # Example: 疯狂动物城2_2025... -> 疯狂动物城2
                # Example: 无名之辈+意义非凡_2025... -> 无名之辈+意义非凡
                match = re.match(r'(.+?)_\d{8}', file)
                if match:
                    movie_name = match.group(1)
                    if movie_name not in movie_files:
                        movie_files[movie_name] = []
                    movie_files[movie_name].append(os.path.join(root, file))
    
    print(f"Found {len(movie_files)} movies.")
    
    stats = []
    
    for movie, files in movie_files.items():
        print(f"Processing {movie}...")
        total_posts = 0
        min_date = None
        
        # We need to find the earliest date first to avoid reading all data if possible,
        # but publish_time is inside files. We have to read them.
        # To save memory, we can read just publish_time.
        
        all_dates = []
        
        for file in files:
            try:
                # Read only publish_time column
                df = pd.read_excel(file, usecols=['publish_time'])
                if not df.empty:
                    total_posts += len(df)
                    # Convert to datetime
                    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
                    current_min = df['publish_time'].min()
                    
                    if pd.notnull(current_min):
                        if min_date is None or current_min < min_date:
                            min_date = current_min
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if min_date:
            stats.append({
                'movie': movie,
                'total_posts': total_posts,
                'start_date': min_date.date(),
                'files': files
            })
            
    return pd.DataFrame(stats)

if __name__ == "__main__":
    df_stats = get_movie_stats()
    print("\nMovie Statistics:")
    print(df_stats)
    df_stats.to_csv('movie_stats.csv', index=False)
