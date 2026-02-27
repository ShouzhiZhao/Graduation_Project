from calendar import c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import networkx as nx
import tqdm
from concurrent.futures import ProcessPoolExecutor

# Set Chinese font
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_sim_data(sim_data_path, sim_net_file):
    print(f"Loading sim network...")
    df = pd.read_csv(sim_net_file)
    G = nx.from_pandas_edgelist(df, 'user_1', 'user_2')
    print(f"Sim Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    npz_files = []
    if os.path.isfile(sim_data_path) and sim_data_path.endswith('.npz'):
        npz_files.append(sim_data_path)
    elif os.path.isdir(sim_data_path):
        print(f"Searching for .npz files in {sim_data_path}...")
        for root, dirs, files in os.walk(sim_data_path):
            for file in files:
                if file.endswith(".npz"):
                    npz_files.append(os.path.join(root, file))
    
    if not npz_files:
        print(f"No .npz files found in {sim_data_path}")
        return None, G.number_of_nodes()

    print(f"Found {len(npz_files)} simulation files. Loading and averaging...")
    
    all_active_counts = []
    all_cum_viewers = []
    all_cum_view_counts = []

    for f in npz_files:
        data = np.load(f)
        X = data['X']
        
        # 1. Active Counts (Current Active Users)
        active_counts = np.sum(X, axis=1)
        
        # 2. Cumulative Viewers (Unique users ever active)
        # np.maximum.accumulate(X, axis=0) ensures that once a node is 1, it stays 1.
        cum_viewers = np.sum(np.maximum.accumulate(X, axis=0), axis=1)
        
        # 3. Cumulative View Counts (Total Person-Times/Activity Volume)
        # Sum of active users at each step accumulated over time
        cum_view_counts = np.cumsum(active_counts)

        all_active_counts.append(active_counts)
        all_cum_viewers.append(cum_viewers)
        all_cum_view_counts.append(cum_view_counts)

    # Align lengths (truncate to min length)
    min_len = min(len(a) for a in all_active_counts)
    
    def align_and_mean(arrays, length):
        aligned = [a[:length] for a in arrays]
        return np.mean(aligned, axis=0)

    avg_active_counts = align_and_mean(all_active_counts, min_len)
    avg_cum_viewers = align_and_mean(all_cum_viewers, min_len)
    avg_cum_view_counts = align_and_mean(all_cum_view_counts, min_len)
    
    print(f"Averaged over {len(all_active_counts)} runs (min_len={min_len}).")
    
    return {
        'active_counts': avg_active_counts,
        'cum_viewers': avg_cum_viewers,
        'cum_view_counts': avg_cum_view_counts
    }, G.number_of_nodes()


def read_single_excel(f):
    return pd.read_excel(f, usecols=['publish_time', 'author_screen_name', 'retweeted_status', 'message_type'])

def load_real_data(real_data_path):
    print("Loading real data...")
    real_files = []
    for root, dirs, files in os.walk(real_data_path):
        for file in files:
            if file.startswith("疯狂动物城2") and file.endswith(".xlsx"):
                real_files.append(os.path.join(root, file))
    dfs = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(read_single_excel, real_files)
        pbar = tqdm.tqdm(total=len(real_files), desc="Loading real data")
        for i, df in enumerate(results):
            dfs.append(df)
            pbar.update(1)
        pbar.close()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df['publish_time'] = pd.to_datetime(full_df['publish_time'])
    
    # Filter only last 30 days
    all_dates = sorted(full_df['publish_time'].dt.date.unique())
    start_date = all_dates[-30]
    full_df = full_df[full_df['publish_time'].dt.date >= start_date]
        
    full_df = full_df.sort_values('publish_time')
    print(f"Filtered Real Data (Last 30 Days): {len(full_df)} records")
    
    # Build Network Nodes (Total Population)
    users = set(full_df['author_screen_name'].dropna().unique())
    
    def extract_author(row):
        if pd.isna(row):
            return None
        if isinstance(row, str):
            data = json.loads(row)
            return data.get('author_name')
        return None

    retweet_rows = full_df['retweeted_status'].dropna()
    print(f"Parsing {len(retweet_rows)} retweet records for network construction...")
        
    authors = retweet_rows.apply(extract_author)
    users.update(authors.dropna().unique())
    
    total_nodes = len(users)
    print(f"Estimated Real Network Nodes: {total_nodes}")
    
    # Calculate Activity (Daily Unique Authors)
    full_df = full_df.set_index('publish_time')
    activity_counts = full_df.resample('D')['author_screen_name'].nunique()
    
    # Cumulative View Counts (Accumulated Posts)
    daily_posts = full_df.resample('D').size()
    cum_view_counts = daily_posts.cumsum()
    
    # Cumulative Viewers (Accumulated Unique Authors)
    df_reset = full_df.reset_index()
    first_sightings = df_reset.drop_duplicates(subset='author_screen_name', keep='first')
    new_users_daily = first_sightings.set_index('publish_time').resample('D').size()
    new_users_daily = new_users_daily.reindex(activity_counts.index, fill_value=0)
    cum_viewers = new_users_daily.cumsum()
    
    return {
        'active_counts': activity_counts,
        'cum_viewers': cum_viewers,
        'cum_view_counts': cum_view_counts
    }, total_nodes


def main():
    # 1. Sim Data
    sim_net_file = '/mlp_vepfs/share/zsz/simulation_agent/data/relationship_1000.csv'
    sim_file_path = '/mlp_vepfs/share/zsz/simulation_agent/simulation_data/Toy_Story/temp/'
    sim_data, sim_total_nodes = load_sim_data(sim_file_path, sim_net_file)
    
    # 2. Real Data
    real_data_path = '/mlp_vepfs/share/zsz/simulation_agent/data_analyze/raw_data/'
    real_data, real_total_nodes = load_real_data(real_data_path)

    # 3. Extract Data
    real_active = real_data['active_counts']
    real_cum_viewers = real_data['cum_viewers']
    real_cum_counts = real_data['cum_view_counts']
    
    sim_active = sim_data['active_counts']
    sim_cum_viewers = sim_data['cum_viewers']
    sim_cum_counts = sim_data['cum_view_counts']

    # 4. Normalize
    real_norm_active = real_active / real_total_nodes
    sim_norm_active = sim_active / sim_total_nodes
    
    # 5. Plot
    plt.figure(figsize=(18, 12))
    
    # Row 1: Active Users
    plt.subplot(3, 2, 1)
    plt.plot(real_norm_active.index, real_norm_active.values, label='Real')
    plt.title(f'Real Active Users (Prop) N={real_total_nodes}')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(sim_norm_active, label='Sim', color='orange')
    plt.title(f'Sim Active Users (Prop) N={sim_total_nodes}')
    plt.grid(True)
    
    # Row 2: Accumulated Viewers
    plt.subplot(3, 2, 3)
    plt.plot(real_cum_viewers.index, real_cum_viewers.values, label='Real')
    plt.title('Real Accumulated Viewers')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(sim_cum_viewers, label='Sim', color='orange')
    plt.title('Sim Accumulated Viewers')
    plt.grid(True)
    
    # Row 3: Accumulated View Counts
    plt.subplot(3, 2, 5)
    plt.plot(real_cum_counts.index, real_cum_counts.values, label='Real')
    plt.title('Real Accumulated View Counts')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.plot(sim_cum_counts, label='Sim', color='orange')
    plt.title('Sim Accumulated View Counts')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_full.png')
    print("Comparison plot saved to comparison_full.png")
    
    # Stats
    print("\n--- Comparison Stats ---")
    print(f"Real Max Active Prop: {real_norm_active.max():.4f}")
    print(f"Sim Max Active Prop: {sim_norm_active.max():.4f}")
    print(f"Real Final Viewers: {real_cum_viewers.iloc[-1]:.4f}")
    print(f"Sim Final Viewers: {sim_cum_viewers[-1]:.4f}")
    print(f"Real Final View Counts: {real_cum_counts.iloc[-1]:.4f}")
    print(f"Sim Final View Counts: {sim_cum_counts[-1]:.4f}")
    
    # Correlation
    def calc_corr(sim_arr, real_arr):
        v_sim = np.array(sim_arr).flatten()
        v_real = np.array(real_arr).flatten()
        return np.corrcoef(v_real, v_sim)[0, 1]

    print(f"Correlation (Active): {calc_corr(sim_norm_active, real_norm_active):.4f}")
    print(f"Correlation (Viewers): {calc_corr(sim_cum_viewers, real_cum_viewers):.4f}")
    print(f"Correlation (View Counts): {calc_corr(sim_cum_counts, real_cum_counts):.4f}")

if __name__ == "__main__":
    main()
