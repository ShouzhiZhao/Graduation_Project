import pandas as pd
import os
import sys

# Add parent directory to path to import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import paths

import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import imageio.v2 as imageio
import shutil
import tempfile

import json
from concurrent.futures import ProcessPoolExecutor
import tqdm

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS Chinese font
plt.rcParams['axes.unicode_minus'] = False

def read_single_excel(f):
    usecols = ['id', 'content', 'message_type', 'reposts_count', 'comments_count', 'likes_count', 
               'retweeted_status', 'author_screen_name', 'publish_time']
    return pd.read_excel(f, usecols=usecols)

def load_data(base_dir, topic):
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if file name starts with the topic and ends with .xlsx
            # We use startswith to catch variations but be careful
            if topic in file and file.endswith(".xlsx"):
                # Ignore temp files or special description files
                if file.startswith('.') or file.startswith('字段说明'):
                    continue
                all_files.append(os.path.join(root, file))
    print(f"Found {len(all_files)} files for topic '{topic}'.")
    
    dfs = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(read_single_excel, all_files)
        pbar = tqdm.tqdm(total=len(all_files), desc=f"Loading topic '{topic}'")
        for i, df in enumerate(results):
            dfs.append(df)
            pbar.update(1)
        pbar.close()
 
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(full_df)}")
    return full_df

def clean_data(df):
    # Convert author_screen_name to string to avoid type errors
    df['author_screen_name'] = df['author_screen_name'].astype(str)
    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    # Drop rows with missing publish_time
    df = df.dropna(subset=['publish_time'])
    # Map message_type to readable labels
    type_map = {1: '原发文章', 2: '评论', 3: '转发文章'}
    df['message_type_label'] = df['message_type'].map(type_map)
    
    return df

def descriptive_stats(df, output_dir, topic):
    print(f"\n--- Descriptive Statistics for {topic} ---")
    
    # 1. Message Type Distribution
    type_counts = df['message_type_label'].value_counts()
    print("Message Type Counts:")
    print(type_counts)
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=type_counts.index, y=type_counts.values)
    plt.title(f'{topic} 信息类型分布')
    plt.ylabel('数量')
    plt.savefig(os.path.join(output_dir, 'message_type_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Daily Activity
    df['date'] = df['publish_time'].dt.date
    daily_counts = df['date'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6), dpi=300)
    daily_counts.plot(kind='line', marker='o')
    plt.title(f'{topic} 每日信息量趋势')
    plt.ylabel('数量')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'daily_trend.png'), dpi=300)
    plt.close()
    
    # 2.5 Daily Active Users
    daily_active_users = df.groupby('date')['author_screen_name'].nunique().sort_index()
    
    plt.figure(figsize=(12, 6), dpi=300)
    daily_active_users.plot(kind='line', marker='o', color='orange')
    plt.title(f'{topic} 每日活跃用户数趋势')
    plt.ylabel('活跃用户数')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'daily_active_users.png'), dpi=300)
    plt.close()
    
    # 3. Top Authors (by volume)
    top_authors = df['author_screen_name'].value_counts().head(10)
    print("\nTop 10 Active Authors:")
    print(top_authors)
    
    # 4. Engagement Stats
    print("\nEngagement Stats (Mean):")
    print(df[['reposts_count', 'comments_count', 'likes_count']].mean())

def propagation_analysis(df, output_dir, topic):
    print(f"\n--- Propagation Analysis for {topic} ---")
    
    # Filter for reposts (message_type == 3)
    reposts = df[df['message_type'] == 3].copy()
    
    if reposts.empty:
        print("No reposts found for propagation analysis.")
        return None
    
    # Build the full graph first to determine layout and top nodes
    print("Building full graph for layout...")
    G_full = nx.DiGraph()
    
    # Store edges with time: (source, target, time)
    timed_edges = []
    
    for index, row in reposts.iterrows():
        source_status = row['retweeted_status']
        target_user = row['author_screen_name']
        publish_time = row['publish_time']
        if pd.isna(source_status):
            continue
        source_user = None
        if isinstance(source_status, str):
            try:
                status_dict = json.loads(source_status)
                source_user = status_dict.get('author_name')
            except json.JSONDecodeError:
                match = re.search(r'"author_name":\s*"([^"]+)"', source_status)
                if match:
                    source_user = match.group(1)
        if source_user and target_user:
            timed_edges.append((source_user, target_user, publish_time))
            if G_full.has_edge(source_user, target_user):
                G_full[source_user][target_user]['weight'] += 1
            else:
                G_full.add_edge(source_user, target_user, weight=1)
                    
            
    if not timed_edges:
        print("Could not extract edges.")
        return None
        
    print(f"Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges.")
    
    # Select Top Nodes for Visualization (e.g., Top 1000 by degree)
    node_degrees = dict(G_full.degree(weight='weight'))
    # If graph is small, take fewer nodes
    num_top_nodes = min(1000, G_full.number_of_nodes())
    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:num_top_nodes]
    G_sub = G_full.subgraph(top_nodes).copy()
    
    # Compute fixed layout. Use a fixed seed and k value to stabilize the layout
    pos = nx.spring_layout(G_sub, k=0.01, seed=42, iterations=100)
    
    # Pre-calculate node sizes for top 20 to ensure consistency across frames
    num_top_draw = min(20, len(top_nodes))
    top_20 = top_nodes[:num_top_draw]
    other_nodes = top_nodes[num_top_draw:]
    
    # Prepare for GIF generation
    frames_dir = tempfile.mkdtemp()
    min_time = df['publish_time'].min()
    max_time = df['publish_time'].max()
    time_steps = pd.date_range(start=min_time, end=max_time, periods=30)
    images = []
    
    # Set global figure limits to prevent axis rescaling jitter
    x_values = [coord[0] for coord in pos.values()]
    y_values = [coord[1] for coord in pos.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.1
    xlim = (x_min - padding, x_max + padding)
    ylim = (y_min - padding, y_max + padding)

    for i, current_time in enumerate(time_steps):
        plt.figure(figsize=(16, 16), dpi=300)
        
        # Build graph at this time step containing only top nodes
        current_edges = [(u, v) for u, v, t in timed_edges if t <= current_time and u in top_nodes and v in top_nodes]
        
        # Create temporary graph for this frame
        G_frame = nx.DiGraph()
        G_frame.add_nodes_from(top_nodes)
        G_frame.add_edges_from(current_edges)
        
        nx.draw_networkx_nodes(G_frame, pos, nodelist=other_nodes, node_size=50, node_color='blue', alpha=0.8)
        nx.draw_networkx_nodes(G_frame, pos, nodelist=top_20, node_size=500, node_color='orange', alpha=0.9)
        nx.draw_networkx_edges(G_frame, pos, arrowstyle='->', arrowsize=3, width=1, edge_color='gray', alpha=0.15)
        
        labels = {node: node for node in top_20}
        nx.draw_networkx_labels(G_frame, pos, labels, font_size=8, font_family='Arial Unicode MS')
        
        plt.title(f"{topic} 传播网络演化 - {current_time.strftime('%Y-%m-%d %H:%M')}", fontsize=18)
        
        # Enforce fixed axis limits
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.axis('off')

        frame_path = f"{frames_dir}/frame_{i:03d}.png"
        plt.savefig(frame_path)
        plt.close()
        
        images.append(imageio.imread(frame_path))
        
    # Save GIF
    output_gif = os.path.join(output_dir, 'propagation_evolution.gif')
    imageio.mimsave(output_gif, images, fps=1)
    print(f"GIF saved to {output_gif}")

    # Cleanup
    shutil.rmtree(frames_dir)
    
    return G_full

def export_to_readme(df, G_full, output_dir, topic):
    print(f"\n--- Exporting to README.md for {topic} ---")
    
    # 1. Basic Stats
    total_count = len(df)
    type_counts = df['message_type_label'].value_counts()
    
    # 2. Top Authors
    top_authors = df['author_screen_name'].value_counts().head(10)
    
    # 3. Network Stats
    num_nodes = G_full.number_of_nodes()
    num_edges = G_full.number_of_edges()
    
    out_degree = dict(G_full.out_degree(weight='weight'))
    sorted_influencers = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    
    in_degree = dict(G_full.in_degree(weight='weight'))
    sorted_reposters = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]

    readme_content = f"""# 《{topic}》数据分析报告

## 1. 数据概况
- **总数据量**: {total_count:,} 条
- **时间跨度**: {df['publish_time'].min()} 至 {df['publish_time'].max()}

## 2. 描述性统计

### 信息类型分布
| 类型 | 数量 | 占比 |
|------|------|------|
"""
    for label, count in type_counts.items():
        percentage = (count / total_count) * 100
        readme_content += f"| {label} | {count:,} | {percentage:.2f}% |\n"

    readme_content += """
### 活跃发布用户 (Top 10)
| 用户名 | 发布数量 |
|--------|----------|
"""
    for user, count in top_authors.items():
        readme_content += f"| {user} | {count:,} |\n"

    readme_content += f"""
## 3. 传播网络分析

### 网络结构
- **节点数 (参与用户)**: {num_nodes:,}
- **边数 (转发关系)**: {num_edges:,}

### 核心影响力账号 (被转发最多)
| 排名 | 用户名 | 被转发次数 |
|------|--------|------------|
"""
    for i, (user, count) in enumerate(sorted_influencers, 1):
        readme_content += f"| {i} | {user} | {count:,} |\n"

    readme_content += """
### 活跃传播者 (转发他人最多)
| 排名 | 用户名 | 转发次数 |
|------|--------|----------|
"""
    for i, (user, count) in enumerate(sorted_reposters, 1):
        readme_content += f"| {i} | {user} | {count:,} |\n"

    readme_content += """
## 4. 可视化结果
### 信息类型分布
![Message Type Distribution](message_type_distribution.png)

### 每日热度趋势
![Daily Trend](daily_trend.png)

### 每日活跃用户数趋势
![Daily Active Users](daily_active_users.png)

### 传播网络动态演化
![Propagation Evolution](propagation_evolution.gif)
"""

    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md generated successfully for {topic}.")

def get_topics(base_dir):
    # Heuristic to find topics from filenames
    # Filenames are like "Topic_DateRange_Range.xlsx"
    # We can split by "_" and take the first part
    topics = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".xlsx"):
                # Ignore temp files or special description files
                if file.startswith('.') or file.startswith('字段说明'):
                    continue
                parts = file.split('_')
                if parts:
                    topics.add(parts[0])
    return sorted(list(topics))

def main():
    base_dir = paths.DATA_ANALYZE_DIR / 'raw_data'
    results_root = paths.DATA_ANALYZE_DIR / 'results'
    
    topics = get_topics(str(base_dir))
    print(f"Detected topics: {topics}")
    
    for topic in topics:
        # Create output directory for this topic
        output_dir = os.path.join(results_root, topic)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing topic: {topic}")
        df = load_data(base_dir, topic)
        
        if df is not None:
            df = clean_data(df)
            descriptive_stats(df, output_dir, topic)
            G_full = propagation_analysis(df, output_dir, topic)
            if G_full:
                 export_to_readme(df, G_full, output_dir, topic)
        
if __name__ == "__main__":
    main()
