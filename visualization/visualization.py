import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio.v2 as imageio
import pickle

def draw_network_propagation(sim, target_movie, round_num, seed_agents, watched_agents, output_dir, pos=None, newly_watched_agents=None, propagation_edges=None):
    """
    Draws the social network and highlights agents who have watched the target movie.
    Returns the position layout and propagation edges to be reused.
    """
    if propagation_edges is None:
        propagation_edges = set()
    
    G = nx.Graph()
    
    # Add nodes
    all_agents = sim.agents.values()
    for agent in all_agents:
        G.add_node(agent.id)
    
    # Add edges based on contacts
    for agent in all_agents:
        contacts = sim.data.get_all_contacts_id(agent.id)
        for contact_id in contacts:
            if G.has_node(contact_id):
                G.add_edge(agent.id, contact_id)
    
    # Calculate layout if not provided
    layout_file = os.path.join(os.path.join(output_dir, "../"), "layout_1000.pkl")
    
    if pos is None:
        if os.path.exists(layout_file):
            print(f"Loading layout from {layout_file}...")
            with open(layout_file, "rb") as f:
                pos = pickle.load(f)
        else:
            pos = nx.spring_layout(G, seed=42, k=0.7, iterations=50)
            # Save layout
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving layout to {layout_file}...")
            with open(layout_file, "wb") as f:
                pickle.dump(pos, f)

            
    # Draw
    plt.figure(figsize=(16, 16), dpi=500)
    
    # Draw edges (background)
    nx.draw_networkx_edges(G, pos,width=0.5, alpha=0.2, edge_color='gray')
    
    # Highlight propagation edges (edges connecting newly watched to previously watched)
    if newly_watched_agents:
        # Previously watched includes seeds and anyone who watched before this round
        previously_watched = (set(watched_agents) | set(seed_agents)) - set(newly_watched_agents)
        
        for new_agent in newly_watched_agents:
            if new_agent in G:
                for neighbor in G.neighbors(new_agent):
                    if neighbor in previously_watched:
                        propagation_edges.add((new_agent, neighbor))
        
    if propagation_edges:
        # Draw these edges with a highlighting color (e.g., Red/Orange) and thicker line
        nx.draw_networkx_edges(G, pos, edgelist=propagation_edges, edge_color='#FF4136', width=1, alpha=0.6)

    # Separate nodes into groups
    seed_nodes = [n for n in G.nodes() if n in seed_agents]
    watched_nodes = [n for n in G.nodes() if n in watched_agents and n not in seed_agents]
    other_nodes = [n for n in G.nodes() if n not in seed_agents and n not in watched_agents]
    
    # Draw nodes in order: others -> watched -> seeds (last one is on top)
    
    # 1. Other nodes (Blue)
    if other_nodes:
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=other_nodes,
                               node_color='#0074D9', 
                               node_size=200, 
                               edgecolors='black',
                               linewidths=1)
                               
    # 2. Watched nodes (Orange)
    if watched_nodes:
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=watched_nodes,
                               node_color='#FF851B', 
                               node_size=200, 
                               edgecolors='black',
                               linewidths=1)
                               
    # 3. Seed nodes (Red) - Drawn last to be on top
    if seed_nodes:
        nx.draw_networkx_nodes(G, pos, 
                               nodelist=seed_nodes,
                               node_color='#FF4136', 
                               node_size=400, 
                               edgecolors='black',
                               linewidths=2)
    
    # Add labels only for seeds to avoid clutter, or no labels
    # nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Propagation of '{target_movie}'\nRound: {round_num} | Watchers: {len(watched_agents)} | Seeds: {len(seed_agents)}", 
              fontsize=16, fontweight='bold')
    plt.axis('off') # Turn off axis
    
    # Add legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='#FF4136', label='Seed Agents'),
        mpatches.Patch(color='#FF851B', label='Watched Agents'),
        mpatches.Patch(color='#0074D9', label='Unwatched Agents'),
        mpatches.Patch(color='#FF4136', label='Propagation Edge', alpha=0.6) # Add legend for edge
    ]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=12)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"network_round_{round_num:03d}.png"), dpi=500, bbox_inches='tight')
    plt.close()
    
    return pos, propagation_edges

def create_gif(output_dir, output_filename="propagation.gif", fps=1):
    """
    Creates a GIF from the generated images in the output directory.
    """
    images = []
    # Collect files sorted by round number
    files = sorted([f for f in os.listdir(output_dir) if f.startswith("network_round_") and f.endswith(".png")])
    
    if not files:
        print("No images found to create GIF.")
        return

    for filename in files:
        images.append(imageio.imread(os.path.join(output_dir, filename)))
        
    output_path = os.path.join(output_dir, output_filename)
    # Use fps instead of duration as duration might not be effective in some versions
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to {output_path}")


