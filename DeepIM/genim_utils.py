import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset

# Add DeepIM directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.graphgps import GraphGPS
from model.vae import VAEModel, Encoder, Decoder


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def load_graph_custom(csv_path):
    df = pd.read_csv(csv_path)
    max_id = 999
    row = df['user_1'].values  
    col = df['user_2'].values
    data = df['closeness'].values
    adj = sp.coo_matrix((data, (row, col)), shape=(max_id+1, max_id+1))
    return adj


def load_data_custom_from_single_movie(root_dir):
    samples = []
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Loading data from {len(subdirs)} subdirectories...")
    
    for subdir in subdirs:
        npz_files = [f for f in os.listdir(subdir) if f.endswith('.npz')]
        if not npz_files: continue
        final_states = []
        seed, sim = None, None
        for f in npz_files:
            data = np.load(os.path.join(subdir, f))
            seed = data['X'][0]
            sim = data['sim'][0]
            activated = np.any(data['X'], axis=0).astype(float)
            final_states.append(activated)
                
        if seed is None or not final_states: continue
        avg_final = np.mean(final_states, axis=0)
        samples.append(np.stack([seed, sim, avg_final], axis=1))
        
    return np.array(samples)


def load_data_custom(root_dir):
    root_dir = Path(root_dir)
    all_pairs = []
    for movie_dir in root_dir.iterdir():
        if movie_dir.is_dir():
            temp_path = movie_dir / 'temp'
            if temp_path.exists():
                print(f"Loading data from {movie_dir.name}...")
                pairs = load_data_custom_from_single_movie(str(temp_path))
                if len(pairs) > 0:
                    all_pairs.append(pairs)
    
    if len(all_pairs) > 0:
        result = np.concatenate(all_pairs, axis=0)
        print(f"Total samples loaded: {len(result)}")
        return result
    else:
        return np.array([])


def sampling(inverse_pairs):
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1*inverse_pairs.shape[0])).indices
    return top_k


def loss_all(x, x_hat, y, y_hat):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    forward_loss = F.mse_loss(y_hat, y, reduction='sum')
    return reproduction_loss+forward_loss, reproduction_loss, forward_loss


def loss_inverse(y_true, y_hat, x_hat):
    forward_loss = F.mse_loss(y_hat, y_true)
    L0_loss = torch.sum(torch.abs(x_hat))/x_hat.shape[1]
    return forward_loss+L0_loss, L0_loss


def get_models(input_dim, hidden_dim, latent_dim, device):
    encoder = Encoder(input_dim=input_dim, 
                      hidden_dim=hidden_dim, 
                      latent_dim=latent_dim)

    decoder = Decoder(input_dim=latent_dim, 
                      latent_dim=latent_dim, 
                      hidden_dim=hidden_dim, 
                      output_dim=input_dim)

    vae_model = VAEModel(Encoder=encoder, Decoder=decoder).to(device)

    # GraphGPS
    # input_dim for GraphGPS: (x + sim)
    # num_nodes is input_dim from arguments
    forward_model = GraphGPS(input_dim=2, 
                             hidden_dim=64, 
                             output_dim=1, 
                             num_layers=2, 
                             num_heads=4,
                             num_nodes=input_dim,
                             dropout=0.1,
                             pe_dim=8).to(device)
    
    return vae_model, forward_model


def get_batch_adj(adj, batch_size):
    """
    Constructs a block diagonal sparse adjacency matrix for a batch of graphs.
    adj: torch.sparse_coo_tensor (N, N)
    batch_size: int
    Returns: torch.sparse_coo_tensor (batch_size*N, batch_size*N)
    """
    if batch_size == 1:
        return adj
        
    indices = adj.indices()
    values = adj.values()
    N = adj.shape[0]
    
    # Repeat indices and values
    shifts = torch.arange(batch_size, device=adj.device) * N
    
    # indices: (2, E)
    # batch_indices: (batch_size, 2, E)
    batch_indices = indices.unsqueeze(0) + shifts.view(-1, 1, 1)
    
    # Flatten to (2, batch_size*E)
    batch_indices = batch_indices.transpose(0, 1).reshape(2, -1)
    
    batch_values = values.repeat(batch_size)
    
    new_shape = (batch_size * N, batch_size * N)
    
    return torch.sparse_coo_tensor(batch_indices, batch_values, new_shape).coalesce()


def compute_pe(adj, k, n):
    # Compute Laplacian eigenvectors
    # adj is scipy sparse matrix
    # Laplacian L = I - D^-1/2 A D^-1/2
    
    # Ensure no self-loops for standard Laplacian definition
    # Use dense solver for stability on small/medium graphs (< 2000 nodes)
    if adj.shape[0] < 2000:
        if sp.issparse(adj):
            adj = adj.toarray()
        np.fill_diagonal(adj, 0)
        
        rowsum = np.array(adj.sum(1))
        # Handle isolated nodes
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        # L = I - D^-1/2 * A * D^-1/2
        laplacian = np.eye(adj.shape[0]) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        try:
            # Dense eigendecomposition
            evals, evecs = np.linalg.eigh(laplacian)
            
            # Sort by eigenvalues
            idx = evals.argsort()
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            # Take k smallest non-trivial eigenvectors
            # First eigenvector has eigenvalue 0 (constant vector for connected graph)
            pe = evecs[:, 1:k+1]
            
            if pe.shape[1] < k:
                pad = np.zeros((pe.shape[0], k - pe.shape[1]))
                pe = np.hstack([pe, pad])
                
        except Exception as e:
            print(f"PE computation failed: {e}")
            pe = np.zeros((n, k))
            
    else:
        # Sparse solver for large graphs
        if not sp.issparse(adj):
            adj = sp.coo_matrix(adj)
            
        adj.setdiag(0)
        adj.eliminate_zeros()
        
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # L = I - D^-1/2 * A * D^-1/2
        laplacian = sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        try:
            # 'SM' = Smallest Magnitude
            evals, evecs = sp.linalg.eigsh(laplacian, k=k+1, which='SM')
            
            idx = evals.argsort()
            evals = evals[idx]
            evecs = evecs[:, idx]
            
            pe = evecs[:, 1:k+1]
            
            if pe.shape[1] < k:
                pad = np.zeros((pe.shape[0], k - pe.shape[1]))
                pe = np.hstack([pe, pad])
                
        except Exception as e:
            print(f"PE computation failed: {e}")
            pe = np.zeros((n, k))
        
    return torch.FloatTensor(pe)


class GraphDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of tuples containing:
            - x: (N,) seed
            - y: (N,) influence
            - adj: (N,N) scipy sparse adj (normalized)
            - pe: (N, k) torch tensor
        """
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_graph_batch(batch):
    # batch is a list of tuples: (x, sim, y, adj, pe)
    
    batch_x = []
    batch_sim = []
    batch_y = []
    batch_adjs = []
    batch_pes = []
    
    for (x, sim, y, adj, pe) in batch:
        batch_x.append(torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x)
        batch_sim.append(torch.tensor(sim, dtype=torch.float32) if not torch.is_tensor(sim) else sim)
        batch_y.append(torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y)
        batch_adjs.append(adj)
        batch_pes.append(pe)
        
    # Stack x, sim, y
    x_batch = torch.stack(batch_x) # (B, N)
    sim_batch = torch.stack(batch_sim) # (B, N)
    y_batch = torch.stack(batch_y) # (B, N)
    
    # Stack PE
    pe_batch = torch.cat(batch_pes, dim=0) # (B*N, k)
    
    # Block diagonal Adj
    # adj is expected to be scipy sparse
    big_adj = sp.block_diag(batch_adjs)
    
    # Convert to torch sparse
    indices = torch.LongTensor(np.vstack((big_adj.row, big_adj.col)))
    values = torch.FloatTensor(big_adj.data)
    shape = big_adj.shape
    
    adj_batch = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    
    return x_batch, sim_batch, y_batch, adj_batch, pe_batch
