import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as sp
import argparse
import os
import sys
import paths

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genim_utils import load_graph_custom, load_data_custom, normalize_adj, get_models, loss_all, GraphDataset, collate_graph_batch, compute_pe


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    # Load data
    print("Loading custom data...")
    adj = load_graph_custom(str(paths.get_relationship_file('relationship_1000.csv')))
    inverse_pairs = load_data_custom(str(paths.get_simulation_output_dir()))
    print(f"Total samples loaded: {len(inverse_pairs)}")

    # Process Adjacency Matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    # Pre-compute Positional Encodings
    print("Pre-computing Positional Encodings...")
    pe = compute_pe(adj_norm, k=8, n=adj_norm.shape[0])
    
    # Prepare Data Loaders
    batch_size = 16
    
    # Create Dataset with pre-computed PE and Adj
    data_list = []
    for i in range(len(inverse_pairs)):
        sample = inverse_pairs[i]
        x = sample[:, 0]
        sim = sample[:, 1]
        y = sample[:, 2]
        # In a real scenario with varying graphs, you would load specific adj and compute specific pe here
        data_list.append((x, sim, y, adj_norm, pe))
        
    train_set = GraphDataset(data_list)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_graph_batch)

    
    # Model Setup
    hidden_dim = 1024
    latent_dim = 512
    input_dim = inverse_pairs.shape[1] # 1000 nodes
    
    vae_model, forward_model = get_models(input_dim, hidden_dim, latent_dim, device)
    
    optimizer = optim.Adam([{'params': vae_model.parameters()}, {'params': forward_model.parameters()}], 
                           lr=1e-4)
    
    forward_model.train()
    
    print("Starting training...")
    for epoch in range(args.epochs):
        total_overall = 0
        total_re = 0
        total_forw = 0
        precision_re = 0
        recall_re = 0
        
        for batch_idx, (x, sim, y, adj_batch, pe_batch) in enumerate(train_loader):
            x = x.float().to(device) # (B, N)
            sim = sim.float().to(device) # (B, N)
            y = y.float().to(device) # (B, N)
            adj_batch = adj_batch.to(device) # (B*N, B*N)
            pe_batch = pe_batch.to(device) # (B*N, k)
            
            optimizer.zero_grad()
            
            bs = x.size(0)
            
            # VAE Forward
            x_hat = vae_model(x)
            
            # Combine x_hat and sim for GAT input: (batch, N, 2) -> (batch*N, 2)
            x_hat_expanded = x_hat.unsqueeze(-1) # (B, N, 1)
            sim_expanded = sim.unsqueeze(-1) # (B, N, 1)
            combined_input = torch.cat([x_hat_expanded, sim_expanded], dim=-1) # (B, N, 2)
            combined_gat = combined_input.view(-1, 2) # (B*N, 2)
            
            # GAT Forward
            y_hat = forward_model(combined_gat, adj_batch, batch_pe=pe_batch)
            
            # Reshape y_hat: (batch*N, 1) -> (batch, N)
            y_hat = y_hat.view(bs, -1)
            
            # Loss
            # loss, re, forw = loss_all(x, x_hat, y.sum(axis=1).unsqueeze(1), y_hat)
            loss, re, forw = loss_all(x, x_hat, y, y_hat)
            
            # Metrics
            x_true_np = x.cpu().detach().numpy()
            x_pred = x_hat.cpu().detach().numpy()
            x_pred[x_pred > 0.5] = 1
            x_pred[x_pred != 1] = 0
            
            # Vectorized Precision/Recall
            tp = (x_true_np * x_pred).sum(axis=1)
            pp = x_pred.sum(axis=1)
            ap = x_true_np.sum(axis=1)
            
            prec = np.divide(tp, pp, out=np.zeros_like(tp), where=pp!=0)
            rec = np.divide(tp, ap, out=np.zeros_like(tp), where=ap!=0)
            
            precision_re += prec.sum()
            recall_re += rec.sum()
            
            total_overall += loss.item()
            total_re += re.item()
            total_forw += forw.item()
            loss = loss / bs
            
            loss.backward()
            optimizer.step()
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

        print("Epoch: {}".format(epoch+1), 
              "\tTotal: {:.4f}".format(total_overall / len(train_set)),
              "\tReconstruction Loss: {:.4f}".format(total_re / len(train_set)),
              "\tForward Loss: {:.4f}".format(total_forw / len(train_set)),
              "\tPrecision: {:.4f}".format(precision_re / len(train_set)),
              "\tRecall: {:.4f}".format(recall_re / len(train_set)),
             )

    # Save models
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(vae_model.state_dict(), 'checkpoints/vae_model.pth')
    torch.save(forward_model.state_dict(), 'checkpoints/forward_model.pth')
    print("Models saved to checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenIM Training")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    args = parser.parse_args()
    
    train(args)
