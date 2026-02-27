import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import argparse
import os
import sys
import paths

# Add current directory for genim_utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from genim_utils import load_graph_custom, load_data_custom, normalize_adj, get_models, sampling


def optimize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    # Load data
    print("Loading custom data...")
    adj_scipy = load_graph_custom(str(paths.get_relationship_file('relationship_1000.csv')))
    inverse_pairs = load_data_custom(str(paths.get_simulation_output_dir()))
    
    # Process Adjacency Matrix for Model
    adj = adj_scipy
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.Tensor(adj.toarray()).to_sparse().to(device)
    
    # Model Setup
    hidden_dim = 1024
    latent_dim = 512
    input_dim = inverse_pairs.shape[1]
    
    vae_model, forward_model = get_models(input_dim, hidden_dim, latent_dim, device)
    
    # Load checkpoints
    vae_checkpoint = paths.get_deepim_checkpoint('vae_model.pth')
    forward_checkpoint = paths.get_deepim_checkpoint('forward_model.pth')
    
    if os.path.exists(vae_checkpoint) and os.path.exists(forward_checkpoint):
        vae_model.load_state_dict(torch.load(vae_checkpoint, map_location=device))
        forward_model.load_state_dict(torch.load(forward_checkpoint, map_location=device))
        print("Models loaded.")
    else:
        print("Checkpoints not found! Please run train_genim.py first.")
        return

    # Freeze models
    for param in vae_model.parameters():
        param.requires_grad = False
    for param in forward_model.parameters():
        param.requires_grad = False
        
    encoder = vae_model.Encoder
    decoder = vae_model.Decoder
    
    # Sampling for initialization
    topk_seed_indices = sampling(inverse_pairs)
    
    z_hat = 0
    sim_avg = 0
    for i in topk_seed_indices:
        # inverse_pairs[i] is (1000, 3) -> seed is [:, 0], sim is [:, 1]
        seed_tensor = torch.FloatTensor(inverse_pairs[i, :, 0]).unsqueeze(0).to(device)
        sim_tensor = torch.FloatTensor(inverse_pairs[i, :, 1]).unsqueeze(0).to(device)
        z_hat += encoder(seed_tensor)
        sim_avg += sim_tensor
    z_hat = z_hat / len(topk_seed_indices)
    sim_avg = sim_avg / len(topk_seed_indices)
    
    y_true = torch.ones((1, input_dim)).to(device) # Shape (1, 1000)
    
    z_hat = z_hat.detach()
    z_hat.requires_grad = True
    z_optimizer = optim.Adam([z_hat], lr=1e-4)
    
    print("Starting optimization...")
    for i in range(args.iterations):
        x_hat = decoder(z_hat)
        # Combine x_hat and sim: (1, N) -> (N, 2)
        x_hat_exp = x_hat.squeeze(0).unsqueeze(-1) # (N, 1)
        sim_exp = sim_avg.squeeze(0).unsqueeze(-1) # (N, 1)
        combined = torch.cat([x_hat_exp, sim_exp], dim=-1) # (N, 2)
        y_hat = forward_model(combined, adj)            
        
        # Custom loss with count constraint
        forward_loss = F.mse_loss(y_hat, y_true)

        # Constraint: penalize only if current_k > args.seed_num
        current_k = torch.sum(x_hat)
        count_loss = F.relu(current_k - args.seed_num) ** 2
        
        # Weight for the count constraint
        lambda_k = 0.1
        loss = forward_loss + lambda_k * count_loss
        
        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()
        
        if (i+1) % 10 == 0:
            print('Iteration: {}'.format(i+1),
                  '\t Total Loss:{:.5f}'.format(loss.item()),
                  '\t Forward Loss:{:.5f}'.format(forward_loss.item()),
                  '\t Count:{:.2f}'.format(current_k.item())
                 )

    # Final result
    x_final = decoder(z_hat)
    top_k = x_final.topk(args.seed_num)
    seed = top_k.indices[0].cpu().detach().numpy()
    
    print("Selected Seed Nodes:", seed)
    
    # Evaluation
    print("Evaluating diffusion...")
    x_input = torch.zeros(input_dim).to(device)
    x_input[seed] = 1
    with torch.no_grad():
        # Combine x_input and sim_avg
        x_exp = x_input.unsqueeze(-1) # (N, 1)
        sim_exp = sim_avg.squeeze(0).unsqueeze(-1) # (N, 1)
        combined_eval = torch.cat([x_exp, sim_exp], dim=-1) # (N, 2)
        y_pred = forward_model(combined_eval, adj)
        model_influence = y_pred.sum().item()
    print('Model Predicted Influence: {:.4f}'.format(model_influence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenIM Optimization")
    parser.add_argument("--iterations", type=int, default=1000, help="Optimization iterations")
    parser.add_argument("--seed_num", type=int, default=5, help="Target seed size")
    args = parser.parse_args()
    
    optimize(args)
