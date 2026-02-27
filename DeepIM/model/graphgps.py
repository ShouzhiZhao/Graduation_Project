import torch
import torch.nn as nn
import torch.nn.functional as F

class GPSLayer(nn.Module):
    def __init__(self, dim_h, num_heads, dropout=0.1):
        super().__init__()
        self.dim_h = dim_h

        # Local MPNN
        self.local_model = GCNLayer(dim_h, dim_h)

        # Global Attention
        self.self_attn = nn.MultiheadAttention(
            dim_h, num_heads, dropout=dropout, batch_first=True
        )

        # Norms (GraphGPS: BatchNorm)
        self.norm_local = nn.BatchNorm1d(dim_h)
        self.norm_attn = nn.BatchNorm1d(dim_h)
        self.norm_ffn = nn.BatchNorm1d(dim_h)

        self.dropout = nn.Dropout(dropout)

        # FFN (4x hidden dim, Transformer standard)
        self.ffn = nn.Sequential(
            nn.Linear(dim_h, dim_h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_h * 4, dim_h),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj_norm, batch_size, num_nodes, attn_bias=None):
        """
        x: (B*N, D)
        adj_norm: (B*N, B*N) sparse, block-diagonal
        attn_bias: (B, N, N) or None
        """

        # ===== Local MPNN =====
        h_local = self.local_model(x, adj_norm)
        x = x + self.dropout(h_local)
        x = self.norm_local(x)

        # ===== Global Attention =====
        h = x.view(batch_size, num_nodes, self.dim_h)

        if attn_bias is not None:
            h_attn, _ = self.self_attn(
                h, h, h,
                attn_mask=attn_bias
            )
        else:
            h_attn, _ = self.self_attn(h, h, h)

        h_attn = h_attn.reshape(batch_size * num_nodes, self.dim_h)
        x = x + self.dropout(h_attn)
        x = self.norm_attn(x)

        # ===== FFN =====
        h_ffn = self.ffn(x)
        x = x + h_ffn
        x = self.norm_ffn(x)

        return x


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()

    def forward(self, x, adj_norm):
        # adj_norm: D^{-1/2} (A + I) D^{-1/2}
        out = self.linear(x)
        out = torch.sparse.mm(adj_norm, out)
        out = self.act(out)
        return out


class GraphGPS(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        num_nodes,
        dropout=0.1,
        pe_dim=8
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim

        # Input embedding
        self.input_emb = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding (shared graph)
        if pe_dim > 0:
            self.pe_lin = nn.Linear(pe_dim, hidden_dim)
        
        # GPS Layers
        self.layers = nn.ModuleList([
            GPSLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, adj_norm, batch_pe=None, attn_bias=None):
        """
        x: (B*N, input_dim)
        adj_norm: block-diagonal sparse adj
        batch_pe: (B*N, pe_dim) or None
        attn_bias: (B, N, N) or None
        """
        total_nodes = x.size(0)
        batch_size = total_nodes // self.num_nodes

        h = self.input_emb(x)

        if batch_pe is not None and hasattr(self, 'pe_lin'):
            h = h + self.pe_lin(batch_pe)

        for layer in self.layers:
            h = layer(h, adj_norm, batch_size, self.num_nodes, attn_bias)

        h = self.out_head(h)
        return h
