"""
gcn_model.py
------------
Two-layer Graph Convolutional Network for spatial temperature refinement.

Implemented in pure PyTorch — no PyTorch Geometric required.

Mathematical formulation
------------------------
Each GCN layer computes:

    H^(l+1) = σ( Â · H^(l) · W^(l) + b^(l) )

where
    Â    = D^(-1/2) (A + I) D^(-1/2)   (symmetrically normalised adjacency)
    H^(l) = node feature matrix at layer l  [N × F]
    W^(l) = learnable weight matrix         [F_in × F_out]
    σ     = ReLU (hidden layer) / Identity (output layer)

Graph topology
--------------
    Nodes  : 600 Delhi grid zones (25 rows × 24 cols, 2 km spacing)
    Edges  : 4-neighbour (N, S, E, W) adjacency — fixed for all timesteps

Node features (5 per zone)
--------------------------
    [RF-predicted temperature, humidity, wind_speed, latitude, longitude]

Target
------
    Actual zone temperature (°C) including Urban Heat Island effect
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Graph construction utilities
# ─────────────────────────────────────────────────────────────

def build_adjacency_matrix(grid_rows: int, grid_cols: int) -> np.ndarray:
    """
    Build a symmetric 4-neighbour adjacency matrix for a rectangular grid.

    Parameters
    ----------
    grid_rows, grid_cols : int
        Dimensions of the grid.

    Returns
    -------
    adj : np.ndarray, shape (N, N), dtype float32
        Binary adjacency matrix where N = grid_rows × grid_cols.
    """
    n = grid_rows * grid_cols
    adj = np.zeros((n, n), dtype=np.float32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r * grid_cols + c

            if c + 1 < grid_cols:               # East neighbour
                east = idx + 1
                adj[idx, east] = 1.0
                adj[east, idx] = 1.0

            if r + 1 < grid_rows:               # South neighbour
                south = (r + 1) * grid_cols + c
                adj[idx, south] = 1.0
                adj[south, idx] = 1.0

    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric normalisation: Â = D^(-1/2) (A + I) D^(-1/2).

    Adds self-loops so every node attends to itself, then normalises
    by the square root of degree for numerical stability.

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)

    Returns
    -------
    adj_norm : np.ndarray, shape (N, N), dtype float32
    """
    a_hat = adj + np.eye(adj.shape[0], dtype=np.float32)   # A + I
    degree = a_hat.sum(axis=1)
    d_inv_sqrt = np.diag(np.power(degree, -0.5).astype(np.float32))
    return (d_inv_sqrt @ a_hat @ d_inv_sqrt).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# GCN Layer
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single GCN layer:  H_out = Â · H_in · W + b

    Parameters
    ----------
    in_features : int   — dimensionality of input node features
    out_features : int  — dimensionality of output node features
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (N, in_features)
        adj_norm : (N, N)  pre-computed normalised adjacency matrix

        Returns
        -------
        (N, out_features)
        """
        return torch.mm(adj_norm, torch.mm(x, self.W)) + self.b


# ─────────────────────────────────────────────────────────────
# 2-Layer GCN Model
# ─────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.

    Architecture
    ------------
    Input [N × 5]
        └─► GCNLayer(5 → 32) → ReLU → Dropout(0.3)
            └─► GCNLayer(32 → 1)
                └─► Output [N × 1]  (predicted zone temperature)

    Parameters
    ----------
    in_features  : int, default 5   — number of node features
    hidden       : int, default 32  — hidden layer width
    out_features : int, default 1   — output dimension
    dropout      : float, default 0.3
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden: int = 32,
        out_features: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.layer1  = GCNLayer(in_features, hidden)
        self.layer2  = GCNLayer(hidden, out_features)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (N, in_features)
        adj_norm : (N, N)

        Returns
        -------
        (N, 1)  predicted temperature for each node
        """
        x = F.relu(self.layer1(x, adj_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, adj_norm)
        return x
