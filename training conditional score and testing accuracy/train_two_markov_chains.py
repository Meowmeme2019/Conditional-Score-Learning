#!/usr/bin/env python
# coding: utf-8
"""
Train conditional score network for a perturbed Markov chain (P₁), using the same architecture
as the P₀ model (assumed already trained).

- Loads saved kernel of P₀
- Perturbs P₀ kernel to form new kernel P₁
- Simulates data from P₁
- Trains conditional score network on P₁ data
- Saves trained model as P₁_model_*.pth

Author: Melody
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os


# ================================================================
# These must be defined in Messy_data_Training_conditional_score.py:
# - sample_markov_chain
# - ConditionalScoreNet
# - hyvarinen_loss
# ================================================================
from Messy_data_Training_conditional_score import ConditionalScoreNet, sample_markov_chain, hyvarinen_loss

# def compute_hyvarinen_score(model, x, y):
#     """
#     Compute Hyvärinen score S_H(y,x;θ)
#     """
#     model.eval()
#     x = x.to(device)
#     y = y.clone().detach().to(device).requires_grad_(True)

#     with torch.set_grad_enabled(True):   # allow gradients for y
#         psi = model(x, y)
#         norm_term = 0.5 * (psi ** 2).sum(dim=1)

#         grads = []
#         for i in range(psi.shape[1]):
#             grad_i = torch.autograd.grad(
#                 psi[:, i].sum(), y, create_graph=False, retain_graph=True
#             )[0][:, i]
#             grads.append(grad_i)
#         divergence = torch.stack(grads, dim=1).sum(dim=1)

#     return norm_term + divergence


# ================================================================
# Configuration
# ================================================================
def train_score_model(X, save_prefix, A_str, b_str, Sigma_str):
    x_data = X[9999:-1]
    y_data = X[10000:]

    dataset = TensorDataset(
        torch.tensor(x_data, dtype=torch.float32).to(device),
        torch.tensor(y_data, dtype=torch.float32).to(device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConditionalScoreNet(d=d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc=f"Training {save_prefix}"):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = hyvarinen_loss(model, x_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[{save_prefix} Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

    model_path = f"{save_prefix}_128_3_messy_markov_model_A_{A_str}_b_{b_str}_Sigma_{Sigma_str}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"{save_prefix} model saved to: {model_path}")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    DATA_DIR = "/ihome/tbanerjee/wuc3/markov_train"
    d = 10
    n_post= 100000
    batch_size = 64
    hidden_dim = 128
    num_layers = 3
    lr = 1e-4
    num_epochs = 400
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)


    # ---------------------------------------------------------------------
    # Load P₀ Data and Kernel
    # ---------------------------------------------------------------------
    param_path_P0 = "P_0_messy_kernel_params_A_0.73_b_1.90_Sigma_0.05.pt"
    data_path_P0 = "P_0_messy_markov_path_A_0.73_b_1.90_Sigma_0.05.pt"

    P0_params = torch.load(param_path_P0)
    X0 = torch.load(data_path_P0).numpy()

    A0 = P0_params["A"]
    b0 = P0_params["b"]
    Sigma0 = P0_params["Sigma"]

    rho_A0 = torch.max(torch.abs(torch.linalg.eigvals(A0))).item()
    min_eig_Sigma0 = torch.linalg.eigvalsh(Sigma0).min().item()
    b_norm0 = np.linalg.norm(b0)

    A_str0 = f"{rho_A0:.2f}"
    b_str0 = f"{b_norm0:.2f}"
    Sigma_str0 = f"{min_eig_Sigma0:.2f}"

    # =============================
    # Perturb to create P_1 kernel
    # =============================

    # Perturb eigenvalues for A1
    g0 = torch.Generator().manual_seed(seed)
    g1 = torch.Generator().manual_seed(seed + 123)
    eigs = 1.55* torch.rand(d, generator=g0) + 0.05 * torch.rand(d, generator=g1) - 0.8  # perturb A1 a little bit
    D1 = torch.diag(eigs)

    # Random invertible matrix V
    V1 = torch.randn(d, d)
    while torch.linalg.matrix_rank(V1) < d:
        V1 = torch.randn(d, d)

    # Construct A1
    A1 = V1 @ D1 @ torch.linalg.inv(V1)

    # b_1 is just like b_0
    b1 = np.random.rand(d) 

    # Perturb R used in Sigma1
    R0 = torch.randn(d, d, generator=g0)
    R1 = torch.randn(d, d, generator=g1)
    R = 0.95* R0 + 0.05 * R1
    Sigma1 = R @ R.T + 5e-2 * torch.eye(d)  # PD by construction

    # =============================
    # Stability and summary
    # =============================
    rho_A1 = torch.max(torch.abs(torch.linalg.eigvals(A1))).item()
    min_eig_Sigma1 = torch.linalg.eigvalsh(Sigma1).min().item()
    b_norm1 = np.linalg.norm(b1)

    A_str1 = f"{rho_A1:.2f}"
    b_str1 = f"{b_norm1:.2f}"
    Sigma_str1 = f"{min_eig_Sigma1:.2f}"


    # ================================================================
    # Generate Markov chain from P₁
    # ================================================================
    X1 = sample_markov_chain(n_post, A1, b1, Sigma1, seed=seed, show_progress=True)

    # Save P₁ kernel and data
    param_path = f"P_1_messy_kernel_params_A_{A_str1}_b_{b_str1}_Sigma_{Sigma_str1}.pt"
    data_path = f"P_1_messy_markov_path_A_{A_str1}_b_{b_str1}_Sigma_{Sigma_str1}.pt"
    torch.save({"A": A1, "b": b1, "Sigma": Sigma1}, param_path)
    torch.save(torch.from_numpy(X1), data_path)
    print(f"P₁ parameters saved to: {param_path}")
    print(f"P₁ data saved to: {data_path}")


    # ---------------------------------------------------------------------
    # Train P₀ model
    # ---------------------------------------------------------------------
    train_score_model(X0, save_prefix="P_0", A_str=A_str0, b_str=b_str0, Sigma_str=Sigma_str0)

    # ---------------------------------------------------------------------
    # Train P₁ model
    # ---------------------------------------------------------------------
    train_score_model(X1, save_prefix="P_1", A_str=A_str1, b_str=b_str1, Sigma_str=Sigma_str1)
        