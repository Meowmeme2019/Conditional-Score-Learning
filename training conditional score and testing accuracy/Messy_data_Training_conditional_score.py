#!/usr/bin/env python
# coding: utf-8

from tqdm.notebook import tqdm
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from torch.utils.data import TensorDataset, DataLoader

from torchvision import transforms
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from scipy import integrate
from torchvision.utils import make_grid


def sample_markov_chain(n_steps, A, b, Sigma, x0=None, seed=None, show_progress=False):
    """
    Generate a sample path from a Gaussian Markov chain:
        X_{t+1} | X_t ~ N(A X_t + b, Sigma)

    Parameters
    ----------
    n_steps : int
        Number of transitions (path will have length n_steps+1).
    A : np.ndarray (d x d)
        Linear transformation matrix.
    b : np.ndarray (d,)
        Bias vector.
    Sigma : np.ndarray (d x d)
        Covariance matrix (positive definite).
    x0 : np.ndarray (d,), optional
        Initial state. Defaults to zero vector.
    seed : int, optional
        Random seed for reproducibility.
    show_progress : bool, optional
        If True, show tqdm progress bar.

    Returns
    -------
    X : np.ndarray of shape (n_steps+1, d)
        The simulated Markov chain sample path.
    """
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    Sigma = np.array(Sigma, dtype=np.float32)
    x = np.array(x0, dtype=np.float32) if x0 is not None else np.zeros_like(b, dtype=np.float32)


    # Initialize
    if x0 is None:
        x = np.zeros(d)
    else:
        x = np.array(x0)

    X = [x]
    iterator = range(n_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating Markov chain")

    for _ in iterator:
        noise = rng.multivariate_normal(mean=np.zeros(d, dtype=np.float32), cov=Sigma)
        x = A @ x + b + noise
        X.append(x)

    return np.array(X)

def compute_hyvarinen_score(model, x, y, device):
    """
    Compute Hyvärinen score S_H(y,x;θ)
    """
    model.eval()
    x = x.to(device)
    y = y.clone().detach().to(device).requires_grad_(True)

    with torch.set_grad_enabled(True):
        psi = model(x, y)
        loss1 = 0.5 * (psi ** 2).sum(dim=1)

        grads = []
        for i in range(psi.shape[1]):
            grad = torch.autograd.grad(psi[:, i].sum(), y, create_graph=True)[0][:, i]
            grads.append(grad)
        divergence = torch.stack(grads, dim=1).sum(dim=1)
        H_score = loss1 + divergence
    return H_score



#@title Define the Network

class ConditionalScoreNet(nn.Module):
    """
    Neural network to approximate conditional score:
        ψ(y, x; θ) ≈ ∇_y log p(y | x)

    Input:  concatenated vector (x, y) ∈ R^{2d}
    Output: vector in R^d
    """

    def __init__(self, d, hidden_dim=256, num_layers=5):
        super().__init__()
        layers = []

        # First layer: input = 2d, hidden_dim
        layers.append(nn.Linear(2*d, hidden_dim))
        layers.append(nn.SiLU())  # smooth activation

        # Middle layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # Final layer: hidden_dim → d (score vector dimension)
        layers.append(nn.Linear(hidden_dim, d))

        # Register as Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Forward pass.
        x: tensor of shape (batch, d)
        y: tensor of shape (batch, d)
        Returns: ψ(y, x; θ) ∈ R^{batch × d}
        """
        inp = torch.cat([x, y], dim=1)  # concatenate along features
        return self.net(inp)


#@title Define the loss function

def hyvarinen_loss(model, x, y):
    """
    Hyvarinen loss for conditional score learning.

    Args:
        model: ConditionalScoreNet
        x: tensor (batch, d)
        y: tensor (batch, d)

    Returns:
        scalar loss
    """
    x.requires_grad_(False)
    y.requires_grad_(True)
    psi = model(x, y)

    loss1 = 0.5 * (psi ** 2).sum(dim=1).mean()

    # Compute divergence ∇_y · ψ
    grads = []
    for i in range(psi.shape[1]):
        grad = torch.autograd.grad(psi[:, i].sum(), y, create_graph=True)[0][:, i]
        grads.append(grad)
    divergence = torch.stack(grads, dim=1).sum(dim=1)

    loss2 = divergence.mean()
    return loss1 + loss2


def evaluate_score_convergence(model_path, kernel_path, path_path, burn_in=0, max_plot=1000, hidden_dim =128, num_layers=4, device="cuda"):
    """
    Compare predicted and true conditional scores along the Markov path.

    Parameters:
        model_path: str
            Path to trained score network (.pth file)
        kernel_path: str
            Path to saved kernel parameters (.pt with A, b, Sigma)
        path_path: str
            Path to the Markov path .pt file (shape [T, d])
        burn_in: int
            Number of steps to skip before evaluation (e.g. 100)
        max_plot: int
            Number of time steps to plot
        device: "cuda" or "cpu"
    """

    # Load path
    X = torch.load(path_path).float().to(device)
    d = X.shape[1]
    
    # Load kernel parameters
    loaded_params = torch.load(kernel_path)
    A_P = loaded_params["A"]
    b_P = loaded_params["b"]
    Sigma_P = loaded_params["Sigma"]
    # Convert kernel parameters to tensors
    A = torch.tensor(A_P, dtype=torch.float32, device=device)
    b = torch.tensor(b_P, dtype=torch.float32, device=device)
    Sigma = torch.tensor(Sigma_P, dtype=torch.float32, device=device)
    Sigma_inv = torch.linalg.inv(Sigma)
    

    # Load model
    model = ConditionalScoreNet(d=d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    #  Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Select test dataset
    X_test_x = torch.tensor(X[burn_in:max_plot-1], dtype=torch.float32).to(device)   # X_{n-1}
    X_test_y = torch.tensor(X[burn_in+1: max_plot], dtype=torch.float32).to(device) # X_n
    
    N = X_test_x.shape[0]
    
    # Compute true score
    with torch.no_grad():
        mu = X_test_x @ A.T + b  # shape (N, d)
        true_score = - (X_test_y - mu) @ Sigma_inv.T  # ∇_y log p(y|x)

    
    # Compute predicted score
    model.eval()
    with torch.no_grad():
        pred_score = model(X_test_x, X_test_y)  # shape (N, d)


    # === Metrics ===
    # MSE = || pred - true ||^2
    mse = torch.mean((pred_score - true_score)**2).item()
    
    # VarScale = || true ||^2
    varscale = torch.mean(true_score**2).item()
    
    # Relative error
    relative_error = mse / varscale

    # === Report ===
    print(f"MSE        = {mse:.6e}")
    print(f"VarScale   = {varscale:.6e}")
    print(f"Rel. Error = {relative_error:.6e}")
 


# ---- Only run this if script is executed directly ----
if __name__ == "__main__":
    # Original script behavior here

    d = 10
    n_pre = 200000
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate matrix A: with all the eigenven value |\lambda_i|<1. A= VDV^{-1}

    # Step 1: get random eigen values
    eigs = 1.6*torch.rand(d) - 0.8   # uniform(-0.8,0.8), avoid getting eigen values ~ 1, 
    D = torch.diag(eigs)

    # Step 2: random invertible matrix V
    V = torch.randn(d, d)
    while torch.linalg.matrix_rank(V) < d:
        V = torch.randn(d, d)

    # Step 3: construct A with VDV^{-1}
    A_P = V @ D @ torch.linalg.inv(V)

    # Generate bias b_P
    b_P = np.random.rand(d)  # fills b_P with random values between 0 and 1

    # Generate covariance matrix Sigma: must be positive definite. Sigma = RR' + \delta I_d 
    R = torch.randn(d, d)
    delta = 5e-2  # small regularization term, make sure the Sigma is strictly p.d.
    Sigma_P = R @ R.T + delta * torch.eye(d)  

    # check stability
    rho_A = torch.max(torch.abs(torch.linalg.eigvals(A_P))).item()
    print("The largest abs eigenvalue of A:", rho_A, "Therefore, the markov chain is stable.")

    eigvals_Sigma = torch.linalg.eigvalsh(Sigma_P)
    min_eig_Sigma = eigvals_Sigma.min().item()
    print("Minimum eigenvalue of Sigma:", min_eig_Sigma, "Therefore, the covariance matrix is p.d.")  # should be > 0


    X = sample_markov_chain(
        n_steps=n_pre,
        A=A_P,
        b=b_P,
        Sigma=Sigma_P,
        seed=seed,
        show_progress=True  # <--- tqdm on
    )

    print("Generated full data path shape:", X.shape)

    # Format parameter values into the filename
    A_str = f"{rho_A:.2f}"               # the largest eigen value
    b_str = f"{np.linalg.norm(b_P):.2f}"  # save 2 digit
    Sigma_str = f"{min_eig_Sigma:.2f}"       # smallest eigen value of Sigma

    kernel_params = {
        "A": A_P,
        "b": b_P,
        "Sigma": Sigma_P
    }

    torch.save(kernel_params, f"200k_messy_kernel_params_A_{A_str}_b_{b_str}_Sigma_{Sigma_str}.pt")  
    print("Saved kernel parameters.")

    filename = f"200k_messy_markov_path_A_{A_str}_b_{b_str}_Sigma_{Sigma_str}.pt"
    torch.save(torch.from_numpy(X), filename)
    print("200k_messy data path Saved to:", filename)

    hidden_dim = 512
    num_layers = 5
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-5

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Build dataset of pairs, starting from the 50th step
    x_data = X[9999:-1]   # X_{n-1}, start at index 9999
    y_data = X[10000:]     # X_n, start at index 10000

    print("x_data shape:", x_data.shape)  # (T-1, d)
    print("y_data shape:", y_data.shape)  # (T-1, d)

    #@title Train dataset
    dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32).to(device),
                            torch.tensor(y_data, dtype=torch.float32).to(device))


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Parameters
    d = X.shape[1]   # dimension of your Markov process
    model = ConditionalScoreNet(d=d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)  # using GPU

    # Stable optimizer setup
    lr = 1e-4  
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    num_epochs = 200
    # Outer loop with tqdm progress bar
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = hyvarinen_loss(model, x_batch, y_batch)
            loss.backward()
            
            # Add gradient clipping to stabilize high-order gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()


        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f}")


    # Save the trained model parameters in src directory
    model_path = f"512_5_messy_markov_model_A_{A_str}_b_{b_str}_Sigma_{Sigma_str}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

