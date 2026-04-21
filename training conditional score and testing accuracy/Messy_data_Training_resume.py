#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. Load stored kernel parameters and data path
# ============================================================
param_file = "messy_kernel_params_A_0.73_b_1.90_Sigma_0.05.pt"
path_file = "messy_markov_path_A_0.73_b_1.90_Sigma_0.05.pt"
model_file = "messy_markov_model_A_0.73_b_1.90_Sigma_0.05.pth"
	
params = torch.load(param_file, map_location=device)
A_P, b_P, Sigma_P = params["A"], params["b"], params["Sigma"]
X = torch.load(path_file).float()

d = X.shape[1]
print(f"Loaded Markov path of shape {X.shape}")

# ============================================================
# 2. Prepare dataset (same as before)
# ============================================================
x_data = X[9999:-1]
y_data = X[10000:]

dataset = TensorDataset(x_data.to(device), y_data.to(device))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print("Dataset ready:", len(dataset), "pairs")

# ============================================================
# 3. Define model and load pretrained weights
# ============================================================
class ConditionalScoreNet(nn.Module):
    def __init__(self, d, hidden_dim=512, num_layers=6):
        super().__init__()
        layers = [nn.Linear(2*d, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, d))
        self.net = nn.Sequential(*layers)
    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

model = ConditionalScoreNet(d=d, hidden_dim=512, num_layers=6).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
print(f"Loaded pretrained model weights from {model_file}")

# ============================================================
# 4. Define Hyvärinen loss (same as before)
# ============================================================
def hyvarinen_loss(model, x, y):
    x.requires_grad_(False)
    y.requires_grad_(True)
    psi = model(x, y)
    loss1 = 0.5 * (psi ** 2).sum(dim=1).mean()
    grads = []
    for i in range(psi.shape[1]):
        grad = torch.autograd.grad(psi[:, i].sum(), y, create_graph=True)[0][:, i]
        grads.append(grad)
    divergence = torch.stack(grads, dim=1).sum(dim=1)
    loss2 = divergence.mean()
    return loss1 + loss2

# ============================================================
# 5. Continue training with smaller learning rate
# ============================================================
lr = 1e-5   # or 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 100   # continue for another 100 epochs
for epoch in tqdm(range(num_epochs), desc="Continuing training"):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        loss = hyvarinen_loss(model, x_batch, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    tqdm.write(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.6f}")

# ============================================================
# 6. Save updated model
# ============================================================
new_model_path = f"continue_layer6_messy_markov_lr{lr:.0e}_A_0.73_b_1.90_Sigma_0.05.pth"
torch.save(model.state_dict(), new_model_path)
print(f"Continued model saved to {new_model_path}")
