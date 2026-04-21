import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# -------------------------------------------------------------
# Include parent folder for network imports
# -------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Messy_data_Training_conditional_score import ConditionalScoreNet, hyvarinen_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
hidden_dim = 128
num_layers = 3
batch_size = 128
num_epochs = 100
lr = 1e-4

# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
root_dir = "/ihome/tbanerjee/wuc3/markov_train/non_Gaussian_MC"
save_dir = root_dir

# Training files (generated from your synthesis script)
train_files = [
    "markov_chain_alpha_0.30_dt_0.05_sigma_0.30_shift_0.20_full.pt",
    "markov_chain_alpha_0.60_dt_0.05_sigma_0.50_shift_0.90_full.pt",
]

# -------------------------------------------------------------
# Load one Markov chain and create (x, y) dataset
# -------------------------------------------------------------
def load_markov_chain(path):
    chain = torch.load(path)
    # first 1000 for burn-in, and leave last 10000 for testing
    train_chain = chain[1000:-10000]
    x = chain[:-1]
    y = chain[1:]
    return TensorDataset(x, y)

# -------------------------------------------------------------
# Train model on a given dataset
# -------------------------------------------------------------
def train_model(chain_name, dataset):
    print(f"\n=== Training Conditional ScoreNet on {chain_name} ===")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # infer dimension automatically
    d = dataset.tensors[0].shape[1]
    model = ConditionalScoreNet(d=d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc=f"Training {chain_name}"):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = hyvarinen_loss(model, x_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[{chain_name} | Epoch {epoch+1:03d}] Avg Loss: {avg_loss:.6f}")

    model_name = f"{chain_name}_hidden{hidden_dim}_layers{num_layers}.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    for file in train_files:
        chain_path = os.path.join(root_dir, file)
        dataset = load_markov_chain(chain_path)
        chain_name = os.path.splitext(file)[0]
        train_model(chain_name, dataset)

    print("\n All non-Gaussian chains trained successfully.")
