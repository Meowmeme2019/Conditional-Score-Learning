import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Messy_data_Training_conditional_score import ConditionalScoreNet, hyvarinen_loss, compute_hyvarinen_score
from MoCapAnimateAsfAmc import MotionCapture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
hidden_dim = 512
num_layers = 4
batch_size = 128
num_epochs = 300
lr = 5e-5
d = 93  # dimension of each frame


# Paths
root_dir = "/ihome/tbanerjee/wuc3/markov_train/training_amc"
save_dir = root_dir
 

# Trial configuration
# Trial configuration
training_sets = {
    "running": {
        "02": ["03"],
        "09": [f"{i:02d}" for i in range(1, 12)],
        "16": ["08"] + [f"{i:02d}" for i in range(35, 47)] + [f"{i:02d}" for i in range(48, 58)],
        "35": [f"{i:02d}" for i in range(17, 26)],          # trial 26 for testing
        "38": ["03"]                                            # "38" for testing.
    },
    # "basketball": {
    #     "06": [f"{i:02d}" for i in range(2, 15)]  # 02-14 for training, 15 for testing. 
    # },
    "jumping": {
        "13": ["11", "13", "19", "32", "39", "40", "41" ], 
        "16": [f"{i:02d}" for i in range(1, 11)], 
        "49": ["02", "03"]
    }
}

testing_sets = {
    "running": {
        "35": ["26"]    
    },
    "basketball": {
        "06": ["15"] 
    }, 
    "jumping": {
        "13": ["42"]
    }
}

def load_all_trials(class_name, trial_dict):
    datasets = []
    for subject, trials in trial_dict.items():
        subject_path = os.path.join(root_dir, class_name, f"Subject_{subject}")
        asf_file = os.path.join(subject_path, f"{subject}.asf")
        for trial in trials:
            amc_file = os.path.join(subject_path, f"Trial_{trial}.amc")
            try:
                motion = MotionCapture(asf_file, amc_file)
                T, B, _ = motion.data.shape
                X = motion.data.reshape(T, B * 3)
                x_data = X[:-1]
                y_data = X[1:]
                dataset = TensorDataset(
                    torch.tensor(x_data, dtype=torch.float32),
                    torch.tensor(y_data, dtype=torch.float32)
                )
                datasets.append(dataset)
            except Exception as e:
                print(f"failed to load {amc_file}: {e}")
    return ConcatDataset(datasets)

def train_model(class_name, trial_dict):
    print(f"Training class: {class_name}")
    dataset = load_all_trials(class_name, trial_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = ConditionalScoreNet(d=d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm(range(num_epochs), desc=f"Training {class_name}"):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = hyvarinen_loss(model, x_batch.to(device), y_batch.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"[{class_name} Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

    model_path = os.path.join(save_dir, f"{hidden_dim}_{num_layers}_{class_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {class_name} model to: {model_path}")

if __name__ == "__main__":
    # === TRAINING PHASE ===
    for class_name, trials in training_sets.items():
        train_model(class_name, trials)
    print("Training complete.")

