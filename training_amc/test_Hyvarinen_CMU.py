#%%


import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

print("Current working directory:", os.getcwd())
print("Project root added:", project_root)
print("File exists?", os.path.exists(os.path.join(project_root, "Messy_data_Training_conditional_score.py")))
print("sys.path entries:")
for p in sys.path:
    print("   ", p)



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
M = 1e9  # truncation level

# Paths
root_dir = "/ihome/tbanerjee/wuc3/markov_train/training_amc"
save_dir = root_dir
hybrid_path = os.path.join(save_dir, "running_basketball_hybrid_path.pt")
score_output_path = os.path.join(save_dir, "running_basketball_score_difference.pt")


# Trial configuration
training_sets = {
    "running": {
        "02": ["03"],
        "09": [f"{i:02d}" for i in range(1, 12)],
        "16": ["08"] + [f"{i:02d}" for i in range(35, 47)] + [f"{i:02d}" for i in range(48, 58)],
        "35": [f"{i:02d}" for i in range(17, 27)]
        #"38": ["03"] # "38" for testing.
    },
    "basketball": {
        "06": [f"{i:02d}" for i in range(2, 15)]  # 02-14 for training, 15 for testing. 
    }
}
    

# === TEST PATH GENERATION (Hybrid) ===
print("Generating hybrid path for testing...")
run_asf = os.path.join(root_dir, "running", "Subject_38", "38.asf")
run_amc = os.path.join(root_dir, "running", "Subject_38", "Trial_03.amc")

bb_asf = os.path.join(root_dir, "basketball", "Subject_06", "06.asf")
bb_amc = os.path.join(root_dir, "basketball", "Subject_06", "Trial_15.amc")

motion_run = MotionCapture(run_asf, run_amc)
T1, B1, _ = motion_run.data.shape
X_run = motion_run.data.reshape(T1, B1 * 3)

motion_bb = MotionCapture(bb_asf, bb_amc)
T2, B2, _ = motion_bb.data.shape
X_bb = motion_bb.data.reshape(T2, B2 * 3)

assert B1 == B2, "Mismatch in bone count between running and basketball trials"
X_hybrid = torch.tensor(np.concatenate([X_run, X_bb], axis=0), dtype=torch.float32)
torch.save(X_hybrid, hybrid_path)
print(f"Saved hybrid path to: {hybrid_path}")
print(f"Shape: {X_hybrid.shape} = ({T1}+{T2}, {B1*3})")


# === TESTING PHASE: HYVARINEN SCORE DIFFERENCE ===
print(" Testing score difference between trained models...")
X = torch.load(hybrid_path).float().to(device)
x_pairs = X[:-1]
y_pairs = X[1:]

# Load models
def get_model_path(name):
    paths = glob.glob(os.path.join(save_dir, name))
    if not paths:
        raise FileNotFoundError(f" Could not find model matching: {name}")
    return paths[0]

P0_model_pth = get_model_path("512_4_running.pth")
P1_model_pth = get_model_path("512_4_basketball.pth")


model_P0 = ConditionalScoreNet(d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
model_P1 = ConditionalScoreNet(d, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
model_P0.load_state_dict(torch.load(P0_model_pth, map_location=device))
model_P1.load_state_dict(torch.load(P1_model_pth, map_location=device))
model_P0.eval()
model_P1.eval()

with torch.no_grad():
    score_p0 = compute_hyvarinen_score(model_P0, x_pairs, y_pairs, device)
    score_p1 = compute_hyvarinen_score(model_P1, x_pairs, y_pairs, device)
    diff_score = score_p0 - score_p1
    diff_score = torch.clamp(diff_score, min=-M, max=M).cpu().numpy()

torch.save(torch.tensor(diff_score), score_output_path)
print(f"Saved score difference to: {score_output_path}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(diff_score, label="Hyvärinen score difference", lw=0.8)
plt.axvline(T1, color="red", linestyle="--", label=f"Change point (n={T1})")
plt.axhline(0, color="black", linestyle="--", lw=0.8)
plt.xlabel("Time index (n)")
plt.ylabel("Score: S_H(running) - S_H(basketabll)")
plt.title("Hyvärinen Score Difference Across running → basketball")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "hyvarinen_score_plot.pdf"))
plt.show()

# Print summary stats
print(f"[P0 region] Mean: {diff_score[:T1].mean():.4f}")
print(f"[P1 region] Mean: {diff_score[T1+1:].mean():.4f}")