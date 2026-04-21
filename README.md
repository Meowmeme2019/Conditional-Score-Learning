# Conditional Score Learning for Quickest Change Detection in Markov Transition Kernels

**Wuxia Chen, Taposh Banerjee, Vahid Tarokh**  
IEEE Transactions on Signal Processing (Accepted, 2026)  
[[Paper]](https://arxiv.org/abs/YOUR_ARXIV_ID) <!-- replace with your arXiv link -->

---

## Overview

This repository contains the code for our paper on quickest change detection in 
high-dimensional Markov processes with unknown transition kernels.

Classical change detection methods (e.g., CUSUM) require explicit knowledge of 
the pre- and post-change likelihood ratio — which is intractable in high dimensions. 
We propose learning the **conditional score function** ∇_y log p(y|x) directly from 
data using neural networks, and using it to build a likelihood-free CUSUM statistic 
(SCUSUM) based on Hyvärinen score differences.

Key contributions:
- A conditional score learning framework for dependent (Markov) sequential data
- A score-based CUSUM procedure with a truncated variant for numerical stability
- Theoretical guarantees: exponential lower bounds on mean time to false alarm 
  (via Hoeffding's inequality for uniformly ergodic Markov chains) and asymptotic 
  upper bounds on detection delay
- Empirical validation on non-Gaussian synthetic Markov chains and real-world 
  CMU Motion Capture data (93-dimensional, 120 Hz)

---

## Repository Structure

```
.
├── non_Gaussian_MC/                          # Synthetic non-Gaussian Markov chain experiments
│   ├── Train_NonGaussian_ConditionalScore.py # Training script for non-Gaussian MC
│   ├── change_detecion_Gaussian_Kernel_nonlinear_mean_submitted_version_oct_30.ipynb
│   └── markov_chain_*.pt / *.pth            # Saved trajectories and model checkpoints
│
├── training conditional score and testing accuracy/   # CMU & Gaussian kernel experiments
│   ├── Messy_data_Training_conditional_score.py       # Score network training (main)
│   ├── Messy_data_Training_resume.py                  # Resume training from checkpoint
│   ├── train_two_markov_chains.py                     # Train on two Markov chain settings
│   ├── MoCapAnimateAsfAmc.py                          # Motion capture data loader
│   ├── Hyvarinens_score_diff_2_markov_chains.ipynb    # Score difference visualization
│   ├── Synthesize_Gaussian_datapath.ipynb             # Synthetic data generation
│   ├── Training_conditional_score.ipynb               # Training notebook
│   ├── load_path_to_see_conditional_score.ipynb       # Load and inspect trained scores
│   └── P_*.pth / *.pt                                 # Saved model checkpoints
│
├── training_amc/                             # CMU Motion Capture data (AMC/ASF format)
│   └── running/                             # Motion sequences by subject
│
├── .gitignore
└── README.md
```
---


## Experiments

### Synthetic Non-Gaussian Markov Chains
- Data: Markov chains in **R^10** with sequences of length **tens of thousands**
- Pre-change: α = 0.3, σ = 0.3; Post-change: α = 0.6, σ = 0.5
- Score learning relative error: ~2% on held-out data
- SCUSUM detects change at n = 120 with detection delay = 16

### CMU Motion Capture (Real Data)
- Data: **93-dimensional** joint angle time-series at **120 Hz**
- Three activity transitions tested: Running→Basketball, Basketball→Jumping, Jumping→Running
- Detection delays: 33–74 frames (**0.25–0.6 seconds**) using a universal threshold

---

## Requirements

```bash
pip install torch numpy scipy matplotlib
```

Training was run on GPU via university HPC cluster (Slurm).

---

## Usage

**Train score network:**
```bash
python Messy_data_Training_conditional_score.py
```

**Resume training from checkpoint:**
```bash
python Messy_data_Training_resume.py
```

**Train on two Markov chain settings:**
```bash
python train_two_markov_chains.py
```

---

## Citation

```bibtex
@article{chen2026conditional,
  title={Conditional Score Learning for Quickest Change Detection in Markov Transition Kernels},
  author={Chen, Wuxia and Banerjee, Taposh and Tarokh, Vahid},
  journal={IEEE Transactions on Signal Processing},
  year={2026}
}
```

---

## Acknowledgements

This work was supported in part by the U.S. National Science Foundation 
under Grants 2334897 and 2334898.
