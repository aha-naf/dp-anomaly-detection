Differentially Private Trajectory Anomaly Detection:
This project studies anomaly detection on trajectory data under differential privacy using the GeoLife dataset.
Overview
We compare anomaly detection performance between:
Raw trajectories (no privacy)
Differentially private trajectories (Xiao-style PIM)
Pipeline:
Load and preprocess trajectories
Discretize into grid states
Apply point-wise DP (PIM)
Extract features
Train Isolation Forest
Compare anomaly scores
Files
unsupervised_xiao_fast.py → Main DP + anomaly detection pipeline
epsilon_sweep_experiment.py → Runs experiments across different ε values
Metrics
Spearman correlation
Top-k overlap (10, 20, 50)
Precision / Recall / F1
Mean absolute score difference

How to Run
python epsilon_sweep_experiment.py

Notes
Uses GeoLife dataset (not included)
Results saved in priv_exp_outputs/
