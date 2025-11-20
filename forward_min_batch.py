import os, glob, numpy as np, torch, torch.nn as nn

root = "./datasets/EgoCOT_base"
files = sorted(glob.glob(os.path.join(root, "EGO_*.npy")))
assert files, f"No .npy files found under {root}"
sample_path = files[0]
arr = np.load(sample_path, allow_pickle=False)

x = torch.from_numpy(arr).float().reshape(1, -1)
in_dim = x.shape[1]

model = nn.Sequential(
    nn.Linear(in_dim, 128),
    nn.GELU(),
    nn.Linear(128, 16)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)

with torch.no_grad():
    y = model(x)

print(f"[OK] Loaded: {os.path.basename(sample_path)}")
print(f"Input shape: {tuple(x.shape)}  ->  Output shape: {tuple(y.shape)}  (device={device})")
