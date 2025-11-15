import sys
from pathlib import Path
# Ensure project root (one level up) is on path so `src` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import LatentODEModel

model = LatentODEModel(input_dim=6, hidden_dim=32, latent_dim=16, output_dim=6, dropout=0.1)
model.eval()

# Dummy batch
batch_size, seq_len, input_dim = 2, 10, 6
vitals = torch.randn(batch_size, seq_len, input_dim)
times = torch.linspace(0, 24, seq_len).unsqueeze(0).repeat(batch_size, 1)
mask = torch.ones(batch_size, seq_len)

out = model(vitals, times, mask)
print('OK - forward pass success')
for k, v in out.items():
    if hasattr(v, 'shape'):
        print(f"  - {k}: {v.shape}")
