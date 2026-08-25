"""
Pure-torch benchmark, no gym/SB3/sim involved at all. Isolates whether
slow training is a torch/CPU-environment problem (this script is also
slow) or specific to the PPO/rollout-buffer path (this script is fast,
so the problem is elsewhere -- e.g. memory/swap pressure from the
rollout buffer).

    python isolate_torch_cost.py

While this runs, in another terminal watch memory pressure:

    free -h -s 2       # updates every 2s; watch "available" and "swap"
    # or: htop / vmstat 2
"""
import time
import torch
import torch.nn as nn

print(f"torch.get_num_threads() = {torch.get_num_threads()}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")


class SmallGridCNN(nn.Module):
    def __init__(self, features_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * 17 * 17, features_dim), nn.ReLU())
        self.head = nn.Linear(features_dim, 6)  # stand-in policy head

    def forward(self, x):
        return self.head(self.linear(self.cnn(x)))


model = SmallGridCNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

batch_size = 4096
n_minibatches = 16
n_epochs = 10
total_passes = n_minibatches * n_epochs

print(f"Running {total_passes} forward+backward passes of batch_size={batch_size} "
      f"on shape (8,17,17) -- matching one PPO train() call's workload...")

t0 = time.time()
for _ in range(total_passes):
    x = torch.rand(batch_size, 8, 17, 17)
    target = torch.randint(0, 6, (batch_size,))
    opt.zero_grad()
    out = model(x)
    loss = nn.functional.cross_entropy(out, target)
    loss.backward()
    opt.step()
dt = time.time() - t0

print(f"Total: {dt:.2f}s for {total_passes} passes "
      f"({total_passes * batch_size / dt:,.0f} samples/s)")
print()
print("If this number is also very slow (tens/hundreds of seconds), the "
      "problem is torch/CPU-environment-level, not our training code.")
print("If this is fast (a few seconds or less) but the real model.train() "
      "call is still ~330s, the difference is memory/swap pressure from "
      "the much larger rollout buffer -- check `free -h` while training.")