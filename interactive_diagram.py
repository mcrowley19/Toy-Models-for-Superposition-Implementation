# My plan is to take advantage of how small the weight matrix is to regularly take snapshots of it and use that to render my own canvas on my website
# Preferred to Matplotlib so I can continue to brag about how performant it is
import torch
import torch.nn as nn
import numpy as np

class ReLUToyModel(nn.Module):
  # Parameters from the paper: n = 20, m = 5, I_i = 0.7 ** i

  def __init__(self):
    super().__init__()

    # We do this rather than passing them as default parameters so we can use the "device" parameter for .to(device)
    importance=(0.9**torch.arange(5))[None, :]
    feature_probability = (20 ** -torch.linspace(0, 1, 50))[None,:, None]

    device = 'mps'
    self.feature_probability = feature_probability.to(device)
    self.importance = importance.to(device)
    self.device = device

    self.m = 2
    self.n_instances = 50
    self.n = 5
    self.n_batches = 1024
    self.W = nn.Parameter(torch.empty((50, 5, 2), device=device))
    nn.init.xavier_normal_(self.W)
    self.b = nn.Parameter(torch.zeros((50, 5),device=self.device))


  def forward(self, x):
    x = x.unsqueeze(-2)
    W_expanded = self.W.unsqueeze(0)
    h = x @ W_expanded
    h = h.squeeze(-2)
    W_extended2 = self.W.unsqueeze(0).transpose(-2,-1)
    h_extended = h.unsqueeze(-2)
    x2 = h_extended @ W_extended2
    x2 = (x2.squeeze(-2)) + self.b
    nonlin = nn.ReLU()
    return nonlin(x2)


  def generate_data(self):
      feat = torch.rand(self.n_batches, self.n_instances, self.n, device=self.W.device)

      # This is like an if statement.
      # We are making a tensor where each element has 30% chance of showing element from our feat tensor and 70% chance of showing 0

      batch = torch.where(
          torch.rand((self.n_batches, self.n_instances, self.n), device=self.W.device) <= self.feature_probability,
          feat,
          torch.zeros((self.n_batches, self.n_instances, self.n), device=self.W.device)
      )

      return batch
  

model = ReLUToyModel()
optim = torch.optim.AdamW([model.W, model.b], lr=1e-3)

save_every = 40
write_arr = []
for step in range(10000):
    optim.zero_grad()
    batch = model.generate_data()
    out = model(batch)

    loss = (model.importance * (batch.abs() - out)**2).mean(dim=(0,2)).sum()
    loss.backward()
    optim.step()
    if step % save_every == 0:
        write_arr.append(model.W.detach().cpu().numpy().copy())
    if step % 1000 == 0:
      print("loss: ",loss)


arr = np.stack(write_arr).astype(np.float32) 
arr.tofile('matrices.bin') 