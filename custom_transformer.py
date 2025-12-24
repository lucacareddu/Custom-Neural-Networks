import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
seed = 42 # any number 
set_deterministic(seed=seed)


class TimeSeriesDataset(Dataset):
    """
    Synthetic multivariate time-series dataset (>=4 features)
    with meaningful class-dependent patterns.
    
    Each sample shape: (seq_len, num_features)
    Label: 0 or 1
    """

    def __init__(self, num_samples=1000, seq_len=100, num_features=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features

        self.X, self.y = self._generate_dataset()

    def _generate_class0(self):
        """
        Class 0: low frequency + downward trend + mild noise
        """
        t = np.linspace(0, 2*np.pi, self.seq_len)
        
        v0 = 0.7 * np.sin(1 * t)                      # slower sine wave
        v1 = np.linspace(1, 0.2, self.seq_len)        # downward trend
        v2 = 0.2 * np.sin(0.5*t + 1)                  # correlated slow wave
        v3 = np.random.normal(0, 0.05, self.seq_len)  # noise variable

        X = np.stack([v0, v1, v2, v3], axis=1)

        # additional noisy features if needed
        if self.num_features > 4:
            extra = np.random.normal(0, 0.05, (self.seq_len, self.num_features - 4))
            X = np.concatenate([X, extra], axis=1)

        return X

    def _generate_class1(self):
        """
        Class 1: higher frequency + upward trend + event spikes
        """
        t = np.linspace(0, 4*np.pi, self.seq_len)

        v0 = 1.0 * np.sin(3 * t)                      # faster sine wave
        v1 = np.linspace(0.2, 1.2, self.seq_len)      # upward trend
        v2 = 0.3 * np.sin(2*t + 0.5)                  # correlated faster wave
        v3 = np.random.normal(0, 0.05, self.seq_len)

        # Add event spikes (makes class 1 very learnable)
        spike_positions = np.random.choice(self.seq_len, 3, replace=False)
        v0[spike_positions] += np.random.uniform(2, 4, 3)
        v2[spike_positions] += np.random.uniform(1, 3, 3)

        X = np.stack([v0, v1, v2, v3], axis=1)

        if self.num_features > 4:
            extra = np.random.normal(0, 0.05, (self.seq_len, self.num_features - 4))
            X = np.concatenate([X, extra], axis=1)

        return X

    def _generate_dataset(self):
        X_list = []
        y_list = []

        for _ in range(self.num_samples):
            label = np.random.randint(0, 2)

            if label == 0:
                x = self._generate_class0()
            else:
                x = self._generate_class1()

            X_list.append(x)
            y_list.append(label)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.long)
        return X, y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X, self.y


class Mod(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        h_size = 32
        self.h_size = h_size

        self.W1 = nn.Linear(8,h_size, bias=False)
        self.cls_tok = nn.Parameter(torch.randn(h_size))
        self.W_q, self.W_k, self.W_v = nn.Linear(h_size,h_size, bias=False), nn.Linear(h_size,h_size, bias=False), nn.Linear(h_size,h_size, bias=False)
        self.W2 = nn.Linear(h_size,1, bias=False)

    def forward(self, waves):
        B = waves.shape[0]
        h_1 = self.W1(waves)
        h_pre2 = torch.concat([h_1, self.cls_tok.unsqueeze(0).unsqueeze(0).expand(B,1,-1)], dim=1)
        Q, K, V = self.W_q(h_pre2), self.W_k(h_pre2), self.W_v(h_pre2)
        scores = torch.nn.functional.softmax(Q @ K.mT/ (self.h_size**0.5), dim=-1)
        h_2 = (scores @ V) + h_pre2
        z = self.W2(h_2[:,-1])
        return h_1.detach(), h_pre2.detach(), Q.detach(), K.detach(), V.detach(), scores.detach(), h_2.detach(), z.squeeze()


dataset = TimeSeriesDataset(
    num_samples=128,
    seq_len=256,
    num_features=8
)

waves, labels = dataset[0]
waves = (waves - waves.mean()) / (waves.std() + 1e-7)

f = Mod()

bce_loss = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(f.parameters(), lr=1)


# -------- Training loop --------
for epoch in range(300):
    h_1, h_pre2, Q, K, V, scores, h_2, z = f(waves)
    
    loss = bce_loss(z, labels.float())
    loss.backward()

    # ours vs autograd
    with torch.no_grad():
        
        W2_grad = h_2[:,-1][:,:,None] @ (z - labels)[:,None,None]
        print("Mine: ",W2_grad.mean(dim=0).T)
        print("Torch: ",f.W2.weight.grad)
        
        # f.W2.weight = torch.nn.Parameter(f.W2.weight - 1 * W2_grad.mean(dim=0).T)

        # Wk_grad = h_pre2.mT @ scores[:,-1].unsqueeze(-1) @ ((z - labels)[:,None,None] @ f.W2.weight)
        # f.W_k.weight = torch.nn.Parameter(f.W_k.weight - 1 * Wk_grad.mean(dim=0).T)

    # optim.step()
    optim.zero_grad()

    acc = ((torch.sigmoid(z) > 0.5).long() == labels).float().mean()

    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1} -> loss: {loss.item()}, acc: {acc}")
