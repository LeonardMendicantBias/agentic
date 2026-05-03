import torch
from torch import nn
import torch.optim as optim

import lightning as L


'''
    RL-based refiner takes reconstructed image, which can be blurry and noisy, as input, and outputs regions of interest.
    The regions of interest are modeled as a binary mask.
    The reward is defined as:
        1. Pixels are of interest and included in the results of AI-native tasks: positive reward
        2. Pixels are of interest but not included in the results of AI-native tasks: negative reward
        3. Pixels are not of interest but included in the results of AI-native tasks: negative reward
        4. Pixels are not of interest and not included in the results of AI-native tasks: positive reward
'''


class ActorCritic(nn.Module):

    def __init__(self,
        obs_dim,
        action_dim
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
        )

        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy(x)
        value = self.value(x)
        return logits, value
    
    
class A3CLightning(L.LightningModule):

    def __init__(self, obs_dim, action_dim, lr=1e-4, gamma=0.99):
        super().__init__()

        self.model = ActorCritic(obs_dim, action_dim)
        self.gamma = gamma
        self.lr = lr

        # Important for A3C
        self.model.share_memory()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
    