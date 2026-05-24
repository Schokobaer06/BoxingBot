import torch.nn as nn

# Einfaches DQN-Netz für die Aktionsauswahl
class DQN(nn.Module):

    def __init__(self, action_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(3136, 512),
            nn.ReLU(),

            nn.Linear(512, action_size)
        )

    def forward(self, x):
        # Vorwärtsdurchlauf durch das Netz
        return self.network(x)