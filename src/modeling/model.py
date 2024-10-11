import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim, hidden_dim=64):
        super(TwoTowerModel, self).__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, user_embed, item_embed):
        user_repr = self.user_tower(user_embed)
        item_repr = self.item_tower(item_embed)
        return user_repr, item_repr
