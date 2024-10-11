import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_embed_dim,
        item_embed_dim,
        hidden_dim=64,
        output_dim=64,
        dropout_p=0.5,
    ):
        super(TwoTowerModel, self).__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, user_embed, item_embed):
        user_repr = self.user_tower(user_embed)
        item_repr = self.item_tower(item_embed)
        return user_repr, item_repr
