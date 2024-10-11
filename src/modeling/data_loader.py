from typing import List, Tuple

import cloudpickle
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src import config  # noqa: F401


class TripletDataset(Dataset):
    def __init__(
        self,
        user_embeds: pd.DataFrame,
        item_embeds: pd.DataFrame,
        ratings: pd.DataFrame,
        rating_threshold: int = 5,
    ) -> None:
        self.user_embeds: pd.DataFrame = user_embeds
        self.item_embeds: pd.DataFrame = item_embeds
        self.ratings: pd.DataFrame = ratings
        self.rating_threshold: int = rating_threshold

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[List[float], List[float], int]:
        targets: pd.Series = self.ratings.iloc[idx]
        users: List[float] = self.user_embeds.loc[targets["user-id"]].values
        items: List[float] = self.item_embeds.loc[targets["isbn"]].values
        ratings: int = (
            (
                (targets["book-rating"] > self.rating_threshold).astype(float)
                - 0.5
            )
            * 2
        ).astype(int)
        return users, items, ratings


def make_dataloader(input_filepath, batch_size=32):

    # load dataset
    data = cloudpickle.load(open(input_filepath, "rb"))
    result = {}
    for phase, df in data.items():
        dataset = TripletDataset(df["users"], df["items"], df["ratings"])
        result[phase] = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    return result
