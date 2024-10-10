import click
import cloudpickle
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import config  # noqa: F401
from src.modeling.model import TwoTowerModel


class TripletDataset(Dataset):
    def __init__(self, user_embeds, item_embeds, ratings, rating_threshold=5):
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.ratings = ratings
        self.rating_threshold = rating_threshold

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        targets = self.ratings.iloc[idx]
        users = self.user_embeds.loc[targets["user-id"]].values
        items = self.item_embeds.loc[targets["isbn"]].values
        ratings = (targets["book-rating"] > self.rating_threshold).astype(int)
        return users, items, ratings


def train(dataloader, num_epochs=3):
    # モデル、オプティマイザ、損失関数の定義
    model = TwoTowerModel(user_embed_dim=384, item_embed_dim=384)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()

    logger.info(dataloader)
    for u, i, l in dataloader:
        logger.info(f"{u=}")
        logger.info(f"{i=}")
        logger.info(f"{l=}")
        logger.info(f"{u.shape=}")
        logger.info(f"{i.shape=}")
        logger.info(f"{l.shape=}")
        break

    # 学習ループ
    for epoch in tqdm(range(num_epochs)):
        for user_embeds, item_embeds, labels in tqdm(dataloader):
            optimizer.zero_grad()
            user_repr, item_repr = model(user_embeds, item_embeds)
            loss = criterion(user_repr, item_repr, labels)
            loss.backward()
            optimizer.step()

    return model


def load_dataset(input_filepath):
    data = cloudpickle.load(open(input_filepath, "rb"))["train"]
    dataset = TripletDataset(data["users"], data["items"], data["ratings"])
    return dataset


def make_dataloader(dataset):

    return DataLoader(dataset, batch_size=32, shuffle=True)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--num_epochs", type=int, default=1)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("build features")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load dataset
    dataset = load_dataset(kwargs["input_filepath"])

    # make dataloader
    dataloader = make_dataloader(dataset)

    # build features
    model = train(dataloader, num_epochs=kwargs["num_epochs"])

    # output file
    torch.save(model.state_dict(), kwargs["output_filepath"])

    # logging

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
