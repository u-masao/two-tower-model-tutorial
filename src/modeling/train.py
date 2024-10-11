import click
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

from src import config  # noqa: F401
from src.modeling.data_loader import make_dataloader
from src.modeling.model import TwoTowerModel


def train(dataloader, num_epochs=3):

    logger.info(dataloader)
    for u, i, l in dataloader["train"]:
        logger.info(f"{u=}")
        logger.info(f"{i=}")
        logger.info(f"{l=}")
        logger.info(f"{u.shape=}")
        logger.info(f"{i.shape=}")
        logger.info(f"{l.shape=}")
        break

    # モデル、オプティマイザ、損失関数の定義
    model = TwoTowerModel(user_embed_dim=384, item_embed_dim=384)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()

    # 学習ループ
    for epoch in tqdm(range(num_epochs)):
        for user_embeds, item_embeds, labels in tqdm(dataloader["train"]):
            optimizer.zero_grad()
            user_repr, item_repr = model(user_embeds, item_embeds)
            loss = criterion(user_repr, item_repr, labels)
            loss.backward()
            optimizer.step()

    return model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--num_epochs", type=int, default=1)
@click.option("--batch_size", type=int, default=32)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("build features")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # make dataloader
    dataloader = make_dataloader(
        kwargs["input_filepath"], batch_size=kwargs["batch_size"]
    )

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
