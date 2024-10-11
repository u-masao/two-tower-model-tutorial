from collections import OrderedDict
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

from src import config  # noqa: F401
from src.modeling.data_loader import make_dataloader
from src.modeling.model import TwoTowerModel


def train(
    dataloader,
    output_model_dir,
    num_epochs=3,
    log_interval=500,
    model_save_interval_epochs=10,
):
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

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
    model = TwoTowerModel(
        user_embed_dim=384, item_embed_dim=384, hidden_dim=384, output_dim=384
    )
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()

    # 学習ループ
    for epoch in range(num_epochs):
        with tqdm(dataloader["train"]) as pbar:
            for batch_idx, (user_embeds, item_embeds, labels) in enumerate(
                pbar
            ):
                optimizer.zero_grad()
                user_repr, item_repr = model(user_embeds, item_embeds)
                loss = criterion(user_repr, item_repr, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    pbar.set_description(f"epoch={epoch}")
                    pbar.set_postfix(OrderedDict(loss=loss.item()))
                    metrics = {"loss": loss.item()}
                    step = len(pbar) * epoch + batch_idx
                    mlflow.log_metrics(metrics, step=step)

        if epoch % model_save_interval_epochs == 0:
            torch.save(
                model.state_dict(),
                output_model_dir / f"model_epoch{epoch:04}.pth",
            )

    return model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--output_model_dir", type=click.Path(), default="models/logs/")
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--num_epochs", type=int, default=1)
@click.option("--batch_size", type=int, default=32)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("train")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # autolog
    mlflow.pytorch.autolog()

    # make dataloader
    dataloader = make_dataloader(
        kwargs["input_filepath"], batch_size=kwargs["batch_size"]
    )

    # build features
    model = train(
        dataloader,
        num_epochs=kwargs["num_epochs"],
        output_model_dir=kwargs["output_model_dir"],
    )

    # output file
    torch.save(model.state_dict(), kwargs["output_filepath"])

    # logging

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
