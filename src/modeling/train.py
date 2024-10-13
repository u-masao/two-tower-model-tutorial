from collections import OrderedDict
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config  # noqa: F401
from src.modeling.data_loader import make_dataloader
from src.modeling.model import TwoTowerModel


def count_correct(
    user_repr: torch.Tensor,
    item_repr: torch.Tensor,
    labels: torch.Tensor,
    proba_threshold: float,
    cosine_eps: float = 1e-8,
):
    # コサインを計算
    cosine = F.cosine_similarity(user_repr, item_repr, dim=1, eps=cosine_eps)

    # 正解件数をカウント
    return (cosine > proba_threshold).eq(labels == 1).sum().item(), (
        cosine > proba_threshold
    ).sum().item()


def train_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CosineEmbeddingLoss,
    epoch: int,
    log_interval: int = 100,
    proba_threshold: float = 0.5,
):

    # init vars
    train_loss = 0
    correct = 0
    positive = 0

    # switch model to train mode
    model.train()

    with tqdm(dataloader["train"]) as pbar:
        for batch_idx, (user_embeds, item_embeds, labels) in enumerate(pbar):

            # train
            optimizer.zero_grad()
            user_repr, item_repr = model(user_embeds, item_embeds)
            train_batch_loss = criterion(user_repr, item_repr, labels)
            train_loss += train_batch_loss.item()
            train_batch_loss.backward()
            optimizer.step()

            # コサインを計算し正解件数をカウント
            batch_correct, batch_positive = count_correct(
                user_repr, item_repr, labels, proba_threshold
            )
            correct += batch_correct
            positive += batch_positive

            # logging
            if batch_idx % log_interval == 0:
                pbar.set_description(f"train epoch={epoch}")
                pbar.set_postfix(
                    OrderedDict(train_batch_loss=train_batch_loss.item())
                )
                metrics = {
                    "train.batch.loss": train_batch_loss.item(),
                    "train.loss.mean": train_loss / (batch_idx + 1),
                }
                step = len(pbar) * epoch + batch_idx
                mlflow.log_metrics(metrics, step=step)

    # make metrics
    train_data_size = len(dataloader["train"].dataset)
    train_loss /= train_data_size
    train_metrics = {
        "train.loss": train_loss,
        "train.data_size": train_data_size,
        "train.correct": correct,
        "train.accuracy": correct / train_data_size,
        "train.positive": positive,
    }
    logger.info(f"{train_metrics=}")
    mlflow.log_metrics(
        train_metrics,
        step=epoch,
    )
    return train_metrics


def test_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CosineEmbeddingLoss,
    epoch: int,
    log_interval: int = 100,
    proba_threshold: float = 0.5,
):

    # init vars
    test_loss = 0
    correct = 0
    positive = 0

    # switch model to eval
    model.eval()

    with torch.no_grad():
        with tqdm(dataloader["test"]) as pbar:
            for batch_idx, (user_embeds, item_embeds, labels) in enumerate(
                pbar
            ):
                # 推論
                user_repr, item_repr = model(user_embeds, item_embeds)

                # calc loss
                test_batch_loss = criterion(
                    user_repr, item_repr, labels
                ).item()
                test_loss += test_batch_loss

                # 正解件数をカウント
                batch_correct, batch_positive = count_correct(
                    user_repr, item_repr, labels, proba_threshold
                )
                correct += batch_correct
                positive += batch_positive

                # logging
                if batch_idx % log_interval == 0:
                    pbar.set_description(f"test epoch={epoch}")
                    pbar.set_postfix(
                        OrderedDict(test_batch_loss=test_batch_loss)
                    )
                    metrics = {"test.batch.loss": test_batch_loss}
                    step = len(pbar) * epoch + batch_idx
                    mlflow.log_metrics(metrics, step=step)

    # make metrics
    test_data_size = len(dataloader["test"].dataset)
    test_loss /= test_data_size
    test_metrics = {
        "test.loss": test_loss,
        "test.data_size": test_data_size,
        "test.correct": correct,
        "test.accuracy": correct / test_data_size,
        "test.positive": positive,
    }
    logger.info(f"{test_metrics=}")
    mlflow.log_metrics(
        test_metrics,
        step=epoch,
    )
    return test_metrics


def train(
    dataloader: DataLoader,
    output_model_dir: str,
    num_epochs: int = 3,
    log_interval: int = 100,
    model_save_interval_epochs: int = 10,
    proba_threshold: float = 0.001,
    loss_margin: float = 0.0,
    hidden_dim: int = 384,
    output_dim: int = 384,
    lr: float = 0.001,
):

    # make output dir
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    # 入力データの次元数確認
    some_sample = dataloader["train"].dataset[0]
    user_embed_dim = some_sample[0].shape[0]
    item_embed_dim = some_sample[1].shape[0]

    # モデル、オプティマイザ、損失関数の定義
    model = TwoTowerModel(
        user_embed_dim=user_embed_dim,
        item_embed_dim=item_embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=loss_margin)

    # train evaluate loop
    for epoch in range(num_epochs):
        # train
        train_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            epoch,
            log_interval=log_interval,
        )

        if epoch % model_save_interval_epochs == 0:
            torch.save(
                model.state_dict(),
                output_model_dir / f"model_epoch{epoch:04}.pth",
            )

        # evaluate
        test_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            epoch,
            log_interval=log_interval,
            proba_threshold=proba_threshold,
        )

    return model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--output_model_dir", type=click.Path(), default="models/logs/")
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--num_epochs", type=int, default=1)
@click.option("--batch_size", type=int, default=32)
@click.option("--model_save_interval_epochs", type=int, default=1)
@click.option("--proba_threshold", type=float, default=0.5)
@click.option("--log_interval", type=int, default=500)
@click.option("--loss_margin", type=float, default=0.5)
@click.option("--hidden_dim", type=int, default=384)
@click.option("--output_dim", type=int, default=384)
@click.option("--lr", type=float, default=0.001)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("train")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # make dataloader
    dataloader = make_dataloader(
        kwargs["input_filepath"], batch_size=kwargs["batch_size"]
    )

    # build features
    model = train(
        dataloader,
        output_model_dir=kwargs["output_model_dir"],
        num_epochs=kwargs["num_epochs"],
        log_interval=kwargs["log_interval"],
        model_save_interval_epochs=kwargs["model_save_interval_epochs"],
        proba_threshold=kwargs["proba_threshold"],
        loss_margin=kwargs["loss_margin"],
        hidden_dim=kwargs["hidden_dim"],
        output_dim=kwargs["output_dim"],
        lr=kwargs["lr"],
    )

    # output file
    torch.save(model.state_dict(), kwargs["output_filepath"])

    # logging

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
