import click
import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from src import config  # noqa: F401
from src.modeling.data_loader import make_dataloader
from src.modeling.model import TwoTowerModel


def predict(model, dataloader):
    result = []
    model.eval()
    for user_embeds, item_embeds, labels in tqdm(dataloader["test"]):
        user_repr, item_repr = model(user_embeds, item_embeds)
        cosine = F.cosine_similarity(user_repr, item_repr, dim=1, eps=1e-8)
        predict = (cosine > 0.5).to(int)
        logger.info(predict == labels)
        result.append(predict)
    return torch.cat(result)


def load_model(model_filepath, user_embed_dim=384, item_embed_dim=384):
    model = TwoTowerModel(
        user_embed_dim=user_embed_dim, item_embed_dim=item_embed_dim
    )
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()
    return model


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("input_model_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--batch_size", type=int, default=32)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("predict")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load model
    model = load_model(kwargs["input_model_filepath"])

    # make dataloader
    dataloader = make_dataloader(
        kwargs["input_filepath"], batch_size=kwargs["batch_size"]
    )

    # build features
    predicted = predict(model, dataloader)

    # output file
    pd.DataFrame(predicted.tolist()).to_parquet(kwargs["output_filepath"])

    # logging

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
