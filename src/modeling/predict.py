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


def load_model(
    model_filepath: str,
    user_embed_dim: int = 384,
    item_embed_dim: int = 384,
    hidden_dim: int = 384,
    output_dim: int = 384,
    dropout_p: float = 0.5,
):
    model = TwoTowerModel(
        user_embed_dim=user_embed_dim,
        item_embed_dim=item_embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_p=dropout_p,
    )
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()
    return model


def predict(model, dataloader):
    result = []
    model.eval()
    for user_embeds, item_embeds, labels in tqdm(dataloader["test"]):
        user_repr, item_repr = model(user_embeds, item_embeds)
        cosine = F.cosine_similarity(user_repr, item_repr, dim=1, eps=1e-8)
        result.append(torch.vstack([cosine, labels]).T)
    result_df = pd.DataFrame(torch.cat(result).tolist())
    result_df.columns = ["proba", "label"]
    return result_df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("input_model_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--batch_size", type=int, default=32)
@click.option(
    "--output_chart_dir", type=click.Path(), default="reports/figures/predict/"
)
@click.option("--dropout_p", type=float, default=0.5)
@click.option("--hidden_dim", type=int, default=384)
@click.option("--output_dim", type=int, default=384)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("predict")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load model
    model = load_model(
        kwargs["input_model_filepath"],
        hiddin_dim=kwargs["hidden_dim"],
        output_dim=kwargs["output_dim"],
        dropout_p=kwargs["dropout_p"],
    )

    # make dataloader
    dataloader = make_dataloader(
        kwargs["input_filepath"], batch_size=kwargs["batch_size"]
    )

    # build features
    predicted_df = predict(model, dataloader)

    # output file
    predicted_df.to_parquet(kwargs["output_filepath"])

    # logging
    logger.info(predicted_df.describe())

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
