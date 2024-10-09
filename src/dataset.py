import click
import cloudpickle
import mlflow
import pandas as pd
from loguru import logger

from src import config  # noqa: F401


def load_dataset(input_users_path, input_items_path, input_ratings_path):
    users = pd.read_csv(input_users_path)
    items = pd.read_csv(input_items_path)
    ratings = pd.read_csv(input_ratings_path)
    logger.info(users)
    logger.info(items)
    logger.info(ratings)

    return users, items, ratings


@click.command()
@click.argument("input_users_path", type=click.Path(exists=True))
@click.argument("input_items_path", type=click.Path(exists=True))
@click.argument("input_ratings_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("make dataset")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load dataset
    users, items, ratings = load_dataset(
        kwargs["input_users_path"],
        kwargs["input_items_path"],
        kwargs["input_ratings_path"],
    )

    # make output data
    results = {}
    log_params = {}
    for key in ["users", "items", "ratings"]:
        df = eval(key)
        results[key] = df
        log_params[f"input.length.{key}"] = df.shape[0]
        log_params[f"input.columns.{key}"] = df.shape[1]

    # output file
    cloudpickle.dump(open(kwargs["output_path"], "wb"), results)

    # logging
    mlflow.log_params(log_params)
    logger.info(log_params)

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
