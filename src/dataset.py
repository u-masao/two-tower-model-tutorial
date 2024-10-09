from typing import Any, Dict

import click
import cloudpickle
import mlflow
import pandas as pd
from loguru import logger

from src import config  # noqa: F401


def get_data_types():
    return {
        "users": {
            "User-ID": int,
            "Location": str,
            "Age": float,
        },
        "items": {
            "ISBN": str,
            "Book-Title": str,
            "Book-Author": str,
            "Year-Of-Publication": int,
            "Publisher": str,
            "Image-URL-S": str,
            "Image-URL-M": str,
            "Image-URL-L": str,
        },
        "ratings": {
            "User-ID": int,
            "ISBN": str,
            "Book-Rating": int,
        },
    }


def load_dataset(
    input_users_path,
    input_items_path,
    input_ratings_path,
    delimiter=";",
    encoding="ISO-8859-1",
):

    data_types = get_data_types()

    # load users data
    users = pd.read_csv(
        input_users_path,
        sep=delimiter,
        encoding=encoding,
        dtype=data_types["users"],
    )

    # load items data
    items = pd.read_csv(
        input_items_path,
        sep=delimiter,
        encoding=encoding,
        dtype=data_types["items"],
    )

    # load ratings data
    ratings = pd.read_csv(
        input_ratings_path,
        sep=delimiter,
        encoding=encoding,
        dtype=data_types["ratings"],
    )

    # column name to lower
    for df in [users, items, ratings]:
        df.columns = df.columns.str.lower()

    return users, items, ratings


def split_dataset(
    users: pd.DataFrame,
    items: pd.DataFrame,
    ratings: pd.DataFrame,
    test_sample_ratio: float,
    random_state: int = 1234,
) -> Dict[str, Any]:

    # users: random split
    users_test = users.sample(
        frac=test_sample_ratio, random_state=random_state
    )
    users_train = users.drop(users_test.index)

    # items: time series split(publish year)
    year_column = "year-of-publication"
    target_items = items[
        items[year_column].between(1950, 2020, inclusive="both")
    ]
    year_threshold = (
        ratings.merge(target_items, on="isbn", how="inner")[year_column]
        .quantile(1 - test_sample_ratio)
        .item()
    )
    items_test = target_items[target_items[year_column] >= year_threshold]
    items_train = target_items.drop(items_test.index)

    logger.info(f"{year_threshold=}")

    # ratings: depends users and items
    ratings_test = ratings.merge(
        users_test["user-id"], how="inner", on="user-id"
    ).merge(items_test["isbn"], how="inner", on="isbn")
    ratings_train = ratings.merge(
        users_train, how="inner", on="user-id"
    ).merge(items_train, how="inner", on="isbn")

    return {
        "train": {
            "users": users_train,
            "items": items_train,
            "ratings": ratings_train,
        },
        "test": {
            "users": users_test,
            "items": items_test,
            "ratings": ratings_test,
        },
    }


@click.command()
@click.argument("input_users_path", type=click.Path(exists=True))
@click.argument("input_items_path", type=click.Path(exists=True))
@click.argument("input_ratings_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--test_sample_ratio", type=float, default=0.2)
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

    # debug output
    for df in [users, items, ratings]:
        logger.info(df)
        logger.info(df.dtypes)
        logger.info(df.describe())

    # split dataset
    splitted = split_dataset(
        users,
        items,
        ratings,
        test_sample_ratio=kwargs["test_sample_ratio"],
    )

    # make output data
    log_params = {}
    for train_test in ["train", "test"]:
        for key in ["users", "items", "ratings"]:
            df = splitted[train_test][key]
            log_params[f"input.length.{train_test}.{key}"] = df.shape[0]
            log_params[f"input.columns.{train_test}.{key}"] = df.shape[1]

    # output file
    cloudpickle.dump(splitted, open(kwargs["output_path"], "wb"))

    # logging
    mlflow.log_params(log_params)
    logger.info(log_params)

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
