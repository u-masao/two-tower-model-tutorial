from typing import Any, Dict, List

import click
import cloudpickle
import mlflow
import pandas as pd
import torch.nn.functional as F
from loguru import logger
from object_cache import object_cache
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src import config  # noqa: F401


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_dataset(
    input_filepath: str,
):
    dataset = cloudpickle.load(open(input_filepath, "br"))
    return dataset


def make_sentences(dataset: Dict[str, Any]):
    sentences = {"users": "", "items": ""}

    # users section
    for key in ["age", "location"]:
        sentences["users"] += (
            f"### {key}\n\n"
            + dataset["users"][key].fillna("unknown").astype(str)
            + "\n"
        )

    # items section
    for key in [
        "book-title",
        "book-author",
        "year-of-publication",
        "publisher",
    ]:
        sentences["items"] += (
            f"### {key}\n\n"
            + dataset["items"][key].fillna("unknown").astype(str)
            + "\n"
        )

    # debug output
    logger.info(sentences)

    return sentences


@object_cache
def embed(
    sentences: List[str],
    model_name: str = "intfloat/multilingual-e5-small",
    max_length=512,
    enable_normalize=False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input texts
    batch_dict = tokenizer(
        [f"passage: {x}" for x in sentences],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )

    # normalize embeddings
    if enable_normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.detach().numpy()


def embed_chunked(
    data, sentences, data_name, data_key, data_prefix, chunk_size
):
    items = []
    for i in tqdm(range(0, len(sentences[data_name]), chunk_size)):
        item_embeds = embed(sentences[data_name][i:].head(chunk_size))
        temp_df = pd.DataFrame(
            item_embeds,
            index=data[data_name][data_key][i:].head(chunk_size),
            columns=[f"{data_prefix}{x}" for x in range(item_embeds.shape[1])],
        )
        items.append(temp_df)
    return pd.concat(items)


def build_features(dataset: Dict[str, Any], threshold=5, chunk_size=100):
    result = {}
    for category in ["train", "test"]:

        # make sentences
        sentences = make_sentences(dataset[category])

        # make user embeds
        users_df = embed_chunked(
            dataset[category], sentences, "users", "user-id", "u", chunk_size
        )

        # make item embeds
        items_df = embed_chunked(
            dataset[category], sentences, "items", "isbn", "i", chunk_size
        )

        # merge
        ratings = dataset[category]["ratings"]
        ratings["rating"] = (ratings["book-rating"] > threshold).astype(int)
        result[category] = (
            ratings.merge(users_df, on="user-id", how="inner")
            .merge(items_df, on="isbn", how="inner")
            .drop(["isbn", "user-id", "book-rating"], axis=1)
        )

    return result


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
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

    # build features
    features = build_features(dataset)

    # output file
    cloudpickle.dump(features, open(kwargs["output_filepath"], "wb"))

    # logging

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
