from pathlib import Path

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src import config  # noqa: F401


def plot_confusion_matrix(
    cm, classes, ax, normalize=False, title="", cmap=plt.cm.Blues
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def draw_histogram(y_pred_proba, title, ax):
    ax.hist(y_pred_proba, bins=100)
    ax.set_yscale("log")
    ax.grid()
    ax.set_ylim([0.1, None])
    ax.set_title(title)


def draw_roc_curve(fpr, tpr, roc_auc, ax):
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid()


def draw_pr_curve(recall, precision, pr_auc, ax):
    ax.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label="PR curve (area = %0.2f)" % pr_auc,
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curve")
    ax.legend(loc="lower right")
    ax.grid()


def analysis(predicted_df, output_dir, prob_threshold=0.0001):

    # define class name
    class_names = ["Negative", "Positive"]

    # 必要な値を作成
    y_true = (predicted_df["label"] == 1).astype(int)
    y_pred_proba = predicted_df["proba"].where(
        predicted_df["proba"] > 0.0, 0.0
    )
    y_pred = (y_pred_proba > prob_threshold).astype(int)

    # ROC曲線とAUCの計算
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc}")

    # PR曲線とAUCの計算
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_proba
    )
    pr_auc = average_precision_score(y_true, y_pred_proba)
    logger.info(f"PR AUC: {pr_auc}")

    # Confusion matrix を計算
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"confusion matrix: {cm}")

    # make figure
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax = ax.flatten()

    # plot true histogram
    draw_histogram(y_true, title="histogram of true", ax=ax[0])

    # plot prob histogram
    draw_histogram(y_pred_proba, title="histogram of proba", ax=ax[1])

    # ROC曲線の描画
    draw_roc_curve(fpr, tpr, roc_auc, ax=ax[2])

    # PR曲線の描画
    draw_pr_curve(recall, precision, pr_auc, ax=ax[3])

    # 混同行列をプロット
    plot_confusion_matrix(
        cm,
        classes=class_names,
        ax=ax[4],
        title=f"Confusion matrix, threshold={prob_threshold}",
    )

    # 正規化された混同行列をプロット
    plot_confusion_matrix(
        cm,
        classes=class_names,
        ax=ax[5],
        normalize=True,
        title=f"Normalized confusion matrix, threshold={prob_threshold}",
    )

    # format chart
    fig.tight_layout()

    # output fig
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir_path / "pred_scores_histogram.png")

    # cleanup
    fig.clf()
    plt.close(fig)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("input_model_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--batch_size", type=int, default=32)
@click.option(
    "--output_chart_dir", type=click.Path(), default="reports/figures/predict/"
)
def main(**kwargs):

    # init log
    logger.info("==== start process ====")
    mlflow.set_experiment("evaluate")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])

    # log cli args
    logger.info(f"cli args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load predicted data
    predicted_df = pd.read_parquet(kwargs["input_filepath"])

    # logging
    logger.info(predicted_df.describe())

    # analysis
    analysis(predicted_df, kwargs["output_chart_dir"])

    # cleanup
    mlflow.end_run()
    logger.success("==== complete process ====")


if __name__ == "__main__":
    main()
