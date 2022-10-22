"""Module containing helper functions."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from typing import TypeAlias

from consts import MODELS

matplotlib.style.use('ggplot')

FloatsList: TypeAlias = list[float, ...]


def save_model(epochs: int, model, optimizer, criterion, model_name: str):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(MODELS, model_name+".pth"))


def plot_confusion_matrix(confusion_matrix: np.ndarray, y_val: np.ndarray,
                          cmap=plt.cm.Blues):
    """Function to save plotted confusion matrix heatmap to disk."""
    classes = np.unique(y_val)

    plt.figure(figsize=(12, 10))
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = sns.heatmap(
        confusion_matrix,
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        cmap=cmap,
        fmt='d'
    ).get_figure()

    fig.savefig(os.path.join(MODELS, "confusion_matrix.png"))


def save_plots(train_acc: FloatsList, valid_acc: FloatsList, train_loss: FloatsList, valid_loss: FloatsList,
               train_f1_score: FloatsList, valid_f1_score: FloatsList):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(MODELS, "accuracy.png"))

    # Loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(MODELS, "loss.png"))

    # F1 score plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_f1_score, color='purple', linestyle='-',
        label='train f1 score'
    )
    plt.plot(
        valid_f1_score, color='olive', linestyle='-',
        label='validataion f1 score'
    )
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(MODELS, "f1-score.png"))
