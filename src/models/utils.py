import itertools

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

import os
import seaborn as sns
from consts import MODELS

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, model_name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(MODELS, model_name+".pth"))


def plot_confusion_matrix(confusion_matrix, y_val,
                          cmap=plt.cm.Blues):
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


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
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
