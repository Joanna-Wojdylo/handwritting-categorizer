"""A module for training a model on training data and checking it on validation data."""
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

from src.data.data_utils import prepare_training_data, Loader
from src.models.cnn_model import CNN
from src.models.utils import save_model, save_plots, plot_confusion_matrix

logger = logging.getLogger()
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())


# Training function.
def train(model, trainloader: Loader, optimizer, criterion) -> tuple[float, float, float]:
    """
    A function used to train the model, the model error is calculated,
     which is then propagated backwards to update the model weights.
    :param model:
    :param trainloader: DataLoader with training data
    :param optimizer: Optimizer, SGD.
    :param criterion: Optimization criterion, Cross Entropy.
    :return:
    """
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    y_true = []
    y_pred = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_f1_score = f1_score(y_true, y_pred, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1_score


# Validation function.
def validate(model, valloader, criterion) -> tuple[float, float, float, list]:
    """
    A function used for model validation, model accuracy and error function are calculated.
    :param model:
    :param valloader: DataLoader for validation set
    :param criterion: Optimization criterion, Cross Entropy
    :return:
    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(valloader), total=len(valloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            # Store val_predictions for confusion matrix calculations and f1-score
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valloader.dataset))
    epoch_f1_score = f1_score(y_true, y_pred, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1_score, y_pred


if __name__ == '__main__':
    # Load the training and validation data loaders.
    x_train, x_val, y_train, y_val, train_loader, valid_loader = prepare_training_data(test_size=0.1)

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    model = CNN().to(device)
    logger.info(f"CNN model: {model}")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_f1_score, valid_f1_score = [], []
    valid_preds = []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc, train_epoch_f1_score = train(model, train_loader,
                                                                        optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc, valid_epoch_f1_score, valid_epoch_preds = validate(model, valid_loader,
                                                                                              criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        valid_preds.append(valid_epoch_preds)
        train_f1_score.append(train_epoch_f1_score)
        valid_f1_score.append(valid_epoch_f1_score)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}, training f1-score: {train_epoch_f1_score:.3f}")
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}, validation f1-score: {train_epoch_f1_score:.3f}")
        print('-' * 50)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, model_name="basic_cnn")
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, train_f1_score, valid_f1_score)
    # Plot confusion matrix
    y_pred_classes = np.array(valid_preds[-1])
    cm = confusion_matrix(y_val, y_pred_classes)
    plot_confusion_matrix(cm, y_val)
    print('TRAINING COMPLETE')
