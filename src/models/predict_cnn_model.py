"""A module containing the functions needed to infer the model on previously unknown data."""
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from consts import MODELS
from consts import RAW_DIR, IMAGE_SIZE
from src.models.cnn_model import CNN
from src.models.utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns


# This is just an example to simulate some data with appropriate shape,
# this is NOT the actual test data
with open(os.path.join(RAW_DIR, "train.pkl"), 'rb') as file:
    (x_all, y_all) = pickle.load(file)
input_data_test = x_all[-4000:]


def predict(model_name: str, input_data: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Function used to predict labels for given data.
    :param model_name: name of the .pth file with the trained model, stored in data/raw location
    :param input_data: Nx3136 numpy array of test examples
    :param image_size: size of the input images
    :return: Nx1 numpy array with the labels predicted by the model
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    input_data = np.reshape(input_data, (-1, 1, image_size, image_size))
    torch_input_data = torch.from_numpy(input_data).type(torch.FloatTensor)
    torch_input_data = torch_input_data.to(device)

    checkpoint = torch.load(os.path.join(MODELS, model_name+".pth"))
    model = CNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        prediction = model(torch_input_data)
        _, predicted = torch.max(prediction.data, 1)
    predicted = predicted.cpu().numpy().reshape(-1, 1)
    return predicted


def predict_and_test(model_name: str, input_data: np.ndarray, labels_to_check: np.ndarray, image_size: int = IMAGE_SIZE):
    predicted_labels = predict(model_name=model_name, input_data=input_data, image_size=image_size)
    predicted_correctly = (predicted_labels == labels_to_check).sum().item()
    prediction_acc = 100. * (predicted_correctly / len(labels_to_check))
    print(f"Test accuracy: {prediction_acc}%")
    prediction_f1_score = f1_score(predicted_labels, labels_to_check, average="weighted")
    print(f"Test F1-score: {prediction_f1_score}%")
    cm = confusion_matrix(predicted_labels, labels_to_check)
    classes = np.unique(labels_to_check)

    plt.figure(figsize=(12, 10))
    plt.title('Confusion matrix')
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        fmt='d'
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    # predict(model_name="basic_cnn", input_data=input_data_test)
    predict_and_test(model_name="basic_cnn", input_data=input_data_test, labels_to_check=y_all[-4000:])
