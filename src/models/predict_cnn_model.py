import os
import pickle

import numpy as np
import torch

from consts import MODELS
from consts import RAW_DIR, IMAGE_SIZE
from src.models.cnn_model import CNN

model_name = "basic_cnn"

# this is just an example to simulate some data with appropriate shape, this is NOT the actual test data
with open(os.path.join(RAW_DIR, "train.pkl"), 'rb') as file:
    (x_all, y_all) = pickle.load(file)
input_data_test = x_all[-4000:]


def predict(input_data: np.ndarray, image_size:int = IMAGE_SIZE):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

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


if __name__ == "__main__":
    predict(input_data_test)
