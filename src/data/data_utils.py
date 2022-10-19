import logging
import os
import pickle
from typing import TypeAlias

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from consts import RAW_DIR, RAW_DATA_FILE, IMAGE_SIZE

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s-%(levelname)s-%(message)s", level=logging.INFO)
logger.setLevel(logging.INFO)

Loader: TypeAlias = torch.utils.data.dataloader.DataLoader


def unpickle_file(file_name: str = RAW_DATA_FILE, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(RAW_DIR, file_name), 'rb') as file:
        (x_all, y_all) = pickle.load(file)
        x_all = np.reshape(x_all, (-1, 1, image_size, image_size))
        y_all = y_all.ravel()
        logger.info("Data unpickled")
        logger.info(f"X shape: {x_all.shape}")
        logger.info(f"y shape: {y_all.shape}")
        return x_all, y_all


def prepare_dataloader(x_array: np.ndarray, y_array: np.ndarray, weighted_random_sampler: bool, batch_size: int = 32) \
        -> Loader:
    torch_x_array = torch.from_numpy(x_array).type(torch.FloatTensor)
    torch_y_array = torch.from_numpy(y_array).type(torch.LongTensor)
    tensor_dataset = torch.utils.data.TensorDataset(torch_x_array, torch_y_array)
    if weighted_random_sampler:
        class_sample_count = np.unique(y_array, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_array])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    logger.info("DataLoader prepareds")
    return data_loader


def prepare_training_data(test_size: float = 0.1) -> tuple[Loader, Loader]:
    x_all, y_all = unpickle_file()
    random_seed = 234  # passing a const int for reproducible outputs across multiple function calls
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=test_size, random_state=random_seed)
    train_loader = prepare_dataloader(x_train, y_train, weighted_random_sampler=True)
    val_loader = prepare_dataloader(x_val, y_val, weighted_random_sampler=False)
    return train_loader, val_loader


if __name__ == "__main__":
    prepare_training_data()


