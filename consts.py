"""Module containing constatns used in the project"""
import os

# project structure
DATA_DIR = "data"
RAW = "raw"
PROCESSED = "processed"

RAW_DIR = os.path.join(DATA_DIR, RAW)
PROCESSED_DIR = os.path.join(DATA_DIR, PROCESSED)

SRC = "src"
MODELS = "models"

# data file name
RAW_DATA_FILE = "train.pkl"

IMAGE_SIZE = 56
