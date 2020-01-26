import pathlib
import os


# In this file necessary global variables for the application are declared

CURR_DIR = os.getcwd() + "\\resources\\"

FEATURES_FILE = pathlib.Path(CURR_DIR + "features.pkl")

DATASET = "Fake News Dataset.xlsx"
