import os
import pandas as pd


def getData(fileName):
    curr_dir = os.getcwd()
    data = pd.read_excel(curr_dir + fileName)
    X = data.iloc[:, 1]
    y = data.iloc[:, -1]
    return X, y