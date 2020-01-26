import pandas as pd
import conf


def getData():
    """
    A function to load the dataset from an excel file
    """
    data = pd.read_excel(conf.CURR_DIR + conf.DATASET)
    X = data.iloc[:, :2]
    y = data.iloc[:, -1]
    return X, y


