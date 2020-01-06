import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def getData(fileName):
    """
    A function to load the dataset from an excel file
    """
    curr_dir = os.getcwd()
    data = pd.read_excel(curr_dir + fileName)
    X = data.iloc[:, :2]
    y = data.iloc[:, -1]
    return X, y


def getVectors(X, vectorizer):
    """
    A function that takes a dataframe as an input and transform article title and article body documents into vectors,
    concatenates the two columns into one and returns the two columns
    """
    vectorbody = vectorizer.fit_transform(X['Body_Parsed'])
    vector_body = vectorbody.toarray()
    vectortitle = vectorizer.fit_transform(X['Title_Parsed'])
    vector_title = vectortitle.toarray()
    vector_tfidf = np.concatenate((vector_title, vector_body), axis=1)
    return vector_tfidf


def getSyntheticData(x_train, y_train, random_state):
    """
    A function that deals with class imbalance by creating synthetic data of the minority class
    """
    sm = SMOTE(random_state=random_state)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    return x_train, y_train


def getOversampledData(x_train, y_train):
    """
    A function that deals with class imbalance by oversampling the minority class
    """

    # concatenate our training data back together
    X = pd.concat([x_train, y_train], axis=1)

    # Separating minority and majority class
    legit = X[X.Label == 'legit']
    fake = X[X.Label == 'fake']

    # Oversample minority class
    fake_oversampled = resample(fake,
                                replace=True,  # sample with replacement
                                n_samples=len(legit),  # match number of majority class
                                random_state=42)  # reproducible results

    # Combining majority and oversampled minority
    oversampled = pd.concat([legit, fake_oversampled])

    # Dividing the train set once more
    y_train = oversampled.Label
    x_train = oversampled.drop('Label', axis=1)

    return x_train, y_train


def getMetrics(y_test, y_predicted, avg="macro"):
    """
    A function that returns the accuracy, recall, precision and f1 metrics of an algorithm's implementation
    """
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average=avg)
    precision = metrics.precision_score(y_test, y_predicted, average=avg)
    f1 = metrics.f1_score(y_test, y_predicted, average=avg)

    return accuracy, recall, precision, f1


def printMetrics(accuracy, recall, precision, f1):
    """
    A function that prints given metrics in a nice format
    """
    print("Accuracy: %f" % accuracy)
    print("Recall: %f" % recall)
    print("Precision: %f" % precision)
    print("F1: %f" % f1)


def plotHeatmap(confusion_matrix, alpha, accuracy, recall, precision, f1):
    """
    A function for plotting the confusion matrix
    """
    class_names = ['fake', 'legit']
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    bottom, top = heatmap.get_ylim()
    heatmap.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix \n[a={:.1f}] [Accuracy:{:.3f}, Recall:{:.3f}, Precision:{:.3f}, F1:{:.3f}]'
              .format(alpha, accuracy, recall, precision, f1), fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig



