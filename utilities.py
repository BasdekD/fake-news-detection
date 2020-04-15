import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn import metrics, model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def getVectors(X, vectorizer):
    """
    A function that takes a dataframe as an input and transform article title and article body documents into vectors,
    concatenates the two columns into one and returns the results
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


def plotHeatmap(confusion_matrix, accuracy, recall, precision, f1):
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
    plt.title('Confusion Matrix \n[Accuracy:{:.3f}, Recall:{:.3f}, Precision:{:.3f}, F1:{:.3f}]'
              .format(accuracy, recall, precision, f1), fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def crossValidation(X, y, model, params):
    """
        A function that uses cross validation with data oversampling in each split.
    """
    # We first split the data in order to have a test with data never seen by our model
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y)

    # Pipeline to apply oversampling in each split of the cross validation
    imbalance_pipeline = make_pipeline(SMOTE(random_state=42), model)

    cv = model_selection.StratifiedKFold(n_splits=10)

    # We want a multi-metric evaluation so we specify the metrics to be used
    scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

    # With GridSearchCV we try each combination of parameters given in each split of the cross validation in order to
    # get the best model. By specifying refit=f1_macro we define that the best model is to be chosen based on f-score
    evaluator = GridSearchCV(
        imbalance_pipeline,
        param_grid=params,
        cv=cv,
        scoring=scoring,
        refit="f1_macro",
        return_train_score=False)
    evaluator.fit(x_train, y_train)

    # cv_results_ is a dict with performance scores for each parameter combination in each split
    train_set_result_dict = evaluator.cv_results_

    # We convert the cv_results_ dict to dataframe for better visual representation
    train_set_result_df = pd.DataFrame.from_dict(train_set_result_dict, orient='columns')

    # Returns the best combination of parameters based on f-score as specified in refit parameter
    best_parameters = evaluator.best_params_

    # The value of the best f-score
    best_f1 = evaluator.best_score_

    # We make a prediction on a totally new test set to measure the performance of our model for completely new data
    y_test_predict = evaluator.predict(x_test)
    accuracy_test_set = accuracy_score(y_test, y_test_predict)
    f1_test_set = f1_score(y_test, y_test_predict, average='macro')
    recall_test_set = recall_score(y_test, y_test_predict, average='macro')
    precision_test_set = precision_score(y_test, y_test_predict, average='macro')
    results_on_test_set = {
        'f1': f1_test_set,
        'recall': recall_test_set,
        'precision': precision_test_set
    }

    # Results visualization as confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_predict)
    plotHeatmap(confusion_matrix, accuracy_test_set, recall_test_set, precision_test_set, f1_test_set).show()

    return train_set_result_df, best_parameters, best_f1, results_on_test_set
