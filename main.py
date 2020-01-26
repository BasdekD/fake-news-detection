import utilities
import feature_extraction
import text_preprocessing
import file_handling
import conf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, model_selection, neural_network, tree, svm, ensemble
import numpy as np
from sklearn import preprocessing
from time import perf_counter
import pandas as pd

'''
Project: Fake-news Detection
Phase 1: Loading Dataset
Phase 2: Feature Extraction 
Phase 3: Data Preprocessing (Lemmatization, punctuation removal, stopword removal)
Phase 4: Representation of textual content as vectors (TF-IDF title & TF-IDF body)
Phase 5: Machine Learning Models
Phase 6: Evaluate performance
Phase 7: Results visualization
'''

# ============================================================================ #
# ========================== Loading Dataset ================================= #

# X = data, y = data labels
X, y = file_handling.getData()

# ============================================================================ #
# ========================== Feature Extraction ============================== #

# If the pickle file with the features exists, the preprocessing and feature extraction phases are skipped for better
# performance. Else, at the end of these phases the FEATURES_FILE is created for further utilization

if not conf.FEATURES_FILE.exists():
    t1_start = perf_counter()

    # NER feature extraction
    entities = ['GPE']       # All Entities: 'GPE', 'LOC', 'ORG', 'PERSON', 'PRODUCT'
    NER_feature = feature_extraction.getNERFeature(X, entities)

    # POS feature extraction
    pos_tags = ['ADP', 'PROPN']   # All POS tags: 'ADJ', 'ADP', 'ADV', 'NOUN', 'PROPN', 'VERB'
    POS_feature = feature_extraction.getPOSFeature(X, pos_tags)

    # Punctuation feature extraction
    punctuation_features = feature_extraction.getPunctuationFeatures(X)

    # Psycholinguistic feature extraction (Negative impact on final results, uncomment to use)
    # affin_features = feature_extraction.psycholinguistic(X)

# ============================================================================ #
# ========================== Data Preprocessing ============================== #

    # Lemmatizing data
    lemmatized_text = text_preprocessing.getLemmatizedText(X)

    # Converting to lowercase
    lowercase_text = text_preprocessing.lowercase(lemmatized_text)

    # Removing punctuation
    punctuation_removed_text = text_preprocessing.removePunctuation(lowercase_text)

    # Removing stopwords
    stopword_removed_text = text_preprocessing.removeStopwords(punctuation_removed_text)

    # Deleting unnecessary columns. Keeping only the preprocessed data
    list_columns = ["Title_stop", "Body_stop"]
    X = stopword_removed_text[list_columns]
    X = X.rename(columns={"Title_stop": "Title_Parsed", "Body_stop": "Body_Parsed"})


# ============================================================================ #
# ========================== TF-IDF Extraction =============================== #

    # Initializing vectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', max_df=0.8, ngram_range=(1, 4))

    # Get TF-IDF of title and body and merge them
    vectors = utilities.getVectors(X, vectorizer)

# ============================================================================ #
# ========================== Collecting Features ============================= #

    punctuation_features = punctuation_features.as_matrix(columns=['exclamation_title', 'ellipsis_title'])
    full_features = np.concatenate((vectors, NER_feature, punctuation_features, POS_feature), axis=1)

    # Converting features to a dataframe for easier processing during oversampling
    full_features = pd.DataFrame(data=full_features)
    full_features.to_pickle(conf.FEATURES_FILE)

# Writing final form of features to a file for better performance
elif conf.FEATURES_FILE.exists():
    t1_start = perf_counter()
    full_features = pd.read_pickle(conf.FEATURES_FILE)


# ============================================================================ #
# ========================== Feature Optimization ============================ #

# Negative impact on final results, uncomment to use

# minMaxScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# full_features = minMaxScaler.fit_transform(full_features)
# full_features = pd.DataFrame(data=full_features)
# selected_features = SelectKBest(chi2, k=50000).fit_transform(full_features, y)

# ============================================================================ #
# ========================== Training ML Model =============================== #

# ========================== Multinomial NB ================================== #

params = {
    'multinomialnb__alpha': [0.15]
}
model = naive_bayes.MultinomialNB()

# ========================== Neural Networks ================================== #

# params = {
#     'mlpclassifier__hidden_layer_sizes': [(50, 50, 50)],
#     'mlpclassifier__activation': ['relu'],
#     'mlpclassifier__solver': ['lbfgs'],
#     'mlpclassifier__alpha': [0.0001],
#     'mlpclassifier__max_iter': [300],
#     'mlpclassifier__random_state': [42],
#     'mlpclassifier__tol': [0.0001]
#
# }
# model = neural_network.MLPClassifier()

# ========================== Decision Tree ==================================== #

# params = {
#     'decisiontreeclassifier__max_depth': [1],
#     'decisiontreeclassifier__criterion': ['gini']
#
# }
# model = tree.DecisionTreeClassifier()

# ========================== Random Forest ==================================== #

# params = {
#     "randomforestclassifier__criterion": ["entropy"],
#     "randomforestclassifier__n_estimators": [5],
#     "randomforestclassifier__max_depth": [3],
#     "randomforestclassifier__max_features": [None]
# }
# model = ensemble.RandomForestClassifier()

# =================== Support Vector Machines (SVM)=========================== #

# minMaxScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# full_features = minMaxScaler.fit_transform(full_features)
# params = {
#     'svc__kernel': ['sigmoid'],
#     'svc__C': [0.1],
#     'svc__gamma': [0.5]
# }
# model = svm.SVC()

# ========================== Gradient Boosting ================================== #

# params = {}
# model = ensemble.GradientBoostingClassifier()

# ============================================================================ #
# ========================== Model Evaluation ================================ #

# ========================== Cross Validation ================================ #

train_set_result, best_parameters, best_f1, results_on_test_set = utilities.crossValidation(
    full_features, y, model, params)


# ============================================================================ #
# ========================== Results Visualization =========================== #

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("The metrics of each split are:\n{}".format(train_set_result))
print("The results on a test set unknown to the model: {}".format(results_on_test_set))
print("Best f-score achieved: {}".format(best_f1))
print("Parameter set with the best performance: {}".format(best_parameters))

# Calculate execution time
t1_stop = perf_counter()
print("Elapsed time:", t1_stop-t1_start)
