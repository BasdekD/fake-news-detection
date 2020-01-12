import utilities
import featureExtraction
import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, model_selection, neural_network, metrics, tree
import numpy as np
from time import perf_counter
import pandas as pd

t1_start = perf_counter()

'''
Project: Fake-news Detection
Phase 1: Loading Dataset
Phase 2: Feature Extraction (Lemmatization, punctuation removal, stopword removal)
Phase 3: Data Prep
Phase 4: Make the 2 Vector (TF-IDF title & TF-IDF body) algorithm's input features
Phase 5: Use a classification algorithm
Phase 6: Evaluate performance
Phase 7: Extract new features, repeat from phase 4
'''

# ============================================================================ #
# ========================== Loading Dataset ================================= #

X, y = utilities.getData('\\resources\\Fake News Dataset.xlsx')


# ============================================================================ #
# ========================== Feature Extraction ============================== #

entities = ['GPE']        # All Entities: 'GPE', 'LOC', 'ORG', 'PERSON', 'PRODUCT'
NER_feature = featureExtraction.getNERFeature(X, entities)

# pos_tags = ['ADP', 'PROPN']          # All POS tags: 'ADJ', 'ADP', 'ADV', 'NOUN', 'PROPN', 'VERB'
# POS_feature = featureExtraction.getPOSFeature(X, pos_tags)
# punctuation_feature = featureExtraction.getPunctuationFeatures(X)


# ============================================================================ #
# ========================== Data Preprocessing ============================== #

# Lemmatizing data
lemmatized_text = preprocessing.getLemmatizedText(X)

# Removing punctuation
punctuation_removed_text = preprocessing.removePunctuation(lemmatized_text)

# Removing stopwords
stopword_removed_text = preprocessing.removeStopwords(punctuation_removed_text)

# Deleting unnecessary columns. Keeping only the preprocessed data
# list_columns = ["Title_stop", "Body_stop", "Sum"]
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

# punctuation_features = X.as_matrix(columns=['Sum'])
full_features = np.concatenate((vectors, NER_feature), axis=1)

# Converting features to a dataframe for easier processing during oversampling
full_features = pd.DataFrame(data=full_features)

# ============================================================================ #
# ========================== Training ML Model =============================== #

# ========================== Multinomial NB ================================== #

alpha = 0.1
model = naive_bayes.MultinomialNB(alpha=alpha)

# ========================== Neural Networks ================================== #

# model = neural_network.MLPClassifier(
#     hidden_layer_sizes=20, activation='relu', solver='adam', tol=0.0001, max_iter=100, alpha=alpha
# )

# ========================== Decision Tree ==================================== #

# max_depth = 8
# model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)

# ============================================================================ #
# ========================== Model Evaluation ================================ #

# ========================== Cross Validation ================================ #

results = utilities.crossValidation(full_features, y, model)
print(results)


# ============================================================================ #
# ========================== Results Visualization =========================== #

# Print Metrics
# utilities.printMetrics(accuracy, recall, precision, f1)

# Plot confusion matrix
# confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
# utilities.plotHeatmap(confusion_matrix, accuracy, recall, precision, f1).show()


# TODO:  What other features could we use?
# Number of Entities mentioned?
# Numbers of each POS appearances?
# Average sentence length?
# Average word length?

# Calculate execution time
t1_stop = perf_counter()
print("Elapsed time:", t1_stop-t1_start)
