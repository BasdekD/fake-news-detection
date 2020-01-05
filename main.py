from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, model_selection, neural_network, metrics
from sklearn.pipeline import make_pipeline
import utilities
import numpy as np
from time import perf_counter
t1_start = perf_counter()

'''
Project: Fake-news Detection
Phase 1: Dataset Loading
Phase 2: Data preprocessing (Lemmatization, stopword removal)
Phase 3: TF-IDF calculation
Phase 4: Make the 2 Vector (TF-IDF title & TF-IDF body) algorithm's input features
Phase 5: Use a classification algorithm
Phase 6: Evaluate performance
Phase 7: Extract new features, repeat from phase 4
'''

# ==================== Phase 1 ======================

X, y = utilities.getData('\\resources\\Fake News Dataset.xlsx')


# ==================== Phase 2 ======================
# Collect Features from dataset
NER_feature = utilities.getNERFeature(X)
POS_feature = utilities.getPOSFeature(X)
features_text = utilities.getLinguisticFeatures(X)

# Lemmatizing data
lemmatized_text = utilities.getLemmatizedText(features_text)

# Removing punctuation
punct_removed_text = utilities.removePunctuation(lemmatized_text)

# Lowercase text
#lowercase_text = utilities.lowercase(punct_removed_text)

# Removing stopwords
stopword_removed_text = utilities.removeStopwords(punct_removed_text)

# Deleting unnecessary columns. Keeping only the preprocessed data
list_columns = ["Title_stop", "Body_stop", "Sum"]
X = stopword_removed_text[list_columns]
X = X.rename(columns={"Title_stop": "Title_Parsed", "Body_stop": "Body_Parsed"})
# ==================== Phase 3 ======================

# Initializing vectorizer
vectorizer = TfidfVectorizer(encoding='unicode', strip_accents='unicode', max_df=0.8, ngram_range=(1, 2))


# ==================== Phase 4 ======================


# ==================== Phase 5 ======================
# TODO:  NB doesn't seem to handle class imbalance well. We should try something else

# Splitting the data
# x_train, x_test, y_train, y_test = model_selection.train_test_split(X['Body_Parsed'], y, random_state=42, stratify=y)
# alpha = 0.1
# model = make_pipeline(vectorizer, naive_bayes.MultinomialNB(alpha=alpha))
# model.fit(x_train, y_train)
# y_predicted = model.predict(x_test)

# Perform TF-IDF in title and body and merge them with the other features

# vectorbody = vectorizer.fit_transform(X['Body_Parsed'])
# vector_body = vectorbody.toarray()
# vectortitle = vectorizer.fit_transform(X['Title_Parsed'])
# vector_title = vectortitle.toarray()
# vector_tfidf = np.concatenate((vector_body, vector_title), axis=1)
# features = X.as_matrix(columns=['Sum'])
# full_features = np.concatenate((vector_tfidf, features), axis=1)
#
# # Splitting the data
# x_train, x_test, y_train, y_test = model_selection.train_test_split(full_features, y, random_state=42, stratify=y)
# alpha = 0.1
# model = naive_bayes.MultinomialNB(alpha=alpha)
# model = neural_network.MLPClassifier(hidden_layer_sizes=20, activation='relu', solver='adam', tol=0.0001, max_iter=100, alpha=alpha)

# model.fit(x_train, y_train)
# y_predicted = model.predict(x_test)
#
# # ==================== Phase 6 ======================
#
# accuracy, recall, precision, f1 = utilities.getMetrics(y_test, y_predicted)
# print("Accuracy: %f" % accuracy)
# print("Recall: %f" % recall)
# print("Precision: %f" % precision)
# print("F1: %f" % f1)
#
# # Plot confusion matrix
# confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
# utilities.plotHeatmap(confusion_matrix, alpha, accuracy, recall, precision, f1).show()
#
# # ==================== Phase 7 ======================
# # TODO:  What other features could we use?
# # Number of Entities mentioned?
# # Numbers of each POS appearances?
#
# # Calculate execution time
# t1_stop = perf_counter()
# print("Elapsed time:", t1_stop-t1_start)
