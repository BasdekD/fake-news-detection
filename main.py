from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, model_selection, neural_network, metrics
import utilities
import numpy as np
from time import perf_counter
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.utils import resample



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

# entities = ['GPE']  # 'GPE', 'LOC', 'ORG', 'PERSON', 'PRODUCT'  # The NER did not improve the model. No useful feature
# NER_feature = utilities.getNERFeature(X, entities)

pos_tags = ['ADP']  # All POS tags: 'ADJ', 'ADP', 'ADV', 'NOUN', 'PROPN', 'VERB'
POS_feature = utilities.getPOSFeature(X, pos_tags)
punct_feature = utilities.getLinguisticFeatures(X)


# ============================================================================ #
# ========================== Data Preprocessing ============================== #

# Lemmatizing data
lemmatized_text = utilities.getLemmatizedText(punct_feature)

# Removing punctuation
punct_removed_text = utilities.removePunctuation(lemmatized_text)

# Removing stopwords
stopword_removed_text = utilities.removeStopwords(punct_removed_text)

# Deleting unnecessary columns. Keeping only the preprocessed data
list_columns = ["Title_stop", "Body_stop", "Sum"]
X = stopword_removed_text[list_columns]
X = X.rename(columns={"Title_stop": "Title_Parsed", "Body_stop": "Body_Parsed"})


# ============================================================================ #
# ========================== TF-IDF Extraction =============================== #

# Initializing vectorizer
vectorizer = TfidfVectorizer(encoding='unicode', strip_accents='unicode', max_df=0.8, ngram_range=(1, 2))

# Perform TF-IDF in title and body and merge them with the other features
vectorbody = vectorizer.fit_transform(X['Body_Parsed'])
vector_body = vectorbody.toarray()
vectortitle = vectorizer.fit_transform(X['Title_Parsed'])
vector_title = vectortitle.toarray()
vector_tfidf = np.concatenate((vector_title, vector_body), axis=1)

# ============================================================================ #
# ========================== Gathering Features ============================== #

punct_features = X.as_matrix(columns=['Sum'])
full_features = np.concatenate((vector_tfidf, punct_features, POS_feature), axis=1)

# Converting features to a dataframe for easier processing during oversampling
full_features = pd.DataFrame(data=full_features)

# ============================================================================ #
# ========================== Splitting Data ======================== #

# Splitting the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(full_features, y, random_state=42, stratify=y)


# ============================================================================ #
# ========================== Dealing With Imbalance ========================== #


# ========================== Method 1: Synthetic Data ========================= #

# sm = SMOTE(random_state=42)
# x_train, y_train = sm.fit_sample(x_train, y_train)


# ========================== Method 2: Oversampling ========================== #

# concatenate our training data back together
X = pd.concat([x_train, y_train], axis=1)

# Seperating minority and majority class
legit = X[X.Label == 'legit']
fake = X[X.Label == 'fake']

# Oversample minority class
fake_oversampled = resample(fake,
                            replace=True,           # sample with replacement
                            n_samples=len(legit),   # match number of majority class
                            random_state=42)        # reproducible results

# Combining majority and oversampled minority
oversampled = pd.concat([legit, fake_oversampled])

# Dividing the train set once more
y_train = oversampled.Label
x_train = oversampled.drop('Label', axis=1)


# ========================== Method 3: Undersampling ========================= #

# TODO


# ============================================================================ #
# ========================== Training ML Model =============================== #

# ========================== Multinomial NB ================================== #

alpha = 0.1
model = naive_bayes.MultinomialNB(alpha=alpha)

# ========================== Neural Networks ================================== #

# model = neural_network.MLPClassifier(
#     hidden_layer_sizes=20, activation='relu', solver='adam', tol=0.0001, max_iter=100, alpha=alpha
# )


# ============================================================================ #
# ========================== Fitting Model =================================== #

model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


# ============================================================================ #
# ========================== Model Evaluation ================================ #

accuracy, recall, precision, f1 = utilities.getMetrics(y_test, y_predicted)
print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)

# Plot confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
utilities.plotHeatmap(confusion_matrix, alpha, accuracy, recall, precision, f1).show()


# TODO:  What other features could we use?
# Number of Entities mentioned?
# Numbers of each POS appearances?

# Calculate execution time
t1_stop = perf_counter()
print("Elapsed time:", t1_stop-t1_start)
