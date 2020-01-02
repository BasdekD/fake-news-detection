from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, metrics, model_selection
from sklearn.pipeline import make_pipeline
import utilities


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

# Lemmatizing data
lemmatized_doc = utilities.getLemmatizedText(X)

# Removing stopwords TODO: Needs optimization (some stop words not included in the vocabulary. Could be added)
stopword_removed_text = utilities.removeStopwords(lemmatized_doc)

# Deleting unnecessary columns. Keeping only the preprocessed data
list_columns = ["Title_stop", "Body_stop"]
X = stopword_removed_text[list_columns]
X = X.rename(columns={"Title_stop": "Title_Parsed", "Body_stop": "Body_Parsed"})

# ==================== Phase 3 ======================

# Initializing vectorizer
vectorizer = TfidfVectorizer(encoding='unicode', strip_accents='unicode', max_df=0.8, ngram_range=(1, 2))


# ==================== Phase 4 ======================
# TODO:  Concat Title and Body (Maybe it will lead to better results). Currently we are considering only the Body


# ==================== Phase 5 ======================
# TODO:  NB doesn't seem to handle class imbalance well. We should try something else

# Splitting the data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X['Body_Parsed'], y, random_state=42, stratify=y)
alpha = 0.1
model = make_pipeline(vectorizer, naive_bayes.MultinomialNB(alpha=alpha))
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


# ==================== Phase 6 ======================


accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average='macro')
precision = metrics.precision_score(y_test, y_predicted, average='macro')
f1 = metrics.f1_score(y_test, y_predicted, average='macro')

print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)


# ==================== Phase 7 ======================
# TODO:  What other features could we use?
# Number of Entities mentioned?
# Numbers of each POS appearances?
