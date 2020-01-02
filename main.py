from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, metrics, model_selection
from sklearn.pipeline import make_pipeline

import utilities


'''
Project: Fake-news Detection
Phase 1: Dataset Loading
Phase 2: TF-IDF calculation
Phase 3: Make the 2 Vector (TF-IDF title & TF-IDF body) algorithm's input features
Phase 4: Use a classification algorithm
Phase 5: Evaluate performance
Phase 6: Extract new features, repeat from phase 4
'''

# ==================== Phase 1 ======================

X, y = utilities.getData('\\resources\\Fake News Dataset.xlsx')


# ==================== Phase 2 ======================
# TODO: Is this step necessary?
# vectorizer = TfidfVectorizer(encoding='unicode', strip_accents='unicode', max_df=0.8)
#
# title_vectorized = vectorizer.fit_transform(article_title)
# body_vectorized = vectorizer.fit_transform(article_body)


# ==================== Phase 3 ======================
# TODO:  Make the 2 Vector (TF-IDF title & TF-IDF body) algorithm's input features (is this step necessary?)


# ==================== Phase 4 ======================
# TODO:  Use a classification algorithm
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0, stratify=y)
alpha = 0.1
model = make_pipeline(TfidfVectorizer(), naive_bayes.MultinomialNB(alpha=alpha))
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


# ==================== Phase 5 ======================
# TODO:  Evaluate performance
accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average='macro')
precision = metrics.precision_score(y_test, y_predicted, average='macro')
f1 = metrics.f1_score(y_test, y_predicted, average='macro')

print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)


# ==================== Phase 6 ======================
# TODO:  What other features could we use?

#"Comment"
