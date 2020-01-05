import os
import pandas as pd
import spacy as sp
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

nlp = sp.load("el_core_news_sm")


def getData(fileName):
    """
    A function to load the dataset from an excel file
    """
    curr_dir = os.getcwd()
    data = pd.read_excel(curr_dir + fileName)
    X = data.iloc[:, :2]
    y = data.iloc[:, -1]
    return X, y


def getLinguisticFeatures(df):
    """
       A function for collecting linguistic features
    """

    n = len(df)
    # Collect the number of words
    result1 = []
    for i in range(0, n):
        result1.append(len(re.findall(r'\S+', df['Title'][i])))
    df['Number_of_words'] = result1

    # Collect the number uppercase words bigger or equal to 5 letters
    uppercase = []
    for i in range(0, n):
        result = re.findall(r'[Α-Ω]{5,}', df['Title'][i])
        uppercase.append(len(result)/result1[i])
    df['Uppercase_words'] = uppercase

    # Collect the number of Ellispis(Αποσιωπητικά)
    ellipsis =[]
    for i in range(0, n):
        result = re.findall(r'…', df['Title'][i])
        ellipsis.append(len(result)/result1[i])
    df['Ellipsis'] = ellipsis

    # Collect the number of Guillemets(Εισαγωγικά)
    guillements = []
    for i in range(0, n):
        result = re.findall(r'«.+?»', df['Title'][i])
        guillements.append(len(result)/result1[i])
    df['Guillements'] = guillements

    # Collect the number of Exclamation Marks
    exclamation = []
    for i in range(0, n):
        result = re.findall(r'!', df['Title'][i])
        exclamation.append(len(result)/result1[i])
    df['Exclamation Marks'] = exclamation

    sum = []
    for i in range(0, n):
        sum.append(df['Exclamation Marks'][i] + df['Ellipsis'][i])
    df['Sum'] = sum
    return df


def getNERFeature(df):
    """
    A function that:
    1. Counts the number of occurrences  of each of the entity categories: GPE, LOC, ORG, PERSON, PRODUCT
    for the title and the body of each article in the dataset
    2. Normalizes the values occurred by dividing with the total number of entities found in the article's title or body
    3. Concatenates the title and body results into a dataframe. Each row of a dataframe represents an article of the
    dataset and the columns are title's entity count followed by body's entity count for each entity category
    4. Returns the dataframe
    """
    num_of_docs = len(df)
    NER_all_titles = []
    NER_all_bodies = []

    for doc_num in range(0, num_of_docs):
        doc_title = nlp(df['Title'][doc_num])
        doc_body = nlp(df['Body'][doc_num])

        # The entity categories we will take into account
        NER_title = {'GPE': 0, 'LOC': 0, 'ORG': 0, 'PERSON': 0, 'PRODUCT': 0}
        NER_body = {'GPE': 0, 'LOC': 0, 'ORG': 0, 'PERSON': 0, 'PRODUCT': 0}
        entity_counter_title = 0
        entity_counter_body = 0

        # Counting the entities in the title by their category
        for ent in doc_title.ents:
            for key in NER_title.keys():
                if ent.label_ == key:
                    entity_counter_title += 1
                    NER_title[key] += 1
        for key in NER_title.keys():
            if entity_counter_title == 0:
                NER_title[key] = 0
            else:
                NER_title[key] /= entity_counter_title
        NER_all_titles.append(list(NER_title.values()))

        # Counting the entities in the body by their category
        for ent in doc_body.ents:
            for key in NER_body.keys():
                if ent.label_ == key:
                    entity_counter_body += 1
                    NER_body[key] += 1
        for key in NER_body.keys():
            if entity_counter_body == 0:
                NER_body[key] = 0
            else:
                NER_body[key] /= entity_counter_body
        NER_all_bodies.append(list(NER_body.values()))

    NER_all_titles = pd.DataFrame(NER_all_titles)
    NER_all_bodies = pd.DataFrame(NER_all_bodies)
    dataframes = [NER_all_titles, NER_all_bodies]
    df = pd.concat(dataframes, axis=1)
    return df


def getPOSFeature(df):
    """
    A function that:
    1. Counts the number of occurrences  of each of the POS categories: ADJ, ADP, ADV, NOUN, PROPN, VERB
    for the title and the body of each article in the dataset
    2. Normalizes the values occurred by dividing with the total number of words in the article's title or body
    3. Concatenates the title and body results into a dataframe. Each row of a dataframe represents an article of the
    dataset and the columns are title's POS count followed by body's POS count for each POS category
    4. Returns the dataframe
    """
    num_of_docs = len(df)
    POS_all_titles = []
    POS_all_bodies = []

    for doc_num in range(0, num_of_docs):
        doc_title = nlp(df['Title'][doc_num])
        doc_body = nlp(df['Body'][doc_num])

        # The POS categories we will take into account
        POS_title = {'ADJ': 0, 'ADP': 0, 'ADV': 0, 'NOUN': 0, 'PROPN': 0, 'VERB': 0}
        POS_body = {'ADJ': 0, 'ADP': 0, 'ADV': 0, 'NOUN': 0, 'PROPN': 0, 'VERB': 0}
        token_counter_title = 0
        token_counter_body = 0

        # Counting the POS in the title by their category
        for token in doc_title:
            for key in POS_title.keys():
                if token.pos_ == key:
                    token_counter_title += 1
                    POS_title[key] += 1
        for key in POS_title.keys():
            POS_title[key] /= token_counter_title
        POS_all_titles.append(list(POS_title.values()))

        # Counting the POS in the body by their category
        for token in doc_body:
            for key in POS_body.keys():
                if token.pos_ == key:
                    token_counter_body += 1
                    POS_body[key] += 1
        for key in POS_body.keys():
            POS_body[key] /= token_counter_body
        POS_all_bodies.append(list(POS_body.values()))

    POS_all_titles = pd.DataFrame(POS_all_titles)
    POS_all_bodies = pd.DataFrame(POS_all_bodies)
    dataframes = [POS_all_titles, POS_all_bodies]
    df = pd.concat(dataframes, axis=1)
    return df




def getLemmatizedText(df):
    """
    A function to Lemmatize the text of a given dataframe
    """
    num_of_docs = len(df)
    lemmatized_title = []
    lemmatized_body = []

    for doc_num in range(0, num_of_docs):
        doc_title = nlp(df['Title'][doc_num])
        doc_body = nlp(df['Body'][doc_num])

        # Lemmatizing Document's Title
        lemmatized_list_title = []
        for token in doc_title:
            lemmatized_list_title.append(token.lemma_)
        lemmatized_text_title = " ".join(lemmatized_list_title)
        lemmatized_title.append(lemmatized_text_title)

        # Lemmatizing Document's Body
        lemmatized_list_body = []
        for token in doc_body:
            lemmatized_list_body.append(token.lemma_)
        lemmatized_text_body = " ".join(lemmatized_list_body)
        lemmatized_body.append(lemmatized_text_body)

    # Appending new lemmatized columns
    df['Title_lem'] = lemmatized_title
    df['Body_lem'] = lemmatized_body

    return df


def removePunctuation(dataframe):
    """
       A function for removing punctuation
    """
    df = dataframe
    df['Title_punct'] = df['Title_lem'].str.replace('[^\w\s]', '')
    df['Body_punct'] = df['Body_lem'].str.replace('[^\w\s]', '')
    return df


def removeStopwords(lemmatized_text):
    """
    A function to remove the stopwords of the text in a dataframe
    """
    nlp = sp.load("el_core_news_sm")
    stop_words = list(nlp.Defaults.stop_words)
    df = lemmatized_text
    df['Title_stop'] = df['Title_punct']
    df['Body_stop'] = df['Body_punct']

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Title_stop'] = df['Title_stop'].str.\
            replace(regex_stopword, '', case=False, regex=True)
        df['Body_stop'] = df['Body_stop'].str.\
            replace(regex_stopword, '', case=False, regex=True)
    return df


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


def getMetrics(y_test, y_predicted, avg="macro"):
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average=avg)
    precision = metrics.precision_score(y_test, y_predicted, average=avg)
    f1 = metrics.f1_score(y_test, y_predicted, average=avg)

    return accuracy, recall, precision, f1

