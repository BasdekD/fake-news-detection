import os
import pandas as pd
import spacy as sp
from nltk.corpus import stopwords


def getData(fileName):
    """
    A function to load the dataset from an excel file
    """
    curr_dir = os.getcwd()
    data = pd.read_excel(curr_dir + fileName)
    X = data.iloc[:, :2]
    y = data.iloc[:, -1]
    return X, y


def getLemmatizedText(df):
    """
    A function to Lemmatize the text of a given dataframe
    """
    num_of_docs = len(df)
    lemmatized_title = []
    lemmatized_body = []
    nlp = sp.load("el_core_news_sm")

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
    df = dataframe
    df['Title_punct'] = df['Title_lem'].str.replace('[^\w\s]', '')
    df['Body_punct'] = df['Body_lem'].str.replace('[^\w\s]', '')
    return df

def lowercase(dataframe):
    df = dataframe
    df['Title_lower'] = df['Title_punct'].str.lower()
    df['Body_lower'] = df['Body_punct'].str.lower()
    return df

def removeStopwords(lemmatized_text):
    """
    A function to remove the stopwords of the text in a dataframe
    """
    nlp = sp.load("el_core_news_sm")
    stop_words = list(nlp.Defaults.stop_words)
    #stop_words1 = list(stopwords.words('greek'))
    df = lemmatized_text
    df['Title_stop'] = df['Title_lower']
    df['Body_stop'] = df['Body_lower']

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Title_stop'] = df['Title_stop'].str.\
            replace(regex_stopword, '', case=False, regex=True)
        df['Body_stop'] = df['Body_stop'].str.\
            replace(regex_stopword, '', case=False, regex=True)
    return df
