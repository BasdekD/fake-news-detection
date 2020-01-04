import os
import pandas as pd
import spacy as sp
import re
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

def collectFeatures(df):
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
    """
       A function for removing punctuation
    """
    df = dataframe
    df['Title_punct'] = df['Title_lem'].str.replace('[^\w\s]', '')
    df['Body_punct'] = df['Body_lem'].str.replace('[^\w\s]', '')
    return df

def lowercase(dataframe):
    """
       A function for converting text to lowercase
    """
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
