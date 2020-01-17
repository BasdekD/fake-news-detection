import re
import spacy as sp
import pandas as pd

import conf

nlp = sp.load("el_core_news_sm")


def getPunctuationFeatures(df):
    """
    A function that:
    1. Counts the number of occurrences  of each of various linguistic features from the title and the body of each
    article in the dataset
    2. Normalizes the values occurred by dividing with the total number of words in the article's title or body
    3. Concatenates the title and body results into a dataframe. Each row of a dataframe represents an article of the
    dataset and the columns are the features collected
    4. Returns the dataframe
    """

    # Collect the number of words
    words_title = []
    words_body = []
    for i in range(0, len(df)):
        words_title.append(len(re.findall(r'\S+', df['Title'][i])))
        words_body.append(len(re.findall(r'\S+', df['Body'][i])))

    # Collect the number uppercase words bigger or equal to 5 letters
    uppercase_title = []
    uppercase_body = []
    for i in range(0, len(df)):
        result1 = re.findall(r'[Α-Ω]{5,}', df['Title'][i])
        uppercase_title.append(len(result1)/words_title[i])
        result2 = re.findall(r'[Α-Ω]{5,}', df['Body'][i])
        uppercase_body.append(len(result2) / words_body[i])

    # Collect the number of Ellispis(Αποσιωπητικά)
    ellipsis_title =[]
    ellipsis_body = []
    for i in range(0, len(df)):
        result1 = re.findall(r'…', df['Title'][i])
        ellipsis_title.append(len(result1)/words_title[i])
        result2 = re.findall(r'…', df['Body'][i])
        ellipsis_body.append(len(result2) / words_body[i])

    # Collect the number of Guillemets(Εισαγωγικά)
    guillements_title = []
    guillements_body = []
    for i in range(0, len(df)):
        result1 = re.findall(r'«.+?»', df['Title'][i])
        guillements_title.append(len(result1)/words_title[i])
        result2 = re.findall(r'«.+?»', df['Body'][i])
        guillements_body.append(len(result2) / words_body[i])

    # Collect the number of Exclamation Marks
    exclamation_title = []
    exclamation_body = []
    for i in range(0, len(df)):
        result1 = re.findall(r'!', df['Title'][i])
        exclamation_title.append(len(result1)/words_title[i])
        result2 = re.findall(r'!', df['Body'][i])
        exclamation_body.append(len(result2) / words_body[i])

    # Create the dataframe
    uppercase_title = pd.DataFrame(uppercase_title, columns=['uppercase_title'])
    uppercase_body = pd.DataFrame(uppercase_body, columns=['uppercase_body'])
    ellipsis_title = pd.DataFrame(ellipsis_title, columns=['ellipsis_title'])
    ellipsis_body = pd.DataFrame(ellipsis_body, columns=['ellipsis_body'])
    guillements_title = pd.DataFrame(guillements_title, columns=['guillements_title'])
    guillements_body = pd.DataFrame(guillements_body, columns=['guillements_body'])
    exclamation_title = pd.DataFrame(exclamation_title, columns=['exclamation_title'])
    exclamation_body = pd.DataFrame(exclamation_body, columns=['exclamation_body'])
    dataframes = [uppercase_title, uppercase_body, ellipsis_title, ellipsis_body,
                  guillements_title, guillements_body, exclamation_title, exclamation_body]
    dataframe = pd.concat(dataframes, axis=1)
    return dataframe


def getNERFeature(df, entity_types):
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
    NER_title = {}
    NER_body = {}

    for doc_num in range(0, num_of_docs):
        doc_title = nlp(df['Title'][doc_num])
        doc_body = nlp(df['Body'][doc_num])

        # The entity categories we will take into account
        for entity_type in entity_types:
            NER_title[entity_type] = 0
            NER_body[entity_type] = 0
        entity_counter_title = 0
        entity_counter_body = 0

        # Counting the entities in the title by their category
        for ent in doc_title.ents:
            entity_counter_title += 1
            for key in NER_title.keys():
                if ent.label_ == key:
                    NER_title[key] += 1
        for key in NER_title.keys():
            if entity_counter_title == 0:
                NER_title[key] = 0
            else:
                NER_title[key] /= entity_counter_title
        NER_all_titles.append(list(NER_title.values()))

        # Counting the entities in the body by their category
        for ent in doc_body.ents:
            entity_counter_body += 1
            for key in NER_body.keys():
                if ent.label_ == key:
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


def getPOSFeature(df, pos_tags):
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
    POS_title = {}
    POS_body = {}

    for doc_num in range(0, num_of_docs):
        doc_title = nlp(df['Title'][doc_num])
        doc_body = nlp(df['Body'][doc_num])

        # The POS categories we will take into account
        for tag in pos_tags:
            POS_title[tag] = 0
            POS_body[tag] = 0
        token_counter_title = 0
        token_counter_body = 0

        # Counting the POS in the title by their category
        for token in doc_title:
            token_counter_title += 1
            for key in POS_title.keys():
                if token.pos_ == key:
                    POS_title[key] += 1
        for key in POS_title.keys():
            if token_counter_title == 0:
                POS_title[key] = 0
            else:
                POS_title[key] /= token_counter_title
        POS_all_titles.append(list(POS_title.values()))

        # Counting the POS in the body by their category
        for token in doc_body:
            token_counter_body += 1
            for key in POS_body.keys():
                if token.pos_ == key:
                    POS_body[key] += 1
        for key in POS_body.keys():
            if token_counter_body == 0:
                POS_body[key] = 0
            else:
                POS_body[key] /= token_counter_body
        POS_all_bodies.append(list(POS_body.values()))

    POS_all_titles = pd.DataFrame(POS_all_titles)
    POS_all_bodies = pd.DataFrame(POS_all_bodies)
    dataframes = [POS_all_titles, POS_all_bodies]
    df = pd.concat(dataframes, axis=1)
    return df

def psycholiguistic(df):
    """
    A function that:
    1. Counts the score from the title and the body based on AFFIN, a psycholinguistic dictionary that was translates
    in greek.
    2. Normalizes the values occurred by dividing with the total number of words in the article's title or body
    3. Concatenates the title and body results into a dataframe. Each row of a dataframe represents an article of the
    dataset and the columns are the features collected
    4. Returns the dataframe
    """
    # Read AFFIN dictionary
    affin = pd.read_excel(conf.CURR_DIR + "AFFIN.xlsx")

    # Convert articles to lowercase because AFFIN is in lowercase
    df['Title_lower'] = df['Title'].str.lower()
    df['Body_lower'] = df['Body'].str.lower()

    # Count the number of words in title and body
    number_of_words_title = []
    number_of_words_body = []
    for i in range(0, len(df)):
        number_of_words_title.append(len(re.findall(r'\S+', df['Title'][i])))
        number_of_words_body.append(len(re.findall(r'\S+', df['Body'][i])))

    # Measure the score for each article
    title = []
    body = []
    for i in range(0, len(df)):
        sum_title = 0
        sum_body = 0
        for y in range(0, len(affin)):
            sum_title += len(re.findall('\\b' + re.escape(affin['word'][y]) + '\\b', df['Title_lower'][i])) * affin['score'][y]
            sum_body += len(re.findall('\\b' + re.escape(affin['word'][y]) + '\\b', df['Body_lower'][i])) * affin['score'][y]
        title.append(sum_title/number_of_words_title[i])
        body.append(sum_body/number_of_words_body[i])

    # Create the dataframe
    affin_title = pd.DataFrame(title)
    affin_body = pd.DataFrame(body)
    dataframes = [affin_title, affin_body]
    dataframe = pd.concat(dataframes, axis=1)
    return dataframe