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
    words_title, words_body = countWords(df)
    # Collect the number uppercase words bigger or equal to 5 letters
    uppercase_title = []
    uppercase_body = []
    for i in range(0, len(df)):
        result1 = re.findall(r'[Α-Ω]{5,}', df['Title'][i])
        uppercase_title.append(len(result1)/words_title[i])
        result2 = re.findall(r'[Α-Ω]{5,}', df['Body'][i])
        uppercase_body.append(len(result2) / words_body[i])

    # Collect the number of Ellispis(Αποσιωπητικά)
    ellipsis_title = []
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


def countWords(df):
    """
    A method to count the number of words in an article's title and body
    """
    # Collect the number of words
    words_title = []
    words_body = []
    for i in range(0, len(df)):
        words_title.append(len(re.findall(r'\S+', df['Title'][i])))
        words_body.append(len(re.findall(r'\S+', df['Body'][i])))
    return words_title, words_body


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

        # Counting the entities in the title by their category
        NER_all_titles = countEntities(entity_types, doc_title, NER_title)
        # Counting the entities in the body by their category
        NER_all_bodies = countEntities(entity_types, doc_body, NER_body)

    NER_all_titles = pd.DataFrame(NER_all_titles)
    NER_all_bodies = pd.DataFrame(NER_all_bodies)
    dataframes = [NER_all_titles, NER_all_bodies]
    df = pd.concat(dataframes, axis=1)
    return df


def countEntities(entity_types, doc_part, NER_part):
    """
    A method to be called by the "getNERFeature" method in order to count the entity percentage in the article part
    (title or body) given
    """
    NER_all = []
    for entity_type in entity_types:
        NER_part[entity_type] = 0
    entity_counter = 0

    for ent in doc_part.ents:
        entity_counter += 1
        for key in NER_part.keys():
            if ent.label_ == key:
                NER_part[key] += 1
    for key in NER_part.keys():
        if entity_counter == 0:
            NER_part[key] = 0
        else:
            NER_part[key] /= entity_counter
    NER_all.append(list(NER_part.values()))
    return NER_all


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

        # Counting the POS in the title by their category
        POS_all_titles = getPOScount(doc_title, POS_title)
        # Counting the POS in the body by their category
        POS_all_bodies = getPOScount(doc_body, POS_body)

    POS_all_titles = pd.DataFrame(POS_all_titles)
    POS_all_bodies = pd.DataFrame(POS_all_bodies)
    dataframes = [POS_all_titles, POS_all_bodies]
    df = pd.concat(dataframes, axis=1)
    return df


def getPOScount(doc_part, POS_part):
    """
    A method to be called by the "getPOSFeature" method in order to count the POS tags percentage in the article part
    (title or body) given
    """
    POS_all = []
    token_counter = 0

    for token in doc_part:
        token_counter += 1
        for key in POS_part.keys():
            if token.pos_ == key:
                POS_part[key] += 1
    for key in POS_part.keys():
        if token_counter == 0:
            POS_part[key] = 0
        else:
            POS_part[key] /= token_counter
    POS_all.append(list(POS_part.values()))
    return POS_all


def psycholinguistic(df):
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
    number_of_words_title, number_of_words_body = countWords(df)

    # Measure the score for each article
    title = []
    body = []
    for i in range(0, len(df)):
        sum_title = 0
        sum_body = 0
        for y in range(0, len(affin)):
            sum_title += \
                len(re.findall('\\b' + re.escape(affin['word'][y]) + '\\b', df['Title_lower'][i])) * affin['score'][y]
            sum_body += \
                len(re.findall('\\b' + re.escape(affin['word'][y]) + '\\b', df['Body_lower'][i])) * affin['score'][y]
        title.append(sum_title/number_of_words_title[i])
        body.append(sum_body/number_of_words_body[i])

    # Create the dataframe
    affin_title = pd.DataFrame(title)
    affin_body = pd.DataFrame(body)
    dataframes = [affin_title, affin_body]
    dataframe = pd.concat(dataframes, axis=1)
    return dataframe
