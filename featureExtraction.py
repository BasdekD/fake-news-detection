import re
import spacy as sp
import pandas as pd


nlp = sp.load("el_core_news_sm")


def getPunctuationFeatures(df):
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
