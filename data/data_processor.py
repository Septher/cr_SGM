import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import re

def process_review_data():
    raw_review_data_path = "raw/amazon_reviews - IEEE.xlsx"
    processed_review_data_path = "processed/review_data.json"

    processed_reviews = dict()
    df = pd.read_excel(raw_review_data_path, sheet_name="total data", engine="openpyxl")
    for index, asin in enumerate(df):
        if index == 0: # skip the first column
            continue
        reviews = df[asin]
        processed_reviews[asin] = [review for review in reviews if len(review.strip()) > 0]
    reviews = clean_data(processed_reviews)
    with open(processed_review_data_path, "w") as outfile:
        json.dump(reviews, outfile)

def process_needs_data():
    raw_needs_path = "raw/needs_byasin (1).csv"
    processed_needs_path = "processed/needs_by_asin.json"
    df = pd.read_csv(raw_needs_path)
    processed_needs = dict()
    for index, asin in enumerate(df):
        if index == 0:
            continue
        needs = df[asin]
        processed_needs[asin] = [need for need in needs if not pd.isnull(need) and len(need.strip()) > 0]
    needs = clean_data(processed_needs)
    with open(processed_needs_path, "w") as outfile:
        json.dump(needs, outfile)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_data(data):
    stop_words = set(stopwords.words('english'))
    for asin, docs in data.items():
        for doc in docs:
            words = nltk.word_tokenize(clean_str(doc))
            words = [word for word in words if re.search("\w", word) is not None and word not in stop_words]
            data[asin] = words
    return data

if __name__ == '__main__':
    nltk.download('stopwords')
    process_review_data()
    process_needs_data()
