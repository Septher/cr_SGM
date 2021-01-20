import pandas as pd
import json

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
        # if index == 1:
        #     print(processed_reviews[asin])
    with open(processed_review_data_path, "w") as outfile:
        json.dump(processed_reviews, outfile)

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
        # if index == 1:
        #     print([need for need in df[asin]])
    with open(processed_needs_path, "w") as outfile:
        json.dump(processed_needs, outfile)

if __name__ == '__main__':
    process_review_data()
    process_needs_data()
