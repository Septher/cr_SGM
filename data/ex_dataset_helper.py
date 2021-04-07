import json
import spacy
import re
import pandas as pd
from nltk.corpus import stopwords
from torchtext.data import Field
from torchtext import data
from model.params import BATCH_SIZE_REVIEW, BATCH_SIZE_NEED, DEVICE, DEVICE_ORDER

def load_label():
    label_path = "experiments/labels_parsed.json"
    with open(label_path, "r") as file:
        label = json.load(file)
        file.close()
    return label

columns_name = ["text"] + DEVICE_ORDER
def label_permutation(label_dict):
    return [label_dict[device] for device in DEVICE_ORDER]

def load_review_data(label_dict, threshold):
    raw_review_data_path = "raw/amazon_reviews - IEEE.xlsx"

    review_data = []
    df = pd.read_excel(raw_review_data_path, sheet_name="total data", engine="openpyxl")
    for index, asin in enumerate(df):
        if index == 0: # skip the first column
            continue
        raw_reviews = df[asin]
        labels = label_permutation(label_dict[asin])
        for review in raw_reviews:
            if len(review.strip()) > 0:
                tokens = tokenize(review)
                # if len(tokens) <= threshold:
                #     continue
                d = [review] + labels
                review_data.append(d)
    return pd.DataFrame(review_data, columns=columns_name)

def load_needs_data(label_dict, threshold):
    raw_needs_path = "raw/needs_byasin (1).csv"
    df = pd.read_csv(raw_needs_path)
    needs_data = []
    for index, asin in enumerate(df):
        if index == 0:
            continue
        raw_needs = df[asin]
        labels = label_permutation(label_dict[asin])
        for need in raw_needs:
            if not pd.isnull(need) and len(need.strip()) > 0:
                tokens = tokenize(need)
                # if len(tokens) <= threshold:
                #     continue
                d = [need] + labels
                needs_data.append(d)
    return pd.DataFrame(needs_data, columns=columns_name)

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

def save_as_tsv(df, file_name):
    path = "experiments/long/%s" % file_name
    df.to_csv(path, sep='\t', index=False, header=False)

# review data -> 90% train 10% val
# need data -> 60% train 20% val 20% test
def dataset_split_and_save(samples, ratio, prefix):
    if len(ratio) != 3 or abs(sum(ratio) - 1) > 0.0001:
        print("ratio error!")
        return [], [], []
    # shuffle the DataFrame rows
    samples = samples.sample(frac=1)
    n = len(samples)
    train_cnt = int(n * ratio[0])
    val_cnt = int(n * ratio[1])
    test_cnt = n - train_cnt - val_cnt
    print("%s data: train_cnt: %d, val_cnt: %d, test_cnt: %d" % (prefix, train_cnt, val_cnt, test_cnt))
    save_as_tsv(samples[:train_cnt], prefix + "_train.tsv")
    save_as_tsv(samples[train_cnt: train_cnt + val_cnt], prefix + "_val.tsv")
    save_as_tsv(samples[-test_cnt:], prefix + "_test.tsv")

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(clean_str(text))]

def to_dataset(prefix, fields):
    return data.TabularDataset.splits(
        path='../data/experiments/short/', train='%s_train.tsv' % prefix,
        validation='%s_val.tsv' % prefix, test='%s_test.tsv' % prefix, format='tsv',
        fields=fields)

def data_prepare():
    label = load_label()
    reviews_with_label = load_review_data(label, 31)
    needs_with_label = load_needs_data(label, 16)
    dataset_split_and_save(reviews_with_label, [0.9, 0.1, 0.0], "review")
    dataset_split_and_save(needs_with_label, [0.60, 0.2, 0.2], "need")

spacy_en = spacy.load("en_core_web_sm")
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, stop_words=set(stopwords.words('english')))
LABEL = Field(sequential=False, use_vocab=False)
fields = [("text", TEXT)] + [(device, LABEL) for device in DEVICE_ORDER]

def show(data):
    texts = [len(tokenize(row["text"])) for _, row in data.iterrows()]
    texts = sorted(texts, reverse=True)
    l = int(len(texts) / 2)
    print(texts[l])

def data_overview():
    label = load_label()
    reviews_with_label = load_review_data(label, 31)
    needs_with_label = load_needs_data(label, 16)
    show(reviews_with_label)
    show(needs_with_label)

# data_overview()
# data_prepare()

def combine(files, out):
    prefix = ""
    df_short = pd.read_csv("../data/experiments/short/%s.tsv" % files, sep="\t", names=columns_name)
    df_long  = pd.read_csv("../data/experiments/long/%s.tsv" % files, sep="\t", names=columns_name)
    df_out = pd.concat([df_short, df_long])
    df_out.to_csv("experiments/merged/%s.tsv" % out, sep='\t', index=False, header=False)

combine("review_val", "review_val")
combine("need_val", "need_val")
combine("need_test", "need_test")
