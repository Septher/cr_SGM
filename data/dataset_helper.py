import json
import spacy
import re
import pandas as pd
from nltk.corpus import stopwords
from torchtext.data import Field
from torchtext import data
from model.params import BATCH_SIZE_REVIEW, BATCH_SIZE_NEED, DEVICE, DEVICE_ORDER
from data.label_parser import label_cnt
import seaborn as sns
import matplotlib.pyplot as plt


def load_label():
    label_path = "processed/labels_parsed.json"
    with open(label_path, "r") as file:
        label = json.load(file)
        file.close()
    return label

columns_name = ["text"] + DEVICE_ORDER
def label_permutation(label_dict):
    return [label_dict[device] for device in DEVICE_ORDER]

def load_review_data(label_dict):
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
                d = [review] + labels
                review_data.append(d)
    return pd.DataFrame(review_data, columns=columns_name)

def load_needs_data(label_dict):
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
    path = "processed/%s" % file_name
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
        path='../data/processed/', train='%s_train.tsv' % prefix,
        validation='%s_val.tsv' % prefix, test='%s_test.tsv' % prefix, format='tsv',
        fields=fields)

def data_prepare():
    label = load_label()
    reviews_with_label = load_review_data(label)
    needs_with_label = load_needs_data(label)
    dataset_split_and_save(reviews_with_label, [0.9, 0.1, 0.0], "review")
    dataset_split_and_save(needs_with_label, [0.60, 0.2, 0.2], "need")


spacy_en = spacy.load("en_core_web_sm")
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, stop_words=set(stopwords.words('english')))
LABEL = Field(sequential=False, use_vocab=False)
fields = [("text", TEXT)] + [(device, LABEL) for device in DEVICE_ORDER]

def get_data_iter():
    need_train, need_val, need_test = to_dataset("need", fields)
    review_train, review_val, review_test = to_dataset("review", fields)

    TEXT.build_vocab(review_train, vectors="glove.6B.200d")

    need_train_iter, need_val_iter, need_test_iter = data.Iterator.splits(
        (need_train, need_val, need_test), sort_key=lambda x: len(x.text),
        batch_sizes=(BATCH_SIZE_NEED, BATCH_SIZE_NEED, BATCH_SIZE_NEED), device=DEVICE, shuffle=True)

    review_train_iter, review_val_iter, _ = data.Iterator.splits(
        (review_train, review_val, review_test), sort_key=lambda x: len(x.text),
        batch_sizes=(BATCH_SIZE_REVIEW, BATCH_SIZE_REVIEW, BATCH_SIZE_REVIEW), device=DEVICE, shuffle=True)

    return review_train_iter, review_val_iter, need_train_iter, need_val_iter, need_test_iter

# data_prepare()

# distribution in the training data
def get_label_distribution():
    paths = {"need": "../data/processed/need_train.tsv", "review": "../data/processed/review_train.tsv"}
    # get the distribution of labels for each task
    num_dict = {"need": {}, "review": {}, "total": {}}
    for device in DEVICE_ORDER:
        for k in ["need", "review", "total"]:
            num_dict[k][device] = {}
            for i in range(label_cnt[device]):
                num_dict[k][device][i] = 0
    for k in ["need", "review"]:
        df = pd.read_csv(paths[k], sep="\t", names=columns_name)
        for _, row in df.iterrows():
            for device in DEVICE_ORDER:
                v = int(row[device])
                num_dict[k][device][v] += 1
                num_dict["total"][device][v] += 1
    return num_dict

def get_criterion_weight():
    paths = {"need": "../data/processed/need_train.tsv", "review": "../data/processed/review_train.tsv"}
    # get the distribution of labels for each task
    num_dict = get_label_distribution()
    # calculate the weight of each label based on the distribution of labels
    weight_dict = {"need": {}, "review": {}}
    for device in DEVICE_ORDER:
        for kd in ["need", "review"]:
            weights = [0] * label_cnt[device]
            for k, v in num_dict[kd][device].items():
                weights[k] = 1.0 / v
            weight_dict[kd][device] = weights
    return weight_dict

# The relationship between recall and training sample size
def draw_picture():
    label_cnt = {
        "cpu": 10,
        "screen": 6,
        "ram": 6,
        "hdisk": 10,
        "gcard": 8
    }
    num_dict = get_label_distribution()
    data_points = []
    with open("test_recall.json", "r") as f:
        import json
        test_recall_dict = json.load(f)
        f.close()
        for device in DEVICE_ORDER:
            for i in range(label_cnt[device]):
                data_points.append((
                    int(num_dict["need"][device][i]),
                    0 if test_recall_dict[device][str(i)]["total"] == 0 else 1.0 * test_recall_dict[device][str(i)]["correct"] / test_recall_dict[device][str(i)]["total"],
                    i,
                    device
                ))
    df = pd.DataFrame(data_points, columns=["sample_cnt", "recall", "label_id", "device"])
    sns.set_theme(style="whitegrid")
    # Draw a categorical scatterplot to show each observation
    # ax = sns.swarmplot(data=df, x="sample_cnt", y="recall", hue="device", size=5)
    # ax.set(ylabel="", xlabel="")
    # ticks = ax.get_xticks()
    # labels = ax.get_xticklabels()
    # ax.set_xticks(ticks[4::5])
    # ax.set_xticklabels(labels[4::5])
    # ax.set_yticks([0, 1])
    sns.scatterplot(
        data=df,
        x="sample_cnt",
        y="recall",
        hue="device"
    )

    #    ax.set(xlim=(0, 500), ylim=(0.0, 1.0))
    plt.show()

# draw_picture()
