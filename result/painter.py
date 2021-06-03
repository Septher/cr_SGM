import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pandas import DataFrame
from model.params import DEVICE_ORDER

def show_result():
    # df = pd.read_csv(file_name, index_col=0)
    # training loss and val loss during training
    # show_loss(df.loc[df["tag"] == "need"])
    # show_loss(df.loc[df["tag"] == "review"])
    # show recall@k
    show_test_recall()

def show_loss(df):
    training_loss = df[["tag", "steps", "training_loss"]].rename(columns={"training_loss": "loss"})
    training_loss["description"] = "training"
    val_loss = df[["tag", "steps", "val_loss"]].rename(columns={"val_loss": "loss"})
    val_loss["description"] = "validation"
    output = pd.concat([training_loss, val_loss])
    sns.set_theme()
    sns.lineplot(data=output, x="steps", y="loss", hue="description")
    plt.show()

def show_cat_plot(data, title):
    df = DataFrame(data=data, columns=columns)
    g = sns.catplot(
        data=df, kind="bar",
        x="criterion", y="score", hue="description",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("criterion", "score")
    g.legend.set_title(title)
    plt.show()


columns = ["description", "criterion", "score"]
files = ["bi-LSTM-no-decoder", "seq2seq", "transformer", "transformer-params-sharing", "seq2seq+bi-decoder", "transformer-bi-decoder-1-layer"]
def show_test_recall():
    sns.set_theme(style="whitegrid")
    data_points = {}
    terms = ["overall"] + DEVICE_ORDER
    for device in terms:
        data_points[device] = []
    for file in files:
        with open("processed_test_result/%s.json" % file, "r") as fd:
            data = json.load(fd)
            fd.close()
            for device in terms:
                d = [(file, "%s@%d" % (c, i), data[device]["%s@%d" % (c, i)]) for c in ["recall"] for i in range(1, 6)]
                data_points[device].extend(d)
    for device in terms:
        show_cat_plot(data_points[device], device)

# show_result()

label_cnt = {
    "cpu": 10,
    "screen": 6,
    "ram": 6,
    "hdisk": 10,
    "gcard": 8
}

def show_recall_for_each_value_of_all_attribute(training_set_path, training_result_path):
    # sample size in training set
    names = ["text"] + DEVICE_ORDER
    training_set = pd.read_csv(training_set_path, sep="\t", names=names)
    info = {}
    for _, row in training_set.iterrows():
        for device in DEVICE_ORDER:
            value = row[device]
            device_info = info.get(device, {})
            cnt = device_info.get(value, 0) + 1
            device_info[value] = cnt
            info[device] = device_info
    # recall
    with open(training_result_path, "r") as file:
        training_result = json.load(file)
        print(training_result)
        print(info)
        file.close()
    data = []
    for device in DEVICE_ORDER:
        for i in range(label_cnt[device]):
            data.append([device, str(i), training_result[device].get(str(i), 0.0), info[device].get(i, 0)])
    df = DataFrame(data=data, columns=["Attribute", "Label", "Recall", "Number of Training Samples"])

    # draw picture
    sns.set_theme(style="whitegrid")
    sns.scatterplot(x="Number of Training Samples", y="Recall",
                    hue="Attribute", size="Label",
                    data=df)
    plt.show()
# show_recall_for_each_value_of_all_attribute("../data/processed/need_train.tsv", "output_result/06-01-16-29.json")
