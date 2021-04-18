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
# files = ["bi-LSTM-no-decoder", "seq2seq", "transformer", "transformer-params-sharing", "seq2seq+bi-decoder", "transformer-bi-decoder-1-layer"]
files = ["transformer-bi-decoder-1-layer", "no_fine_tune", "transformer_fine_tune_new", "transformer_fine_tune_old"]
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

def show_confusion_matrix():
    with open("confusion_matrix.json", "r") as file:
        cm = json.load(file)["gcard"]
        file.close()
    print(cm)
    confusion_matrix = []
    for i in range(9):
        out = [(i, j, cm.get(str(i), {}).get(str(j), 0)) for j in range(9)]
        confusion_matrix.extend(out)
    df = DataFrame(data=confusion_matrix, columns=["True Label", "Predicted Label", "count"])
    df = df.pivot("True Label", "Predicted Label", "count")
    ax = sns.heatmap(df, cmap="YlGnBu", annot=True)
    plt.show()

# show_confusion_matrix()

def read_result(path):
    with open(path, "r") as file:
        result = json.load(file)
        file.close()
        for k in ["screen", "cpu", "ram", "hdisk", "gcard"]:
            print(k)
            for c in ["recall", "precision", "nDCG"]:
                print(c)
                for i in range(1, 6):
                    v = "%.4f" % (result[k]["%s@%d" % (c, i)] * 0.01)
                    print(float(v))

# read_result("processed_test_result/transformer + bi-decoder + long samples.json")
# read_result("processed_test_result/transformer + bi-decoder + short samples.json")