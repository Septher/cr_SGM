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
files = ["lx_baseline", "optimal_params+teacher_forcing_0.5+ls_0.05", "class_weighting"]
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

show_result()