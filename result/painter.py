import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def show_result(file_name):
    df = pd.read_csv(file_name, index_col=0)
    # training loss and val loss during training
    # show_loss(df.loc[df["tag"] == "need"])
    # show_loss(df.loc[df["tag"] == "review"])
    # show recall@k
    show_recall(df.loc[df["tag"] == "need"])
    show_recall(df.loc[df["tag"] == "need"])

def show_loss(df):
    training_loss = df[["tag", "steps", "training_loss"]].rename(columns={"training_loss": "loss"})
    training_loss["description"] = "training"
    val_loss = df[["tag", "steps", "val_loss"]].rename(columns={"val_loss": "loss"})
    val_loss["description"] = "validation"
    output = pd.concat([training_loss, val_loss])
    sns.set_theme()
    sns.lineplot(data=output, x="steps", y="loss", hue="description")
    plt.show()

def show_recall(df):
    data = []
    for i in range(1, 6):
        v = "recall@%d" % i
        d = df[["steps", v]].rename(columns={v: "recall"})
        d["description"] = v
        data.append(d)
    output = pd.concat(data)
    sns.set_theme()
    sns.lineplot(data=output, x="steps", y="recall", hue="description")

# show_result()