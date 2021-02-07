import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

def save_checkpoint(state, file_prefix, file_suffix="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, file_prefix + "_" + file_suffix)


def load_checkpoint(checkpoint, model, optimizer):  # , steps):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def draw(save_file):
    sns.set_theme()
    data = pd.read_csv(save_file, index_col=0)
    sns.lineplot(data=data, x="number of samples", y="training loss")
    # tips = sns.load_dataset("tips")
    # sns.lineplot(data=datas)
    plt.show()

def draw_line():
    files = []

    data_set = []
    for idx, file in enumerate(files, 0):
        data = pd.read_csv(file, index_col=0)
        description = ""
        data["description"] = description
        data_set.append(data)
    output = pd.concat(data_set)
    sns.set_theme()
    # sns.lineplot(data=output, x="number of samples", y="training loss", hue="description")
    sns.lineplot(data=output, x="number of samples", y="testing accuracy", hue="description")
    plt.show()

loss_data_path = "loss.csv"

def save_data_to_csv(draw_points, columns, save_file=loss_data_path):
    points = DataFrame(data=draw_points, columns=columns)
    points.to_csv(save_file)

def show_result(input_file, baseline_file):
    return