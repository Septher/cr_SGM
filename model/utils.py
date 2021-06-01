import torch
import json
from pandas import DataFrame
from datetime import datetime
from model.params import get_params_dict, save_model

prefix = "../result/"
loss_file_prefix = prefix + "training_process/"
model_file_prefix = prefix + "saved_model/"
params_file_prefix = prefix + "params/"
test_file_prefix = prefix + "test_result/"
output_file_prefix = prefix + "output_result/"
columns = ["tag", "device", "epoch", "steps", "training_loss", "val_loss", "score", "description"]

def save_checkpoint(state, path):
    print("=> Saving checkpoint, path: %s" % path)
    torch.save(state, path)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def save_to_csv(draw_points, path):
    points = DataFrame(data=draw_points, columns=columns)
    points.to_csv(path)

def save_to_json(data, path):
    with open(path, "w") as fp:
        json.dump(data, fp)
        fp.close()

def save_all(checkpoints, points, test_result, output_result):
    date_str = datetime.now().strftime("%m-%d-%H-%M")
    model_path = "%s%s.pth.tar" % (model_file_prefix, date_str)
    loss_path = "%s%s.csv" % (loss_file_prefix, date_str)
    param_path = "%s%s.json" % (params_file_prefix, date_str)
    test_result_path = "%s%s.json" % (test_file_prefix, date_str)
    output_result_path = "%s%s.json" % (output_file_prefix, date_str)

    if save_model:
        save_checkpoint(checkpoints, model_path)
    save_to_csv(points, loss_path)
    save_to_json(get_params_dict(), param_path)
    save_to_json(test_result, test_result_path)
    save_to_json(output_result, output_result_path)
