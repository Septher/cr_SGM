import torch

def save_checkpoint(state, file_prefix, file_suffix="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, file_prefix + "_" + file_suffix)


def load_checkpoint(checkpoint, model, optimizer):  # , steps):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])