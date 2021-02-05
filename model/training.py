import torch
import torch.nn as nn
from data.dataset_helper import get_data_iter, TEXT, devices_order
from model.SGM import Encoder, Decoder, Seq2Seq
from model.evaluation import evaluate
import torch.optim as optim
from model.utils import load_checkpoint, save_checkpoint, save_data_to_csv

# training hyper parameters
BATCH_SIZE = 32
REVIEW_NUM_EPOCHS = 5
NEED_NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = False
save_model = True

# model hyper parameters
HIDDEN_SIZE = 512
NUM_LAYERS = 1
# encoder
DROP_OUT_EN = 0.0
WORD_EMBEDDING_SIZE = 100 # GLOVE 6B 100d
# decoder
TASK_EMBEDDING_SIZE = 128
DROP_OUT_DE = 0.0

review_train_iter, review_val_iter, need_train_iter, need_val_iter, need_test_iter = get_data_iter(BATCH_SIZE, DEVICE)

encoder = Encoder(
    embedding_size=WORD_EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    p=DROP_OUT_EN
).to(DEVICE)

decoder = Decoder(
    embedding_size=TASK_EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    p=DROP_OUT_DE
).to(DEVICE)

seq2seq = Seq2Seq(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
pad_idx = TEXT.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# if load_model:
#     load_checkpoint(torch.load("review_checkpoint.pth.tar"), seq2seq, optimizer)

def train(model, optimizer, train_iter, test_iter, num_epochs, data_tag):
    model.train()
    draw_points = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"[{data_tag} Epoch {epoch} / {num_epochs}]")

        for batch_index, batch in enumerate(train_iter, 1):
            samples = batch.text.to(DEVICE)
            task_dict = {
                "cpu": batch.cpu.to(DEVICE),
                "ram": batch.ram.to(DEVICE),
                "screen": batch.screen.to(DEVICE),
                "hdisk": batch.hdisk.to(DEVICE),
                "gcard": batch.gcard.to(DEVICE)
            }
            outputs = model(samples)
            task_loss = [criterion(outputs[index], task_dict[device]) for index, device in enumerate(devices_order)]
            loss = sum(task_loss)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        test_dict = test(model, test_iter, data_tag)
        for device in devices_order:
            for k in range(1, 6):
                draw_points.append((data_tag, device, epoch, epoch_loss, k, test_dict[device]["recall@%d" % k]))
        print("epoch: {}, epoch_loss: {:.6f}, top1_test_recall_sum: {:.3f}".format(epoch + 1, epoch_loss, sum([test_dict[device]["recall@1"] for device in devices_order])))

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint, data_tag)
    save_data_to_csv(draw_points, columns=["tag", "device", "epoch", "loss", "k", "recall"])


def test(model, data_iter, data_tag):
    model.eval()
    output_with_label = []
    for batch_index, batch in enumerate(data_iter):
        samples = batch.text.to(DEVICE)
        task_dict = {
            "cpu": batch.cpu.to(DEVICE),
            "ram": batch.ram.to(DEVICE),
            "screen": batch.screen.to(DEVICE),
            "hdisk": batch.hdisk.to(DEVICE),
            "gcard": batch.gcard.to(DEVICE)
        }
        outputs = model(samples)
        output_with_label.append((outputs, task_dict))
    test_dict = evaluate(output_with_label, data_tag)
    model.train()
    return test_dict

train(seq2seq, optimizer, review_train_iter, review_val_iter, REVIEW_NUM_EPOCHS, "review")
# train(seq2seq, need_train_iter, need_val_iter, NEED_NUM_EPOCHS, "need")
# test(seq2seq, need_test_iter, "need")
