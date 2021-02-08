import sys
sys.path.append('/home/mlu/code/cr_SGM')
import torch
import torch.nn as nn
from data.dataset_helper import get_data_iter, TEXT, devices_order
from model.SGM import Encoder, Decoder, Seq2Seq
from model.evaluation import evaluate
import torch.optim as optim
import time
from model.utils import load_checkpoint, save_checkpoint, save_data_to_csv

# training hyper parameters
BATCH_SIZE = 32
REVIEW_NUM_EPOCHS = 50
NEED_NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = False
save_model = True

# model hyper parameters
HIDDEN_SIZE = 512
NUM_LAYERS = 1
# encoder
DROP_OUT_EN = 0.0
WORD_EMBEDDING_SIZE = 200 # GLOVE 6B 200d
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

def train(model, optimizer, train_iter, val_iter, num_epochs, data_tag, points):
    model.train()
    start_stamp = time.time()
    min_test_loss, checkpoint = 0.0, None
    for epoch in range(num_epochs):
        training_loss = 0.0
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
            training_loss += loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        val_loss, val_result = test(model, val_iter, data_tag)
        if data_tag == "need" and epoch > 5 and (checkpoint is None or min_test_loss > val_loss):
            min_test_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

        for device in devices_order:
            for k in range(1, 6):
                for c in ["recall", "precision", "nDCG"]:
                    points.append((data_tag, device, epoch, training_loss.item(), val_loss.item(), val_result[device]["%s@%d" % (c, k)]))
        print("epoch: {}, training_loss: {:.6f}, val_loss: {:.6f}".format(epoch + 1, training_loss, val_loss))
    cost = int(time.time() - start_stamp)
    print("training time cost: {} min {} sec".format(int(cost / 60), cost % 60))
    # choose the model with min val loss after fine-tune
    if data_tag == "need":
        save_checkpoint(checkpoint, data_tag)
        load_checkpoint(checkpoint, model, optimizer)
    else:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, "%d_%s_review" % (REVIEW_NUM_EPOCHS, "no_clip"))

def test(model, data_iter, data_tag):
    model.eval()
    output_with_label = []
    test_loss = 0.0
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
        task_loss = [criterion(outputs[index], task_dict[device]) for index, device in enumerate(devices_order)]
        loss = sum(task_loss)
        test_loss += loss

    result = evaluate(output_with_label, data_tag)
    model.train()
    return test_loss, result

draw_points = []
train(seq2seq, optimizer, review_train_iter, review_val_iter, REVIEW_NUM_EPOCHS, "review", draw_points)
train(seq2seq, optimizer, need_train_iter, need_val_iter, NEED_NUM_EPOCHS, "need", draw_points)
test(seq2seq, need_test_iter, "need")

save_data_to_csv(draw_points, columns=["tag", "device", "epoch", "training_loss", "val_loss", "score"])
