import sys
sys.path.append('/home/mlu/code/cr_SGM')
import torch.nn as nn
from data.dataset_helper import get_data_iter, TEXT, get_fine_tune_iter
from model.Transformer_BI_DECODER import Transformer
from model.SGM import LabelSmoothing
from model.evaluation import evaluate
import torch.optim as optim
import time
from model.utils import load_checkpoint, save_all
from model.params import *
review_train_iter, review_val_iter, need_train_iter, need_val_iter, need_test_iter = get_data_iter()

seq2seq = Transformer().to(DEVICE)
optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)
pad_idx = TEXT.vocab.stoi["<pad>"]
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
criterion = LabelSmoothing(smoothing=0.05)

if load_model:
    load_checkpoint(torch.load("../result/saved_model/06-03-21-51.pth.tar"), seq2seq, optimizer)

def train(model, optimizer, train_iter, val_iter, num_epochs, data_tag, points):
    model.train()
    start_stamp = time.time()
    min_val_loss, checkpoint, best_steps = 0.0, None, -1
    steps, steps_cut = 0, (50 if data_tag == "review" else 15)
    for epoch in range(num_epochs):
        train_iter.init_epoch()
        training_loss, sample_cnt = 0.0, 0
        print(f"[{data_tag} Epoch {epoch} / {num_epochs}]")

        for batch_index, batch in enumerate(train_iter, 1):
            source = batch.text.to(DEVICE)
            target_dict = {
                "cpu": batch.cpu.to(DEVICE),
                "ram": batch.ram.to(DEVICE),
                "screen": batch.screen.to(DEVICE),
                "hdisk": batch.hdisk.to(DEVICE),
                "gcard": batch.gcard.to(DEVICE)
            }
            outputs = model(source, target_dict)
            task_loss = [criterion(outputs[index], target_dict[device]) for index, device in enumerate(DEVICE_ORDER)]
            loss = sum(task_loss)
            training_loss += loss
            sample_cnt += outputs[0].shape[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            steps += 1
            if steps % steps_cut == 0:
                val_loss, val_result = test(model, val_iter, data_tag)
                if checkpoint is None or min_val_loss > val_loss:
                    min_val_loss = val_loss
                    best_steps = steps
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                for device in DEVICE_ORDER:
                    for k in range(1, 6):
                        for c in ["recall", "precision", "nDCG"]:
                            points.append((data_tag, device, epoch, steps, training_loss.item() / sample_cnt, val_loss.item(), val_result[device]["%s@%d" % (c, k)], "%s@%d" % (c, k)))

                print("steps: {}, training_loss: {:.6f}, val_loss: {:.6f}".format(steps, training_loss / sample_cnt, val_loss))
    cost = int(time.time() - start_stamp)
    print("training time cost: {} min {} sec".format(int(cost / 60), cost % 60))
    # choose the model with min val loss
    load_checkpoint(checkpoint, model, optimizer)
    print("best steps: %d, min val loss: %.4f" % (best_steps, min_val_loss))

def train_no_val(model, optimizer, train_iter, num_epochs, data_tag):
    model.train()
    start_stamp = time.time()
    steps = 0
    for epoch in range(num_epochs):
        train_iter.init_epoch()
        training_loss, sample_cnt = 0.0, 0
        print(f"[{data_tag} Epoch {epoch} / {num_epochs}]")

        for batch_index, batch in enumerate(train_iter, 1):
            source = batch.text.to(DEVICE)
            target_dict = {
                "cpu": batch.cpu.to(DEVICE),
                "ram": batch.ram.to(DEVICE),
                "screen": batch.screen.to(DEVICE),
                "hdisk": batch.hdisk.to(DEVICE),
                "gcard": batch.gcard.to(DEVICE)
            }
            outputs = model(source, target_dict)
            task_loss = [criterion(outputs[index], target_dict[device]) for index, device in enumerate(DEVICE_ORDER)]
            loss = sum(task_loss)
            training_loss += loss
            sample_cnt += outputs[0].shape[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            steps += 1
    cost = int(time.time() - start_stamp)
    print("training time cost: {} min {} sec".format(int(cost / 60), cost % 60))

def test(model, data_iter, data_tag):
    model.eval()
    output_with_label = []
    test_loss, sample_cnt = 0.0, 0
    for batch_index, batch in enumerate(data_iter):
        samples = batch.text.to(DEVICE)
        task_dict = {
            "cpu": batch.cpu.to(DEVICE),
            "ram": batch.ram.to(DEVICE),
            "screen": batch.screen.to(DEVICE),
            "hdisk": batch.hdisk.to(DEVICE),
            "gcard": batch.gcard.to(DEVICE)
        }
        outputs = model(samples, task_dict)
        output_with_label.append((outputs, task_dict))
        task_loss = [criterion(outputs[index], task_dict[device]) for index, device in enumerate(DEVICE_ORDER)]
        loss = sum(task_loss)
        test_loss += loss
        sample_cnt += outputs[0].shape[0]

    result = evaluate(output_with_label, data_tag)
    model.train()
    return test_loss / sample_cnt, result

# draw_points = []
# train(seq2seq, optimizer, review_train_iter, review_val_iter, REVIEW_NUM_EPOCHS, "review", draw_points)
# train(seq2seq, optimizer, need_train_iter, need_val_iter, NEED_NUM_EPOCHS, "need", draw_points)
# _, test_result = test(seq2seq, need_test_iter, "need")

# fine tune
# frozen all params
for param in seq2seq.parameters():
    param.requires_grad = False

# replace the last layer and embedding layer
seq2seq.scr_classifier = nn.Linear(EMBEDDING_SIZE, 6)
seq2seq.cpu_classifier = nn.Linear(EMBEDDING_SIZE, 10)
seq2seq.ram_classifier = nn.Linear(EMBEDDING_SIZE, 6)
seq2seq.hdk_classifier = nn.Linear(EMBEDDING_SIZE, 10)
seq2seq.gcd_classifier = nn.Linear(EMBEDDING_SIZE, 9)
seq2seq.classifier = {
    "screen": seq2seq.scr_classifier,
    "cpu": seq2seq.cpu_classifier,
    "ram": seq2seq.ram_classifier,
    "hdisk": seq2seq.hdk_classifier,
    "gcard": seq2seq.gcd_classifier
}

seq2seq.scr_embedding = nn.Embedding(6, EMBEDDING_SIZE)
seq2seq.cpu_embedding = nn.Embedding(10, EMBEDDING_SIZE)
seq2seq.ram_embedding = nn.Embedding(6, EMBEDDING_SIZE)
seq2seq.hdk_embedding = nn.Embedding(10, EMBEDDING_SIZE)
seq2seq.gcd_embedding = nn.Embedding(9, EMBEDDING_SIZE)
seq2seq.task_embedding = {
    "init": seq2seq.ini_embedding,
    "screen": seq2seq.scr_embedding,
    "cpu": seq2seq.cpu_embedding,
    "ram": seq2seq.ram_embedding,
    "hdisk": seq2seq.hdk_embedding,
    "gcard": seq2seq.gcd_embedding
}

# init weight
nn.init.xavier_normal_(seq2seq.scr_classifier.weight)
nn.init.xavier_normal_(seq2seq.cpu_classifier.weight)
nn.init.xavier_normal_(seq2seq.ram_classifier.weight)
nn.init.xavier_normal_(seq2seq.hdk_classifier.weight)
nn.init.xavier_normal_(seq2seq.gcd_classifier.weight)

optimizer = optim.Adam(seq2seq.parameters(), lr=LEARNING_RATE)

draw_points = []
fine_tune_train_iter, fine_tune_val_iter, fine_tune_test_iter = get_fine_tune_iter()
train(seq2seq, optimizer, fine_tune_train_iter, fine_tune_val_iter, 50, "fine_tune_need", draw_points)
_, test_result = test(seq2seq, fine_tune_test_iter, "fine_tune_need")

checkpoint = {
    "state_dict": seq2seq.state_dict(),
    "optimizer": optimizer.state_dict()
}
save_all(checkpoint, draw_points, test_result)
