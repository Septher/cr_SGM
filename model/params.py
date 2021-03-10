import torch

# number of top result -> top_k
K = 5

# training hyper parameters
BATCH_SIZE_REVIEW = 32
BATCH_SIZE_NEED = 16
REVIEW_NUM_EPOCHS = 15
NEED_NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE_ORDER = ["screen", "cpu", "ram", "hdisk", "gcard"]
REVERSED_DEVICE_ORDER = DEVICE_ORDER[::-1]

load_model = False
save_model = False

# model hyper parameters
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TEACHER_FORCE = 0.5
# encoder
DROP_OUT_EN = 0.5
WORD_EMBEDDING_SIZE = 200 # GLOVE 6B 200d
# decoder
TASK_EMBEDDING_SIZE = 128
DROP_OUT_DE = 0.5

# transformer params
TRANSFORMER_DECODER_LAYER = 6
TRANSFORMER_ENCODER_LAYER = 6
NUM_OF_HEADS = 8
TRANSFORMER_DROPOUT = 0.1
EMBEDDING_SIZE = 200

def get_params_dict():
    return {
        "batch_size_review": BATCH_SIZE_REVIEW,
        "batch_size_need": BATCH_SIZE_NEED,
        "review_epochs": REVIEW_NUM_EPOCHS,
        "need_epochs": NEED_NUM_EPOCHS,
        "lr": LEARNING_RATE,
        "device_order": DEVICE_ORDER,
        "hidden_size" : HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "teacher_force": TEACHER_FORCE,
        "drop_out_encoder": DROP_OUT_EN,
        "drop_out_decoder": DROP_OUT_DE,
        "word_embedding_size": WORD_EMBEDDING_SIZE,
        "task_embedding_size": TASK_EMBEDDING_SIZE,
        "optimizer": "Adam"
    }
