import torch

# training hyper parameters
BATCH_SIZE_REVIEW = 32
BATCH_SIZE_NEED = 16
REVIEW_NUM_EPOCHS = 15
NEED_NUM_EPOCHS = 15
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
