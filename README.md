# Models for computer configuration recommendation

Data
data/processed/\*: training set, validation set and test set of review and need data  
data/raw/\*: unprocessed data  

Model  
params.py: hyper parameters  
SGM.py: seq2seq model  
SGM_BI_DECODER.py: seq2seq + bidirectional decoder  
Transformer.py: transformer model  
Transformer_BI_DECODER.py: transformer with bidirectional decoder mask  

training.py: training process of SGM.py and SGM_BI_DECODER.py  
training_transformer.py:  training process of Transformer.py and Transformer_BI_DECODER.py  


Training result  
result_processor.py:  
The input is a file name list that from folder result/test_result, which are the recall, precision and nDCG of the test set  
The reason why the input is a list is that we wants to train a model with same dataset and hyper parameters in several times to get an average, in order to avoid noise.  
output will be saved into result/processed_test_result with file name set by user, it contains the average result of training results

painter.py:  
users can change the content of the list named files to get the histogram of different results.  


How to train a seq2seq model    
step 1: import either SGM or SGM_BI_DECODER in training.py  
step 2: check hyper parameters in params.py if they are suitable  
step 3: run command python training.py  

How to train a transformer model    
step 1: import either Transformer or Transformer_BI_DECODER in training_transformer.py  
step 2: check hyper parameters in params.py if they are suitable  
step 3: run command python training_transformer.py  


|  hyper parameters   | description  |
|  ----  | ----  |
| BATCH_SIZE_REVIEW  | batch size of review data |
| BATCH_SIZE_NEED | batch size of need data |
| REVIEW_NUM_EPOCHS | training epochs of review data |
| NEED_NUM_EPOCHS | training epochs of need data |
| LEARNING_RATE | learning rate |
| DEVICE_ORDER | the prediction order of devices |
| load_model | whether loads a pre-trained model |
| save_model | whether saves the model |
| HIDDEN_SIZE | hidden size of LSTM |
| NUM_LAYERS | layers of LSTM (both encoder and decoder) |
| TEACHER_FORCE | teacher forcing of LSTM decoder |
| DROP_OUT_EN | dropout of LSTM encoder |
| DROP_OUT_DE | dropout of LSTM decoder |
| WORD_EMBEDDING_SIZE | word embedding size of LSTM (we use glove as our word embedding) |
| TASK_EMBEDDING_SIZE | task embedding size of LSTM decoder |
| TRANSFORMER_ENCODER_LAYER | transformer decoder layer |
| TRANSFORMER_DECODER_LAYER | transformer decoder layer |
| NUM_OF_HEADS | number of transformer heads |
| TRANSFORMER_DROPOUT | dropout of transformer |
| EMBEDDING_SIZE | word embedding size of transformer |

|  experiments   | description  |
|  ----  | ----  |
| bi-LSTM-no-decoder | previous baseline |
| seq2seq | seq2seq model baseline |
| seq2seq+bi-decoder | seq2seq with bidirectional decoder |
| seq2seq + bi-decoder + long samples | experiment for long samples |
| seq2seq + bi-decoder + short samples | experiment for short samples |
| transformer | transformer baseline |
| transformer-1-layer-bi-decoder | transformer with bidirectional mask and use only 1 layer decoder to avoid information leaking|
| transformer + bi-decoder + long samples | experiment for long samples |
| transformer + bi-decoder + short samples | experiment for short samples |
| fine_tune | fine tune new data with no new label by using transformer with bidirectional mask |
| fine_tune + data argumentation(only training set X3) | fine tune new data with new label by using transformer with bidirectional mask and apply data argumentation on training set to get 3 times the number of samples|
| fine_tune + 3X data argumentation | fine tune new data with new label by using transformer with bidirectional mask and apply data argumentation on the whole dataset to get 3 times the number of samples |