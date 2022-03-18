import torch
from model import bert,bert_cnn,bert_rnn
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import train
if __name__ == '__main__':
    path = "/Users/lee/Desktop/text classification/cleaned_text1.csv"
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    data = pd.read_csv(path)
    data.dropna(axis=0, how="any", inplace=True)
    BATCH_SIZE = 4
    RANDOM_SEED = 1000
    MAX_LEN = 512


    bert_save_path = "bert_best_model_ML.bin"

    df_train, df_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

# BERT
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    model = bert.bertModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)

# BERT+CNN
    bert_save_path = "bertCNN_best_model_ML.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    model = bert_cnn.BertcnnModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)

# BERT+LSTM
    bert_save_path = "bertLSTM_best_model_ML.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LM", BATCH_SIZE)
    model = bert_rnn.BertrnnModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)


# BERT
    bert_save_path = "bert_best_model_LAD.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    model = bert.bertModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)

# BERT+CNN
    bert_save_path = "bertCNN_best_model_LAD.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    model = bert_cnn.BertcnnModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)

# BERT+LSTM
    bert_save_path = "bertLSTM_best_model_LAD.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    val_data_loader = utils.create_data_loader(df_val, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LAD", BATCH_SIZE)
    model = bert_rnn.BertrnnModel(n_classes = 2)
    train.train(model,train_data_loader,val_data_loader,test_data_loader,Epochs=2,save_path=bert_save_path)
