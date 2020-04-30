import transformers


MAX_LEN = 192
TRAIN_BATCH_SIZE = 8*8
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "/home/pchlq/workspace/bert-base-uncased/"
MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
