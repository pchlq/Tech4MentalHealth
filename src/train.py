import config
from dataset import MentalHealthDataset
import engine
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from typing import Tuple, List

from model import BertClassifier
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

SEED = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_SIZE = 0.15
LR = 2e-5



def run():
    
    def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]], 
               device: torch.device
              ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    
        x, y = list(zip(*batch))
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = torch.stack(y)
        return x.to(device), y.to(device)


    df = pd.read_csv("../inputs/Train.csv")
    # test = pd.read_csv("../inputs/Test.csv")

    train_df, val_df = train_test_split(df, stratify=df.label, test_size=VALID_SIZE, random_state=SEED)

    labels = ["Depression", "Alcohol", "Suicide", "Drugs"]
    train = pd.concat([train_df["text"], pd.get_dummies(train_df['label'])\
               .reindex(columns=labels)], axis=1)#.reset_index(drop=True)

    valid = pd.concat([val_df["text"], pd.get_dummies(val_df['label'])\
               .reindex(columns=labels)], axis=1)#.reset_index(drop=True)
    
    if DEVICE == 'cpu':
        print('cpu')
    else:
        n_gpu = torch.cuda.device_count()
        print(torch.cuda.get_device_name(0))

    train_dataset = MentalHealthDataset(config.TOKENIZER, train, lazy=True)
    valid_dataset = MentalHealthDataset(config.TOKENIZER, valid, lazy=True)
    collate_fn = partial(collate_fn, device=DEVICE)

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)

    train_iterator = DataLoader(train_dataset, 
                                batch_size=config.TRAIN_BATCH_SIZE, 
                                sampler=train_sampler, 
                                collate_fn=collate_fn)

    valid_iterator = DataLoader(valid_dataset, 
                                batch_size=config.VALID_BATCH_SIZE, 
                                sampler=valid_sampler, 
                                collate_fn=collate_fn)

    # model = BertClassifier().to(DEVICE)
    model = BertClassifier(BertModel.from_pretrained(config.BERT_PATH), 4).to(DEVICE)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays 
    warmup_steps = 10**3 # 10 ** 3
    total_steps = len(train_iterator) * config.EPOCHS - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR) # 1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
    #                                         patience=5, factor=0.3, min_lr=1e-10, verbose=True)

    for epoch in range(config.EPOCHS):
        print('=' * 5, f"EPOCH {epoch}", '=' * 5)
        engine.train_fn(train_iterator, model, optimizer, scheduler)
        engine.eval_fn(valid_iterator, model)


    model.eval()
    test_df = pd.read_csv("../inputs/Test.csv")
    submission = pd.read_csv('../inputs/SampleSubmission.csv')
    res = np.zeros((submission.shape[0], len(labels)))

    for i in tqdm(range(len(test_df) // config.TRAIN_BATCH_SIZE + 1)):
        batch_df = test_df.iloc[i * config.TRAIN_BATCH_SIZE: (i + 1) * config.TRAIN_BATCH_SIZE]
        assert (batch_df["ID"] == submission["ID"][i * config.TRAIN_BATCH_SIZE: (i + 1) * config.TRAIN_BATCH_SIZE]).all(), f"Id mismatch"
        texts = []
        for text in batch_df["text"].tolist():
            text = config.TOKENIZER.encode(text, add_special_tokens=True)
            if len(text) > config.MAX_LEN:
                text = text[:config.MAX_LEN-1] + [config.TOKENIZER.sep_token_id]
            texts.append(torch.LongTensor(text))
        x = pad_sequence(texts, batch_first=True, padding_value=config.TOKENIZER.pad_token_id).to(DEVICE)
        mask = (x != config.TOKENIZER.pad_token_id).float().to(DEVICE)

        with torch.no_grad():
            _, outputs = model(x, attention_mask=mask)
        outputs = outputs.cpu().numpy()
        submission.loc[i * config.TRAIN_BATCH_SIZE: (i * config.TRAIN_BATCH_SIZE + len(outputs)-1), labels] = outputs

    submission.to_csv("../subs/submission_2.csv", index=False)

if __name__ == "__main__":
    run()
    
