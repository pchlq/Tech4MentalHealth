import pandas as pd
import config
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from transformers import BertTokenizer


class MentalHealthDataset(Dataset):
    
    def __init__(self, tokenizer: BertTokenizer, 
                       dataframe: pd.DataFrame, 
                       lazy: bool = False):
                       
        self.tokenizer = tokenizer
        # self.pad_idx = tokenizer.pad_token_id
        self.lazy = lazy
        if not self.lazy:
            self.X = []
            self.Y = []
            for i, (row) in tqdm(dataframe.iterrows()):
                x, y = self.row_to_tensor(self.tokenizer, row)
                self.X.append(x)
                self.Y.append(y)
        else:
            self.df = dataframe
    
    @staticmethod
    def row_to_tensor(tokenizer: BertTokenizer, 
                      row: pd.Series
                     ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        
        tokens = tokenizer.encode(row["text"], add_special_tokens=True)
        if len(tokens) > config.MAX_LEN:
            tokens = tokens[:config.MAX_LEN-1] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        y = torch.FloatTensor(row[["Depression", "Alcohol", "Suicide", "Drugs"]])
        return x, y
    
    def __len__(self):
        if self.lazy:
            return len(self.df)
        else:
            return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if not self.lazy:
            return self.X[index], self.Y[index]
        else:
            return self.row_to_tensor(self.tokenizer, self.df.iloc[index])
