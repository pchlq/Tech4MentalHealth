import pandas as pd
import config
import torch

class MentalHealthDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, lazy: bool = False):
        self.lazy = lazy
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

        if not self.lazy:
            self.X = []
            self.Y = []
            for i, (row) in tqdm(dataframe.iterrows()):
                x, y = self.row_to_tensor(row)
                self.X.append(x)
                self.Y.append(y)
        else:
            self.df = transform_dataset(dataframe)

    @staticmethod
    def transform_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:

        df = pd.concat([dataframe["text"], pd.get_dummies(dataframe['label'])\
               .reindex(columns=["Depression", "Alcohol", "Suicide", "Drugs"])], axis=1)
        return df

    
    @staticmethod
    def row_to_tensor(row: pd.Series
                     ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        
        tokens = self.tokenizer.encode(row["text"], add_special_tokens=True)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len-1] + [tokens[-1]]
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
            return self.row_to_tensor(self.df.iloc[index])
