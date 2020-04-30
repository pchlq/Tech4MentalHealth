import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import log_loss



def loss_fn(outputs, targets):
    return nn.BCELoss(outputs, targets) # nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, scheduler):
    model.train()
    for x, y in tqdm(data_loader):     # total=len(data_loader)

        optimizer.zero_grad()
        mask = (x != 0).float()
        loss, outputs = model(x, attention_mask=mask, labels=y)
        # loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(data_loader):
            mask = (x != 0).float()
            loss, outputs = model(x, attention_mask=mask, labels=y)
            # loss = loss_fn(outputs, y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    for i, name in enumerate(["Depression", "Alcohol", "Suicide", "Drugs"]):
        print(f"{name} log-loss: {log_loss(true[:, i], pred[:, i])}")
    print(f"Eval loss {total_loss / len(data_loader)}")