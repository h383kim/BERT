import torch
from torch import nn
import copy
from time import time
from tqdm import tqdm

"""
One epoch of training over the provided training dataloader.
Args:
    model (nn.Module): The PyTorch model to be trained.
    train_dataloader (DataLoader): Dataloader providing the training data.
                                   Each batch is expected to be a list of strings.
    loss_fn (nn.Module): The loss function used (e.g., nn.CrossEntropyLoss()).
    optimizer (torch.optim.Optimizer): The optimizer used for parameter updates.
Returns:
    float: The average training loss over the entire epoch.
"""
def epoch_train(model, train_dataloader, loss_fn, optimizer, scheduler=None):
    # Set model to train mode
    model.train()
    train_loss = 0.0
    # Loop through batches in the training dataloader
    for X, y_MTP, y_NSP, seg_ids in tqdm(train_dataloader, desc="Training"):
        # Moving data to cpu or gpu
        X, y_MTP, y_NSP, seg_ids = X.to(DEVICE), y_MTP.to(DEVICE), y_NSP.to(DEVICE), seg_ids.to(DEVICE)
        
        # Feed Forward
        NSP_logits, MTP_logits, _ = model(X, seg_ids)
        
        # Calculate Losses
        loss_NSP = loss_fn(NSP_logits, y_NSP)
        loss_MTP = loss_fn(MTP_logits.permute(0, 2, 1), y_MTP)
        loss = loss_NSP + loss_MTP
        batch_loss = loss.item() * X.shape[0]
        train_loss += batch_loss
        
        # Opimizer zero_grad
        optimizer.zero_grad()

        # Backpropagate to compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Compute the average loss across all batches
    train_loss /= len(train_dataloader)
    return train_loss
    