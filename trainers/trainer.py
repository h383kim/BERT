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


"""
Evaluate the model on a validation dataset (without parameter updates).
Args:
    model (nn.Module): The PyTorch model to be evaluated.
    val_dataloader (DataLoader): Dataloader providing the validation data.
                                 Each batch is expected to be a list of strings.
    loss_fn (nn.Module): The loss function used (e.g., nn.CrossEntropyLoss()).
Returns:
    float: The average validation loss over the entire dataset.
"""
def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    eval_loss = 0.0

    for X, y_MTP, y_NSP, seg_ids in tqdm(val_dataloader, desc="Validation"):
        # Moving data to cpu or gpu
        X, y_MTP, y_NSP, seg_ids = X.to(DEVICE), y_MTP.to(DEVICE), y_NSP.to(DEVICE), seg_ids.to(DEVICE)
        
        # Feed Forward
        NSP_logits, MTP_logits, _ = model(X, seg_ids)
        
        # Calculate Losses
        loss_NSP = loss_fn(NSP_logits, y_NSP)
        loss_MTP = loss_fn(MTP_logits.permute(0, 2, 1), y_MTP)
        loss = loss_NSP + loss_MTP
        batch_loss = loss.item() * X.shape[0]
        val_loss += batch_loss

    # Compute the average loss across all batches
    val_loss /= len(val_dataloader)
    return val_loss



"""
Main training loop that orchestrates the training and evaluation phases,
and tracks the best model weights based on validation loss.
Args:
    model (nn.Module): The PyTorch model to be trained.
    train_dataloader (DataLoader): Dataloader providing the training data.
    val_dataloader (DataLoader): Dataloader providing the validation data.
    loss_fn (nn.Module): Loss function to be used during training/evaluation.
    optimizer (torch.optim.Optimizer): Optimizer used for model updates.
    scheduler (torch.optim.Optimizer): Scheduler used for lr updates.
    num_epochs (int, optional): Number of epochs to train. Defaults to 1.
Returns:
    nn.Module: The trained model with the best weights (lowest validation loss).
"""
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs=1):
    # Track the best validation loss and corresponding model weights
    best_loss = float('inf')  # or some large constant
    best_wts = copy.deepcopy(model.state_dict())
    
    # For logging losses across epochs
    loss_dict = {'train_loss': [],
                 'val_loss': []}

    # Loop over the desired number of epochs
    for epoch in range(1, num_epochs + 1):
        start = time()
        # Feed forward / backprop on train_dataloader
        train_loss = epoch_train(model, train_dataloader, loss_fn, optimizer, scheduler)
        # Feed forward on val_dataloader
        val_loss = evaluate(model, val_dataloader, loss_fn)

        # Storing epoch histories
        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)

        # Update model depending on its peformance on validation data
        if val_loss < best_loss:
            best_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())

        # Print epoch summary
        end = time()
        time_elapsed = end - start
        print(f"------------ epoch {epoch} ------------")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Time taken: {time_elapsed / 60:.0f}min {time_elapsed % 60:.0f}s")

    # Load the best weights (lowest validation loss) into the model
    model.load_state_dict(best_wts)
    return model