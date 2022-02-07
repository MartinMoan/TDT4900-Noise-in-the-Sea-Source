#!/usr/bin/env python3
import torch
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from rich import print
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score

import saver

def evaluate(model, testset, threshold=0.5, device=None):
    def cat(t1, t2):
        if t1 is None:
            return t2
        return torch.cat((t1, t2))

    model.eval()
    with torch.no_grad():
        truth = None
        predictions = None
        probabilities = None
        for i, (X, Y) in enumerate(testset):
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            Yhat = Yhat.cpu()
            X, Y = X.cpu(), Y.cpu()
            preds = (Yhat >= threshold).float()

            truth = cat(truth, Y)
            predictions = cat(predictions, preds)
            probabilities = cat(probabilities, Yhat)

        return truth, predictions, probabilities

def train(model, trainset, epochs=8, lr=1e-3, weight_decay=1e-5, loss_ref=torch.nn.BCELoss, optimizer_ref=torch.optim.Adamax, device=None):
    optimizer = optimizer_ref(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunction = loss_ref()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, (X, Y) in enumerate(trainset):
            X, Y = X.to(device), Y.to(device)
            Yhat = model(X)
            
            optimizer.zero_grad()

            loss = lossfunction(Yhat, Y)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss

            print(f"\t\ttraining epoch {epoch} batch {batch} loss {current_loss}")
            
        average_epoch_loss = epoch_loss / len(trainset)
        saver.save(model, mode="training", avg_epoch_loss=average_epoch_loss)
    return model

def log_fold(model, fold, truth, predictions, probabilities):
    try:
        accuracy = accuracy_score(truth, predictions)
        precision = precision_score(truth, predictions)
        f1 = f1_score(truth, predictions)
        roc_auc = roc_auc_score(truth, probabilities)
        print(f"FOLD {fold} accuracy {accuracy:.8f} precision {precision:.8f} f1 {f1:.8f} roc_auc {roc_auc:.8f}")
        saver.save(model, mode="fold_eval", accuracy=accuracy, precision=precision, f1=f1, roc_auc=roc_auc)
    except Exception as ex:
        print(f"An error occured when computing metrics or storing model: {ex}")

def kfoldcv(model_ref, model_kwargs, dataset, device, kfolds=5, batch_size=8, num_workers=1, train_kwargs={}, eval_kwargs={}):
    
    folds = KFold(n_splits=kfolds, shuffle=True)
    
    for fold, (training_samples, test_samples) in enumerate(folds.split(dataset)):
        print("----------------------------------------")
        print(f"Start fold {fold}")
        print()
        trainset = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(training_samples), num_workers=num_workers)
        testset = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_samples), num_workers=num_workers)

        model = model_ref(**model_kwargs)

        model.to(device)
        
        model = train(model, trainset, **train_kwargs)

        truth, predictions, probabilities = evaluate(model, testset, **eval_kwargs)

        log_fold(model, fold, truth, predictions, probabilities)

        print(f"End fold {fold}")
        print("----------------------------------------")
