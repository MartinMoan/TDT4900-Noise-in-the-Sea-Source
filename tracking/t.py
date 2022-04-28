
for epoch in range(epochs):
    # train
    cumu_loss = 0    
    for _, (batch_X, batch_Y) in enumerate(trainset):
        batch_loss = loss.item()
        cumu_loss += batch_loss
        
        log({"batch_loss": batch_loss})

    average_epoch_loss = cumu_loss / len(trainset)
    log({"loss": average_epoch_loss, "epoch": epoch})
    
    # eval
    # Eval method option 1
    cumu_metrics = ...
    for _, (batch_X, batch_Y) in enumerate(evalset):
        preds = model(X)
        cumu_metrics += metrics(preds, batch_Y)
    
    log(cumu_metrics / len(evalset))

    # Eval method option 2
    all_Ys, all_Preds = ..., ...
    for _, (batch_X, batch_Y) in enumerate(evalset):
        preds = model(batch_X)
        all_Ys.append(batch_Y)
        all_Preds.append(preds)
    
    all_metrics = metrics(all_Preds, all_Ys)
    log(all_metrics)


