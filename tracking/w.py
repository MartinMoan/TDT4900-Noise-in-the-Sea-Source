config = {}
all_metrics = {}
sweep_run = wandb.init()
sweep_id = sweep_run.sweep_id or "???"
sweep_run_name = sweep_run.name or sweep_run.id or "???"

for fold in folds:
    # train
    run = wandb.init(
        group=sweep_id,
        job_type=sweep_run_name,
        name=f"Fold {fold}",
        config=config
    )
    for epoch in epochs:
        for batch in batches:
            # train...
    # eval
    metrics = eval(model)
    run.log(metrics)
    run.finish()
    # add metrics to all metrics, but dont log them    
    all_metrics.append(metrics)

# compute final metrics using metrics from all folds
sweep_run.log(mean(all_metrics))
sweep_run.finish()




    