import os
import torch
import wandb
from sklearn.metrics import r2_score
import math
from prettytable import PrettyTable

from learn_materials import utils
from learn_materials.prepare.torch_dataset import (
    combine_graph_data,
    combine_supergraph_data,
    merge_datasets,
)
from learn_materials.models.losses import get_loss, compute_aggretate_metrics, compute_average_metrics


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_and_monitor(hyperparams, model):
    print(f"Number of parameters : {count_parameters(model)}")

    if not hyperparams.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    run = wandb.init(
        config=hyperparams,
        project="learn_materials-" + hyperparams.wandb_project,
        entity=hyperparams.wandb_entity,
    )
    run_name = run.name
    wandb.watch(model, log="all")

    if hyperparams.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available")
    else:
        device = torch.device("cpu")

    loss_function = get_loss(hyperparams)

    train_dataset, validation_dataset = get_datasets(hyperparams)

    collate_fn = combine_supergraph_data if (hyperparams.use_supergraphs) else combine_graph_data

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparams.max_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=hyperparams.num_workers,
        drop_last=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=hyperparams.max_batch,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hyperparams.num_workers,
        # drop_last=True,
    )

    model.to(device)

    train_dataset_size = len(train_dataset)
    validation_dataset_size = len(validation_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.lr, weight_decay=hyperparams.weight_decay)

    if hyperparams.schedule:
        if not hyperparams.lr_cycle:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=hyperparams.factor,
                patience=hyperparams.patience,
                min_lr=1e-6,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams.lr_cycle)

    model.zero_grad()

    for epoch in range(1, hyperparams.epochs + 1):
        print(f"epoch: {epoch}")
        metrics = {"Epoch": epoch}

        model.train()
        total_loss = 0
        gradient_accumulation_steps = 0
        max_gradient_steps = math.ceil(hyperparams.batch_size / hyperparams.max_batch)
        for i, (inputs, targets) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"  batch {i+1:5d}/{len(train_loader):5d}")
            inputs = list(x.to(device) for x in inputs)
            targets = targets.to(device)
            outputs = model(inputs)
            if hyperparams.loss == "magmoms":
                loss = loss_function(outputs, targets, inputs)
                total_loss += loss.item()
            else:
                loss = loss_function(outputs, targets)
                total_loss += loss.item() * len(targets)
            loss.backward()
            gradient_accumulation_steps += 1
            if gradient_accumulation_steps == max_gradient_steps:
                optimizer.step()
                model.zero_grad()
                gradient_accumulation_steps = 0
        mean_train_loss = total_loss / train_dataset_size

        metrics["Train loss"] = mean_train_loss
        print(f"    train loss: {mean_train_loss:.5f}")

        model.eval()
        aggregate_metrics = compute_aggretate_metrics(metrics, None, None, hyperparams, inputs)
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = list(x.to(device) for x in inputs)
                targets = targets.to(device)
                output = model(inputs)
                if hyperparams.loss == "magmoms":
                    loss = loss_function(output, targets, inputs)
                    total_loss += loss.item()
                else:
                    loss = loss_function(output, targets)
                    total_loss += loss.item() * len(targets)

                aggregate_metrics = compute_aggretate_metrics(aggregate_metrics, output, targets, hyperparams, inputs)
            mean_valid_loss = total_loss / validation_dataset_size
            metrics["Validation loss"] = mean_valid_loss
            print(f"    valid loss: {mean_valid_loss:.5f}")
            metrics = compute_average_metrics(metrics, aggregate_metrics, validation_dataset_size, hyperparams)
            wandb.log(metrics)
            if hyperparams.schedule:
                if epoch != 1 and not hyperparams.lr_cycle:
                    scheduler.step(wandb.run.summary["Lowest valid metric"])
                # else:
                #     print(hyperparams.lr_cycle)
                #     scheduler.step()
                learning_rate = optimizer.param_groups[0]["lr"]
                metrics["Learning rate"] = learning_rate
                if not hyperparams.lr_cycle and learning_rate <= hyperparams.minimum_lr:
                    print("learning rate reached minimum")
                    break
                else:
                    print(f"    learning rate: {learning_rate:.2e}")

        if hyperparams.loss == "classification":
            record_best_metric = epoch == 1 or metrics["F1 score"] < wandb.run.summary["Lowest valid metric"]
        else:
            record_best_metric = epoch == 1 or metrics["MAE"] < wandb.run.summary["Lowest valid metric"]
        if record_best_metric:
            wandb.run.summary["Lowest valid metric"] = (
                metrics["F1 score"] if hyperparams.loss == "classification" else metrics["MAE"]
            )
            if hyperparams.save_model:
                utils.save_model(
                    model,
                    hyperparams.model_saves_dir / f"model_params_{run_name}.pt",
                )
                wandb.save(str(hyperparams.model_saves_dir / f"model_params_{run_name}.pt"))

    print("\nOptimization ended.\n")


def evaluate(empty_model, hyperparams):
    folder = os.listdir(hyperparams.saved_model)
    metrics = []
    for i, file in enumerate(folder):
        if hyperparams.wandb:
            run = wandb.init(
                config=hyperparams,
                project="learn_materials-" + hyperparams.wandb_project,
                entity=hyperparams.wandb_entity,
                job_type="evaluation",
            )
        if hyperparams.cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using GPU")
            else:
                device = torch.device("cpu")
                print("GPU requested but not available")
        else:
            device = torch.device("cpu")

        model = utils.load_model(empty_model, hyperparams.saved_model / file, device)

        print("Model loaded")

        dataset = torch.load(hyperparams.run_dataset)

        loss_function = get_loss(hyperparams)

        dataset = merge_datasets(dataset, hyperparams.lattices_data, True)
        dataset.select_targets(hyperparams.target_props)

        collate_fn = combine_supergraph_data if (hyperparams.use_supergraphs) else combine_graph_data

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=hyperparams.num_workers,
        )
        model.eval()

        total_loss = 0
        metrics = {}
        aggregate_metrics = {}
        batch_number = 0

        confusion_matrix = torch.zeros(2, 2)
        aggregate_metrics = compute_aggretate_metrics(metrics, None, None, hyperparams, None)
        all_targets = torch.FloatTensor([])
        all_outputs = torch.FloatTensor([])
        with torch.no_grad():
            for inputs, targets in loader:
                batch_number += 1
                print(f"Evaluating on batch {batch_number}")
                inputs = list(x.to(device) for x in inputs)
                targets = targets.to(device)
                output = model(inputs)
                all_targets = torch.cat((all_targets, targets), 0)
                all_outputs = torch.cat((all_outputs, output), 0)
                loss = loss_function(output, targets)
                total_loss += loss.item() * len(targets)

                aggregate_metrics = compute_aggretate_metrics(aggregate_metrics, output, targets, hyperparams, inputs)
            mean_valid_loss = total_loss / len(loader.dataset)
            metrics["Validation loss"] = mean_valid_loss
            print(f"    valid loss: {mean_valid_loss:.5f}")
            metrics = compute_average_metrics(metrics, aggregate_metrics, len(loader.dataset), hyperparams)
            if hyperparams.loss == "classification":

                positive_targets = all_targets > hyperparams.class_threshold
                positive_preds = torch.round(torch.sigmoid(all_outputs))

                for t, p in zip(positive_targets.view(-1), positive_preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                print(confusion_matrix)
                print(f"Metal accuracy: {confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])}")
                print(f"Non-metal accuracy: {confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])}")

            wandb.log(metrics)
            print(metrics)

            targets_outputs = {
                "targets": all_targets.tolist(),
                "predictions": all_outputs.tolist(),
            }
            # utils.save_json(
            #     targets_outputs,
            #     hyperparams.targets_predictions_file,
            # )


def inference(empty_model, hyperparams):
    if hyperparams.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU")
        else:
            device = torch.device("cpu")
            print("GPU requested but not available")
    else:
        device = torch.device("cpu")

    model = utils.load_model(empty_model, hyperparams.saved_model, device)

    print("Model loaded")

    dataset = torch.load(hyperparams.run_dataset)

    print("Dataset loaded")

    collate_fn = combine_supergraph_data if (hyperparams.use_supergraphs) else combine_graph_data

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hyperparams.num_workers,
    )
    model.eval()

    batch_number = 0

    all_indices = torch.FloatTensor([])
    all_outputs = torch.FloatTensor([])

    with torch.no_grad():
        for inputs, index in loader:
            if inputs is None:
                continue
            batch_number += 1
            print(f"Evaluating on batch {batch_number}")
            inputs = list(x.to(device) for x in inputs)
            index = index.to(device)
            output = model(inputs)
            all_indices = torch.cat((all_indices, index), 0)
            all_outputs = torch.cat((all_outputs, output), 0)

    all_indices = all_indices.tolist()
    all_outputs = all_outputs.tolist()
    icsd_ids = dataset.icsd_ids
    ordered_ids = [icsd_ids[int(i)] for i in all_indices]
    predictions = list(zip(ordered_ids, all_outputs))

    utils.save_json(predictions, hyperparams.predictions_file)


def get_datasets(hyperparams):
    if not hyperparams.use_supergraphs:
        if hyperparams.perov:
            train_dataset = torch.load(hyperparams.perov_graphs_train)
            validation_dataset = torch.load(hyperparams.perov_graphs_validation)
        elif hyperparams.dryrun:
            train_dataset = torch.load(hyperparams.graphs_sample_dataset)
            validation_dataset = torch.load(hyperparams.graphs_sample_dataset)
        elif hyperparams.target_props == ["Band gap"] and hyperparams.loss != "classification":
            train_dataset = torch.load(hyperparams.insulators_graphs_train_dataset)
            validation_dataset = torch.load(hyperparams.insulators_graphs_valid_dataset)
        else:
            train_dataset = torch.load(hyperparams.graphs_train_dataset)
            validation_dataset = torch.load(hyperparams.graphs_validation_dataset)
    if hyperparams.target_props == ["Magmoms"]:
        if hyperparams.dryrun:
            train_dataset = torch.load(hyperparams.magmoms_supergraphs_sample_dataset)
            validation_dataset = torch.load(hyperparams.magmoms_supergraphs_sample_dataset)
        else:
            train_dataset = torch.load(hyperparams.magmoms_supergraphs_train_dataset)
            validation_dataset = torch.load(hyperparams.magmoms_supergraphs_validation_dataset)
    elif hyperparams.target_props == ["Band gap"] and hyperparams.loss != "classification":
        if hyperparams.site_props == "atomic_number":
            train_dataset = torch.load(hyperparams.insulators_supergraphs_train_dataset)
            validation_dataset = torch.load(hyperparams.insulators_supergraphs_validation_dataset)
        if hyperparams.site_props == "hubbard":
            train_dataset = torch.load(hyperparams.insulators_supergraphs_hubbard_train_dataset)
            validation_dataset = torch.load(hyperparams.insulators_supergraphs_hubbard_validation_dataset)
        if hyperparams.site_props == "period":
            train_dataset = torch.load(hyperparams.insulators_period_supergraphs_train_dataset)
            validation_dataset = torch.load(hyperparams.insulators_period_supergraphs_validation_dataset)
    elif hyperparams.use_supergraphs:
        if hyperparams.perov:
            train_dataset = torch.load(hyperparams.perov_supergraphs_train)
            validation_dataset = torch.load(hyperparams.perov_supergraphs_validation)
        elif hyperparams.dryrun:
            train_dataset = torch.load(hyperparams.supergraphs_sample_dataset)
            validation_dataset = torch.load(hyperparams.supergraphs_sample_dataset)
        elif hyperparams.site_props == "atomic_number":
            train_dataset = torch.load(hyperparams.supergraphs_train_dataset)
            validation_dataset = torch.load(hyperparams.supergraphs_validation_dataset)
        elif hyperparams.site_props == "hubbard":
            train_dataset = torch.load(hyperparams.supergraphs_hubbard_train_dataset)
            validation_dataset = torch.load(hyperparams.supergraphs_hubbard_validation_dataset)
        elif hyperparams.site_props == "period":
            train_dataset = torch.load(hyperparams.supergraphs_period_train_dataset)
            validation_dataset = torch.load(hyperparams.supergraphs_period_validation_dataset)
    else:
        if hyperparams.dryrun:
            train_dataset = torch.load(hyperparams.graphs_sample_dataset)
            validation_dataset = torch.load(hyperparams.graphs_sample_dataset)
        else:
            train_dataset = torch.load(hyperparams.graphs_train_dataset)
            validation_dataset = torch.load(hyperparams.graphs_validation_dataset)

    if hyperparams.target_props != ["Magmoms"]:
        train_dataset = merge_datasets(train_dataset, hyperparams.lattices_data, True)
        validation_dataset = merge_datasets(validation_dataset, hyperparams.lattices_data, True)
        train_dataset.select_targets(hyperparams.target_props)
        validation_dataset.select_targets(hyperparams.target_props)
    else:
        train_dataset = merge_datasets(train_dataset, hyperparams.lattices_data, False)
        validation_dataset = merge_datasets(validation_dataset, hyperparams.lattices_data, False)

    return train_dataset, validation_dataset
