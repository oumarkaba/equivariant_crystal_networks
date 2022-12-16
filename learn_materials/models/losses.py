import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean


def get_loss(hyperparams):
    if hyperparams.loss in ["mae", "MAE", "L1"]:
        loss_function = mae_loss
    elif hyperparams.loss in ["mse", "MSE", "L2"]:
        loss_function = mse_loss
    elif hyperparams.loss in ["gaussian"]:
        loss_function = gaussian_regression
    elif hyperparams.loss in ["quantile"]:
        loss_function = quantile_regression
    elif hyperparams.loss in ["classification"]:
        loss_function = lambda logits, targets: classification(logits, targets, hyperparams.class_threshold)
    elif hyperparams.loss in ["two_part"]:
        loss_function = lambda outputs, targets: two_part_model(outputs, targets, hyperparams.class_threshold)
    elif hyperparams.loss in ["magmoms"]:
        loss_function = magnetic_moments_loss
    return loss_function


def compute_aggretate_metrics(aggregate_metrics, output, targets, hyperparams, inputs):
    if hyperparams.loss in ["mse", "MSE", "L2"]:
        if output is None:
            aggregate_metrics["total_mae"] = 0
        else:
            aggregate_metrics["total_mae"] += mae_loss(output, targets) * len(targets)
    if hyperparams.loss in ["mae", "MAE", "L1"]:
        if output is None:
            aggregate_metrics["total_mse"] = 0
        else:
            aggregate_metrics["total_mse"] += mse_loss(output, targets) * len(targets)
    if hyperparams.loss in ["classification"]:
        if output is None:
            aggregate_metrics["total_f1_score"] = 0
            aggregate_metrics["total_correct_predictions"] = 0
        else:
            aggregate_metrics["total_f1_score"] += f1_score(output, targets, hyperparams.class_threshold) * len(targets)
            aggregate_metrics["total_correct_predictions"] += correct_predictions(
                output, targets, hyperparams.class_threshold
            )
    if hyperparams.loss in ["magmoms"]:
        if output is None:
            aggregate_metrics["total_mae"] = 0
        else:
            aggregate_metrics["total_mae"] += magnetic_moments_loss(output, targets, inputs, mae=True)
    if hyperparams.loss in ["two_part"]:
        if output is None:
            aggregate_metrics["total_f1_score"] = 0
            aggregate_metrics["total_correct_predictions"] = 0
            aggregate_metrics["total_mae"] = 0
        else:
            aggregate_metrics["total_f1_score"] += (
                f1_score(
                    output,
                    targets,
                    hyperparams.class_threshold,
                    single_ouput=False,
                )
                * len(targets)
            )
            aggregate_metrics["total_correct_predictions"] += correct_predictions(
                output,
                targets,
                hyperparams.class_threshold,
                single_ouput=False,
            )
            aggregate_metrics["total_mae"] += (
                mae_classification(
                    output,
                    targets,
                    hyperparams.class_threshold,
                    single_output=False,
                )
                * len(targets)
            )

    return aggregate_metrics


def compute_average_metrics(metrics, aggregate_metrics, validation_dataset_size, hyperparams):
    if hyperparams.loss in ["mse", "MSE", "L2"]:
        mae = aggregate_metrics["total_mae"] / validation_dataset_size
        metrics["MAE"] = mae.item()
        print(f"    mae: {mae:.5f}")
    if hyperparams.loss in ["mae", "MAE", "L1"]:
        mse = aggregate_metrics["total_mse"] / validation_dataset_size
        metrics["MSE"] = mse.item()
        print(f"    mse: {mse:.5f}")
    if hyperparams.loss in ["classification", "two_part"]:
        f1 = aggregate_metrics["total_f1_score"] / validation_dataset_size
        accuracy = aggregate_metrics["total_correct_predictions"] / validation_dataset_size
        metrics["F1 score"] = f1.item()
        metrics["Accuracy"] = accuracy.item()
        print(f"    accuracy: {accuracy:.5f}")
        print(f"    f1: {f1:.5f}")
    if hyperparams.loss in ["two_part"]:
        mae = aggregate_metrics["total_mae"] / validation_dataset_size
        metrics["MAE"] = mae.item()
        print(f"    mae: {mae:.5f}")

    return metrics


def mse_loss(prediction, target):
    prediction = prediction.squeeze()
    target = target.squeeze()
    return F.mse_loss(prediction, target)


def mae_loss(prediction, target):
    prediction = prediction.squeeze()
    target = target.squeeze()
    return F.l1_loss(prediction, target)


def split_lastdim_in_two(outputs):
    output_len = outputs.shape[-1]
    outputs = outputs.view(-1, 2, output_len // 2)
    outputs = outputs.transpose(1, 0)
    return outputs[0], outputs[1]


def gaussian_regression(outputs, targets):
    prediction, uncert = split_lastdim_in_two(outputs)
    prediction = prediction.squeeze()
    uncert = uncert.squeeze()
    target = targets.squeeze()
    loss = torch.log(torch.abs(uncert)) + ((target - prediction) ** 2) / (2 * uncert * uncert)
    return torch.mean(loss)


def quantile_regression(outputs, targets, req_quantile=0.9):
    lower_quantile, upper_quantile = split_lastdim_in_two(outputs)
    lower_quantile = lower_quantile.squeeze()
    upper_quantile = upper_quantile.squeeze()
    targets = targets.squeeze()
    lower_loss = (1 - req_quantile) * F.leaky_relu(targets - lower_quantile, req_quantile / (req_quantile - 1))
    upper_loss = req_quantile * F.leaky_relu(targets - upper_quantile, (req_quantile - 1) / req_quantile)
    return torch.mean(lower_loss + upper_loss)


def classification(logits, targets, threshold):
    logits = logits.squeeze()
    targets = targets.squeeze()
    positive_targets = targets > threshold
    positive_targets = positive_targets.float()

    return F.binary_cross_entropy_with_logits(logits, positive_targets)


def magnetic_moments_loss(outputs, targets, inputs, mae=False):
    graph_to_sites = inputs[7]

    positive_moments = outputs.squeeze()
    negative_moments = -outputs.squeeze()

    target = targets.squeeze()
    if mae:
        positive_loss = F.l1_loss(positive_moments, target, reduction="none")
        negative_loss = F.l1_loss(negative_moments, target, reduction="none")
    else:
        positive_loss = F.mse_loss(positive_moments, target, reduction="none")
        negative_loss = F.mse_loss(negative_moments, target, reduction="none")

    positive_per_graph_loss = scatter_mean(positive_loss, graph_to_sites)
    negative_per_graph_loss = scatter_mean(negative_loss, graph_to_sites)

    per_graph_losses = torch.stack([positive_per_graph_loss, negative_per_graph_loss], dim=1)
    per_graph_losses, _ = torch.min(per_graph_losses, dim=1)

    loss = torch.sum(per_graph_losses)
    return loss


def f1_score(outputs, targets, threshold, single_ouput=True, logits=True):
    if single_ouput:
        outputs = outputs.squeeze()
    else:
        outputs, _ = split_lastdim_in_two(outputs)
        outputs = outputs.squeeze()
    targets = targets.squeeze()
    positive_targets = targets > threshold
    if logits:
        probability = torch.sigmoid(outputs)
        positive_predicted = probability > 0.5
    else:
        positive_predicted = outputs > threshold

    target_true = positive_targets.float().sum() + 1e-9
    predicted_true = positive_predicted.float().sum() + 1e-9
    correct_true = ((positive_predicted == positive_targets) * positive_targets).float().sum()

    precision = (correct_true / predicted_true) + 1e-9
    recall = (correct_true / target_true) + 1e-9

    return 2 * precision * recall / (precision + recall)


def correct_predictions(outputs, targets, threshold, single_ouput=True, logits=True):
    if single_ouput:
        outputs = outputs.squeeze()
    else:
        outputs, _ = split_lastdim_in_two(outputs)
        outputs = outputs.squeeze()
    targets = targets.squeeze()
    positive_targets = targets > threshold
    if logits:
        probability = torch.sigmoid(outputs)
        positive_predicted = probability > 0.5
    else:
        positive_predicted = outputs > threshold

    return torch.eq(positive_targets, positive_predicted).float().sum()


def two_part_model(outputs, targets, threshold):
    logits, prediction = split_lastdim_in_two(outputs)
    targets = targets.squeeze()
    logits = logits.squeeze()
    prediction = prediction.squeeze()
    loss_true = -F.logsigmoid(logits)
    loss_false = -F.logsigmoid(-logits)
    reg_loss = F.l1_loss(prediction, targets, reduction="none")
    loss_terms = torch.stack([loss_false, loss_true, reg_loss])

    positive_targets = targets > threshold
    positive_mask = positive_targets.float()
    zero_mask = (~positive_targets).float()
    mask = torch.stack([zero_mask, positive_mask, positive_mask])

    loss = torch.sum(mask * loss_terms, dim=0)
    return torch.mean(loss)


def mae_classification(outputs, targets, threshold, single_output=True, class_output=None):
    if single_output and class_output is None:
        prediction = outputs.squeeze()
    else:
        logits, prediction = (
            (
                class_output,
                outputs,
            )
            if class_output is not None
            else split_lastdim_in_two(outputs)
        )
        logits = logits.squeeze()
        prediction = prediction.squeeze()
        probability = torch.sigmoid(logits)
        positive_predicted = probability > 0.5
        mask = positive_predicted.float()
        prediction = mask * prediction

    targets = targets.squeeze()
    positive_targets = targets > threshold
    mask_positive = positive_targets.float()
    prediction_positive = mask_positive * prediction
    targets_positive = mask_positive * targets
    return (
        F.l1_loss(prediction_positive, targets_positive),
        F.l1_loss(prediction, targets),
    )
