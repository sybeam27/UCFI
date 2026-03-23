import copy
import pandas as pd
import torch.nn as nn

from models.gcn_groupnorm_selective_model import GCNGroupNormSelective
from utils.metrics import evaluate_pyg_model


def train_gcn_groupnorm_selective_model(
    data,
    nfeat,
    hidden_dim=64,
    dropout=0.5,
    lambda_dist=0.05,
    lambda_unc=1.0,
    num_perturbations=4,
    drop_edge_rate=0.1,
    risk_weights=(1.0, 1.0, 1.0),
    priority_exponents=(1.0, 1.0),
    priority_mode="topk",
    priority_k_frac=0.2,
    priority_threshold=None,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    verbose=20,
    selection="tradeoff",
    use_pos_weight=True,
    device="cpu"
):
    pos_weight = None
    if use_pos_weight:
        y_train = data.y[data.idx_train].detach().cpu().numpy()
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        if n_pos > 0:
            pos_weight = n_neg / max(n_pos, 1)

    model = GCNGroupNormSelective(
        nfeat=nfeat,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_dist=lambda_dist,
        lambda_unc=lambda_unc,
        num_perturbations=num_perturbations,
        drop_edge_rate=drop_edge_rate,
        risk_weights=risk_weights,
        priority_exponents=priority_exponents,
        priority_mode=priority_mode,
        priority_k_frac=priority_k_frac,
        priority_threshold=priority_threshold,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
    ).to(device)

    if model.pos_weight is not None:
        model.criterion = nn.BCEWithLogitsLoss(pos_weight=model.pos_weight.to(device))

    best_state = None
    best_score = -1e18 if selection in ["f1", "tradeoff"] else 1e18
    history = []

    for epoch in range(1, epochs + 1):
        train_info = model.optimize(data)

        val_result = evaluate_pyg_model(model, data, split="val")
        test_result = evaluate_pyg_model(model, data, split="test")

        if selection == "f1":
            score = val_result["f1"]
            is_better = score > best_score
        elif selection == "tradeoff":
            score = val_result["f1"] - (val_result["dp"] + val_result["eo"])
            is_better = score > best_score
        elif selection == "fair":
            score = val_result["dp"] + val_result["eo"]
            is_better = score < best_score
        else:
            raise ValueError("selection must be one of ['f1', 'tradeoff', 'fair']")

        if is_better:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": epoch,
            **train_info,
            **{f"val_{k}": v for k, v in val_result.items()},
            **{f"test_{k}": v for k, v in test_result.items()},
        }
        history.append(row)

        if epoch == 1 or epoch % verbose == 0:
            print(
                f"[GCN+GroupNorm+Selective][{epoch:03d}] "
                f"loss={train_info['total_loss']:.4f} "
                f"task={train_info['task_loss']:.4f} "
                f"unc={train_info['unc_loss']:.4f} "
                f"dist={train_info['dist_loss']:.4f} "
                f"selected={train_info['selected_fair_count']} | "
                f"val_acc={val_result['acc']:.4f} "
                f"val_f1={val_result['f1']:.4f} "
                f"val_roc={val_result['roc_auc']:.4f} "
                f"val_dp={val_result['dp']:.4f} "
                f"val_eo={val_result['eo']:.4f}"
            )

    model.load_state_dict(best_state)
    final_val = evaluate_pyg_model(model, data, split="val")
    final_test = evaluate_pyg_model(model, data, split="test")

    return model, pd.DataFrame(history), final_val, final_test