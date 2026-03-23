import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


def classification_metrics_from_logits(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(int)
    y_true = labels.detach().cpu().numpy().astype(int)

    acc = (preds == y_true).mean()
    f1 = f1_score(y_true, preds, zero_division=0)

    try:
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        roc = float("nan")

    return {
        "acc": acc,
        "f1": f1,
        "roc_auc": roc,
    }

def fairness_metrics_from_logits(logits, labels, sens, idx):
    idx_np = idx.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()[idx_np].astype(int)
    s = sens.detach().cpu().numpy()[idx_np].astype(int)

    probs = torch.sigmoid(logits[idx]).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(int)

    mask_0 = (s == 0)
    mask_1 = (s == 1)

    p0 = preds[mask_0].mean() if mask_0.sum() > 0 else 0.0
    p1 = preds[mask_1].mean() if mask_1.sum() > 0 else 0.0
    dp = abs(p0 - p1)

    mask_0_y1 = np.logical_and(mask_0, y_true == 1)
    mask_1_y1 = np.logical_and(mask_1, y_true == 1)

    eo0 = preds[mask_0_y1].mean() if mask_0_y1.sum() > 0 else 0.0
    eo1 = preds[mask_1_y1].mean() if mask_1_y1.sum() > 0 else 0.0
    eo = abs(eo0 - eo1)

    return {
        "dp": dp,
        "eo": eo,
    }

@torch.no_grad()
def evaluate_pyg_model(model, data, split="test"):
    model.eval()
    out = model(data)

    if isinstance(out, dict):
        y_logit = out["logits"]
    elif isinstance(out, tuple):
        y_logit = out[0]
    else:
        raise ValueError("model(data) must return either dict or tuple")

    if split == "train":
        idx = data.idx_train
    elif split == "val":
        idx = data.idx_val
    elif split == "test":
        idx = data.idx_test
    else:
        raise ValueError("split must be one of ['train', 'val', 'test']")

    cls = classification_metrics_from_logits(y_logit[idx], data.y[idx])
    fair = fairness_metrics_from_logits(y_logit, data.y, data.sensitive_attr, idx)

    out_dict = {}
    out_dict.update(cls)
    out_dict.update(fair)
    return out_dict