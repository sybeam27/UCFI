import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


# 분류 성능
def classification_metrics(logits, labels):
    # 현실적으로 f1 + roc_auc 같이 보는 게 기본

    # 확률 변환
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    # 예측 클래스 생성
    preds = (probs > 0.5).astype(int) 
    y_true = labels.detach().cpu().numpy().astype(int)

    # 정확도
    acc = (preds == y_true).mean()
    # 클래스 불균형 고려 성능
    f1 = f1_score(y_true, preds, zero_division=0)

    # threshold-free 성능
    try:
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        roc = float("nan")

    return {"acc": float(acc), "f1": float(f1), "roc_auc": float(roc)}

def classification_fairness_metrics(logits, labels, sens, idx):
    """Classification fairness: DP, EO, SP"""
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy()

    y_true = labels.detach().cpu().numpy()[idx].astype(int)
    s = sens.detach().cpu().numpy()[idx].astype(int)
    probs = torch.sigmoid(logits[idx]).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(int)

    mask_0 = (s == 0)
    mask_1 = (s == 1)

    # Demographic Parity (== Statistical Parity here)
    # 한 집단이 더 많이 긍정 평가 받는지
    p0 = preds[mask_0].mean() if mask_0.sum() > 0 else 0.0
    p1 = preds[mask_1].mean() if mask_1.sum() > 0 else 0.0
    dp = abs(p0 - p1)

    # Equal Opportunity: TPR gap
    # 실제 y=1인 경우만 비교
    # 같이 자격 있는 사람인데 집단 때문에 덜 뽑히는지
    mask_0_y1 = np.logical_and(mask_0, y_true == 1)
    mask_1_y1 = np.logical_and(mask_1, y_true == 1)
    eo0 = preds[mask_0_y1].mean() if mask_0_y1.sum() > 0 else 0.0
    eo1 = preds[mask_1_y1].mean() if mask_1_y1.sum() > 0 else 0.0
    eo = abs(eo0 - eo1)

    return {
        "dp": float(dp),
        "eo": float(eo),
    }


# 회귀 성능
def regression_metrics(preds, labels):
    """
    Regression performance metrics.
    preds: torch.Tensor or np.ndarray, shape (N,) or (N,1)
    labels: torch.Tensor or np.ndarray, shape (N,) or (N,1)
    """
    # 현실적으로 MAE + RMSE 둘 다 보는 게 안전
    if torch.is_tensor(preds):
        y_pred = preds.detach().cpu().numpy().reshape(-1)
    else:
        y_pred = np.asarray(preds).reshape(-1)

    if torch.is_tensor(labels):
        y_true = labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(labels).reshape(-1)

    # 큰 오차에 민감
    mse = mean_squared_error(y_true, y_pred)
    # mse의 직관 버전
    rmse = np.sqrt(mse)
    # 평균 절대 오차 (robust)
    mae = mean_absolute_error(y_true, y_pred)
    # 설명력
    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = float("nan")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

def regression_fairness_metrics(preds, labels, sens, idx=None):
    """
    Regression fairness metrics.

    Returns:
    - mse_gap: group-wise MSE difference
    - rmse_gap: group-wise RMSE difference
    - mae_gap: group-wise MAE difference
    - bias_gap: group-wise signed error mean difference
    - mean_pred_gap: mean prediction difference between groups
    - mean_residual_gap: mean residual difference between groups

    Also returns each group's raw values.
    """
    if torch.is_tensor(preds):
        y_pred = preds.detach().cpu().numpy().reshape(-1)
    else:
        y_pred = np.asarray(preds).reshape(-1)

    if torch.is_tensor(labels):
        y_true = labels.detach().cpu().numpy().reshape(-1)
    else:
        y_true = np.asarray(labels).reshape(-1)

    if torch.is_tensor(sens):
        s = sens.detach().cpu().numpy().reshape(-1).astype(int)
    else:
        s = np.asarray(sens).reshape(-1).astype(int)

    if idx is not None:
        if torch.is_tensor(idx):
            idx = idx.detach().cpu().numpy()
        y_pred = y_pred[idx]
        y_true = y_true[idx]
        s = s[idx]

    mask_0 = (s == 0)
    mask_1 = (s == 1)

    def _group_stats(mask):
        if mask.sum() == 0:
            return {
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "mean_pred": float("nan"),
                "mean_residual": float("nan"),
                "n": 0,
            }

        yt = y_true[mask]
        yp = y_pred[mask]
        residual = yp - yt

        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(yt, yp)

        # 특정 집단을 체계적으로 깍아내리는지 부풀리는지
        bias = residual.mean()          # signed error
        mean_pred = yp.mean()
        mean_residual = residual.mean()

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
            "mean_pred": float(mean_pred),
            "mean_residual": float(mean_residual),
            "n": int(mask.sum()),
        }

    g0 = _group_stats(mask_0)
    g1 = _group_stats(mask_1)

    def _gap(a, b):
        if np.isnan(a) or np.isnan(b):
            return float("nan")
        return float(abs(a - b))

    return {
        "mse_gap": _gap(g0["mse"], g1["mse"]),
        "rmse_gap": _gap(g0["rmse"], g1["rmse"]),
        "mae_gap": _gap(g0["mae"], g1["mae"]),

        # 한 집단을 계속 과소/과대 예측하냐
        "bias_gap": _gap(g0["bias"], g1["bias"]),
        "mean_pred_gap": _gap(g0["mean_pred"], g1["mean_pred"]),

        # bias gap 중복 확인용
        # "mean_residual_gap": _gap(g0["mean_residual"], g1["mean_residual"]),
        # "group_0": g0,
        # "group_1": g1,
    }

# 평가
def evaluate_pyg_model(model, data, split="val", task_type="classification"):
    model.eval()

    if split == "train":
        idx = data.idx_train
    elif split == "val":
        idx = data.idx_val
    elif split == "test":
        idx = data.idx_test
    else:
        raise ValueError("split must be one of ['train', 'val', 'test'].")

    with torch.no_grad():
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(-1)

    y = data.y
    s = data.sensitive_attr

    if task_type == "classification":
        perf = classification_metrics(out[idx], y[idx])
        fair = classification_fairness_metrics(out, y, s, idx)
    elif task_type == "regression":
        perf = regression_metrics(out[idx], y[idx])
        fair = regression_fairness_metrics(out, y, s, idx)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    result = {**perf, **fair}

    result = {
        k: (round(v, 4) if isinstance(v, (float, int)) else v)
        for k, v in result.items()
    }

    return result



#### 나중에 보기
# def fairness_on_subset(logits, labels, sens, mask):
#     if mask.sum() == 0:
#         return {"dp": 0.0, "eo": 0.0, "sp": 0.0}
#     return fairness_metrics_from_logits(logits, labels, sens, mask)

# def neighborhood_fairness_metrics(logits, sensitive_attr, edge_index, idx):
#     """1-hop neighborhood-level DP 평균/표준편차"""
#     if not torch.is_tensor(idx):
#         idx = torch.tensor(idx, device=edge_index.device)

#     src, dst = edge_index
#     probs = torch.sigmoid(logits).detach()
#     local_dps = []

#     for i in idx:
#         neigh_mask = (src == i) | (dst == i)
#         neigh_nodes = torch.unique(torch.cat([src[neigh_mask], dst[neigh_mask]]))
#         if len(neigh_nodes) < 2:
#             continue

#         s_neigh = sensitive_attr[neigh_nodes]
#         p_neigh = probs[neigh_nodes]

#         m0 = (s_neigh == 0)
#         m1 = (s_neigh == 1)
#         if m0.sum() == 0 or m1.sum() == 0:
#             local_dps.append(0.0)
#             continue

#         local_dps.append(abs(p_neigh[m0].mean() - p_neigh[m1].mean()))

#     if len(local_dps) == 0:
#         return {"mean_local_dp": 0.0, "std_local_dp": 0.0}

#     local_dps = torch.tensor(local_dps)
#     return {
#         "mean_local_dp": local_dps.mean().item(),
#         "std_local_dp": local_dps.std().item() if len(local_dps) > 1 else 0.0
#     }

# @torch.no_grad()
# def evaluate_pyg_model(model, data, split="test", 
    #                    boundary_threshold=0.5, 
    #                    priority_percentile=70,
    #                    return_details=False):
    # model.eval()
    # out = model(data)

    # if isinstance(out, dict):
    #     y_logit = out.get("logits") or out.get("y_logit")
    # elif isinstance(out, tuple):
    #     y_logit = out[0]
    # else:
    #     y_logit = out

    # # split index
    # if split == "train": idx = data.idx_train
    # elif split == "val": idx = data.idx_val
    # elif split == "test": idx = data.idx_test
    # else: raise ValueError("Invalid split")

    # # 1. Global metrics
    # cls = classification_metrics_from_logits(y_logit[idx], data.y[idx])
    # fair = fairness_metrics_from_logits(y_logit, data.y, data.sensitive_attr, idx)

    # result = {**cls, **fair}
    # result = {f"global_{k}" if k in ["dp","eo","sp"] else k: v for k, v in result.items()}

    # # 2. Boundary Fairness
    # if hasattr(data, 'boundary_score') and data.boundary_score is not None:
    #     boundary_mask = (data.boundary_score >= boundary_threshold) & idx.bool() if torch.is_tensor(idx) else idx
    #     if boundary_mask.sum() > 10:
    #         fb = fairness_on_subset(y_logit, data.y, data.sensitive_attr, boundary_mask)
    #         result.update({
    #             "boundary_dp": fb["dp"], "boundary_eo": fb["eo"], "boundary_sp": fb["sp"],
    #             "boundary_size": int(boundary_mask.sum())
    #         })

    # # 3. High-Priority (FIPS) Fairness
    # if hasattr(model, 'compute_risk_and_priority'):
    #     try:
    #         p_dict = model.compute_risk_and_priority(data)
    #         priority = p_dict.get("priority_pred")
    #         if priority is not None:
    #             thresh = torch.quantile(priority[idx].float(), priority_percentile / 100.0)
    #             high_mask = (priority >= thresh) & idx.bool() if torch.is_tensor(idx) else idx
    #             if high_mask.sum() > 10:
    #                 fh = fairness_on_subset(y_logit, data.y, data.sensitive_attr, high_mask)
    #                 result.update({
    #                     "high_priority_dp": fh["dp"], "high_priority_eo": fh["eo"], "high_priority_sp": fh["sp"],
    #                     "high_priority_size": int(high_mask.sum())
    #                 })
    #     except:
    #         pass

    # # 4. Neighborhood Fairness
    # if hasattr(data, 'edge_index'):
    #     neigh = neighborhood_fairness_metrics(y_logit, data.sensitive_attr, data.edge_index, idx)
    #     result.update(neigh)

    # if return_details:
    #     result["details"] = {"boundary_threshold": boundary_threshold, "priority_percentile": priority_percentile}

    # return result