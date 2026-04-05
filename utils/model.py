import copy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, SGConv

from utils.metrics import (
    classification_metrics,
    classification_fairness_metrics,
    regression_metrics,
    regression_fairness_metrics,
    evaluate_pyg_model,
)


# =========================================================
# Loader output -> PyG Data
# =========================================================
def build_pyg_data_from_loader_dict(dataset_dict, device, task_type="classification"):
    adj = dataset_dict["adj"]
    if not sp.isspmatrix(adj):
        raise TypeError("dataset_dict['adj'] must be a scipy sparse matrix.")

    adj        = adj.tocoo()
    edge_index = torch.tensor(
        [adj.row, adj.col], dtype=torch.long, device=device,
    )

    x = dataset_dict["features"].float().to(device)

    if task_type == "classification":
        y = dataset_dict["labels"].long().view(-1).to(device)
    elif task_type == "regression":
        y = dataset_dict["labels"].float().view(-1).to(device)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    sensitive_attr = dataset_dict["sens"].long().view(-1).to(device)

    data = Data(x=x, edge_index=edge_index, y=y, sensitive_attr=sensitive_attr)
    data.idx_train = dataset_dict["idx_train"].long().to(device)
    data.idx_val   = dataset_dict["idx_val"].long().to(device)
    data.idx_test  = dataset_dict["idx_test"].long().to(device)

    if "idx_sens_train" in dataset_dict:
        data.idx_sens_train = dataset_dict["idx_sens_train"].long().to(device)

    return data


# =========================================================
# Backbone Models
# =========================================================
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_dim=1, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        if return_hidden:
            return out, h
        return out

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_dim=1, dropout=0.5):
        super().__init__()
        self.conv1   = SAGEConv(in_feats, h_feats)
        self.conv2   = SAGEConv(h_feats, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        if return_hidden:
            return out, h
        return out

class SGC(nn.Module):
    def __init__(self, in_feats, out_dim=1, K=2):
        super().__init__()
        self.conv = SGConv(in_feats, out_dim, K=K)

    def forward(self, data, edge_index=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        out = self.conv(x, edge_index).view(-1)
        if return_hidden:
            return out, x   # SGC는 별도 hidden 없으므로 입력 피처 반환
        return out

def build_backbone(name, in_feats, h_feats, dropout=0.5, sgc_k=2):
    if name == "GCN":
        return GCN(in_feats, h_feats, out_dim=1, dropout=dropout)
    elif name == "GraphSAGE":
        return GraphSAGE(in_feats, h_feats, out_dim=1, dropout=dropout)
    elif name == "SGC":
        return SGC(in_feats, out_dim=1, K=sgc_k)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

class BaselineGNN:
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        task_type="classification",
        name="GCN",
        dropout=0.1,
        sgc_k=2,
    ):
        assert task_type in ["classification", "regression"]
        assert name in ["GCN", "GraphSAGE", "SGC"]

        self.name = f"{name}/Baseline"
        self.backbone_name = name
        self.task_type = task_type
        self.device = device

        self.model = build_backbone(
            name=name,
            in_feats=in_feats,
            h_feats=h_feats,
            dropout=dropout,
            sgc_k=sgc_k,
        ).to(device)

    def _build_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _build_criterion(self):
        if self.task_type == "classification":
            return nn.BCEWithLogitsLoss()
        return nn.MSELoss()

    def _compute_val_score(self, val_result):
        if self.task_type == "classification":
            return float(val_result.get("acc", 0.0))
        else:
            return -float(val_result.get("mae", float("inf")))

    def train_step(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        out = self.model(data).view(-1)
        labels = data.y.float()
        idx_train = data.idx_train

        loss = criterion(out[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        return {
            "total_loss": float(loss.item()),
            "task_loss": float(loss.item()),
        }

    def fit(
        self,
        data,
        epochs=300,
        lr=1e-3,
        weight_decay=0.0,
        patience=50,
        verbose=True,
        print_interval=50,
    ):
        optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        criterion = self._build_criterion()

        best_val_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        counter = 0

        for epoch in range(epochs):
            train_info = self.train_step(data, optimizer, criterion)

            val_result = evaluate_pyg_model(
                self.model,
                data,
                split="val",
                task_type=self.task_type,
            )
            val_score = self._compute_val_score(val_result)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                train_result = evaluate_pyg_model(
                    self.model,
                    data,
                    split="train",
                    task_type=self.task_type,
                )
                print(
                    f"[{self.name}] "
                    f"Epoch {epoch+1:04d} | "
                    f"Loss {train_info['total_loss']:.4f} | "
                    f"Train {train_result} | "
                    f"Val {val_result} | "
                    f"ValScore {val_score:.4f}"
                )

            if counter >= patience:
                break

        self.model.load_state_dict(best_state)

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model,
            data,
            split=split,
            task_type=self.task_type,
        )


# =========================================================
# Fairness Loss Functions
# =========================================================
class GroupWiseNorm(nn.Module):
    """그룹 간 평균/분산 차이를 줄이는 differentiable regularizer"""
    def __init__(self):
        super().__init__()

    def forward(self, z, sensitive_attr, idx=None):
        if idx is not None:
            z              = z[idx]
            sensitive_attr = sensitive_attr[idx]
        if z.dim() == 1:
            z = z.unsqueeze(1)

        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)
        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return z.new_tensor(0.0)

        z0, z1   = z[mask_0], z[mask_1]
        mean0, mean1 = z0.mean(dim=0), z1.mean(dim=0)
        var0  = z0.var(dim=0, unbiased=False)
        var1  = z1.var(dim=0, unbiased=False)

        return torch.abs(mean0 - mean1).mean() + torch.abs(var0 - var1).mean()

class WeightedGroupWiseNorm(nn.Module):
    """FIPS 가중 GroupWiseNorm"""
    def __init__(self):
        super().__init__()

    def forward(self, z, sensitive_attr, weight, idx=None):
        if idx is not None:
            z              = z[idx]
            sensitive_attr = sensitive_attr[idx]
            weight         = weight[idx]
        if z.dim() == 1:
            z = z.unsqueeze(1)

        weight = weight / (weight.mean() + 1e-8)
        w      = weight.unsqueeze(1)

        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)
        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return z.new_tensor(0.0)

        z0, w0 = z[mask_0], w[mask_0]
        z1, w1 = z[mask_1], w[mask_1]

        mean0 = (z0 * w0).sum(dim=0) / (w0.sum() + 1e-8)
        mean1 = (z1 * w1).sum(dim=0) / (w1.sum() + 1e-8)
        var0  = ((z0 - mean0) ** 2 * w0).sum(dim=0) / (w0.sum() + 1e-8)
        var1  = ((z1 - mean1) ** 2 * w1).sum(dim=0) / (w1.sum() + 1e-8)

        return torch.abs(mean0 - mean1).mean() + torch.abs(var0 - var1).mean()

def classification_output_fairness_loss(prob, labels, sensitive_attr, idx=None):
    """
    분류용 output-level fairness surrogate
    - DP: P(ŷ=1|s=0) vs P(ŷ=1|s=1)
    - EO: P(ŷ=1|y=1,s=0) vs P(ŷ=1|y=1,s=1)
    """
    if idx is not None:
        prob           = prob[idx]
        labels         = labels[idx]
        sensitive_attr = sensitive_attr[idx]

    labels = labels.float()
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return prob.new_tensor(0.0)

    dp_loss = torch.abs(prob[mask_0].mean() - prob[mask_1].mean())

    mask_0_y1 = mask_0 & (labels == 1)
    mask_1_y1 = mask_1 & (labels == 1)
    if mask_0_y1.sum() == 0 or mask_1_y1.sum() == 0:
        return dp_loss

    eo_loss = torch.abs(prob[mask_0_y1].mean() - prob[mask_1_y1].mean())
    return dp_loss + eo_loss

def regression_output_fairness_loss(preds, labels, sensitive_attr, idx=None):
    """
    회귀용 output-level fairness surrogate
    - mean_pred_gap: 그룹 간 평균 예측값 차이
    - bias_gap:      그룹 간 잔차 평균 차이
    """
    if idx is not None:
        preds          = preds[idx]
        labels         = labels[idx]
        sensitive_attr = sensitive_attr[idx]

    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return preds.new_tensor(0.0)

    pred0, pred1 = preds[mask_0], preds[mask_1]
    y0,    y1    = labels[mask_0], labels[mask_1]

    mean_pred_gap = torch.abs(pred0.mean() - pred1.mean())
    bias_gap      = torch.abs((pred0 - y0).mean() - (pred1 - y1).mean())
    return mean_pred_gap + bias_gap

def weighted_classification_fairness_loss(prob, labels, sensitive_attr,
                                           weight, idx=None):
    """FIPS 가중 분류 fairness loss (DP + EO)"""
    if idx is not None:
        prob           = prob[idx]
        labels         = labels[idx]
        sensitive_attr = sensitive_attr[idx]
        weight         = weight[idx]

    labels = labels.float()
    weight = weight / (weight.mean() + 1e-8)

    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return prob.new_tensor(0.0)

    p0 = (prob[mask_0] * weight[mask_0]).sum() / (weight[mask_0].sum() + 1e-8)
    p1 = (prob[mask_1] * weight[mask_1]).sum() / (weight[mask_1].sum() + 1e-8)
    dp_loss = torch.abs(p0 - p1)

    mask_0_y1 = mask_0 & (labels == 1)
    mask_1_y1 = mask_1 & (labels == 1)
    if mask_0_y1.sum() == 0 or mask_1_y1.sum() == 0:
        return dp_loss

    eo0 = (prob[mask_0_y1] * weight[mask_0_y1]).sum() / (weight[mask_0_y1].sum() + 1e-8)
    eo1 = (prob[mask_1_y1] * weight[mask_1_y1]).sum() / (weight[mask_1_y1].sum() + 1e-8)
    return dp_loss + torch.abs(eo0 - eo1)

def weighted_regression_fairness_loss(preds, labels, sensitive_attr,
                                       weight, idx=None):
    """FIPS 가중 회귀 fairness loss (mean_pred_gap + bias_gap)"""
    if idx is not None:
        preds          = preds[idx]
        labels         = labels[idx]
        sensitive_attr = sensitive_attr[idx]
        weight         = weight[idx]

    weight = weight / (weight.mean() + 1e-8)
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return preds.new_tensor(0.0)

    pred0, w0 = preds[mask_0], weight[mask_0]
    pred1, w1 = preds[mask_1], weight[mask_1]
    y0, y1    = labels[mask_0], labels[mask_1]

    mean_pred_gap = torch.abs(
        (pred0 * w0).sum() / (w0.sum() + 1e-8) -
        (pred1 * w1).sum() / (w1.sum() + 1e-8)
    )
    bias_gap = torch.abs(
        ((pred0 - y0) * w0).sum() / (w0.sum() + 1e-8) -
        ((pred1 - y1) * w1).sum() / (w1.sum() + 1e-8)
    )
    return mean_pred_gap + bias_gap


# =========================================================
# SBRS: Structural Bias Risk Score
# =========================================================
def compute_sbrs(data):
    """
    SBRS(v) = alpha * w_degree(v) + beta * w_boundary(v) + gamma * w_lhd(v)

    alpha, beta, gamma: 각 컴포넌트의 분산 비율로 자동 결정 (데이터 기반)

    반환: (sbrs_n [N], (alpha, beta, gamma))
    """
    edge_index = data.edge_index
    sens       = data.sensitive_attr
    N          = data.x.size(0)
    device     = data.x.device

    ones     = torch.ones(edge_index.size(1), device=device)
    src, dst = edge_index

    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, ones)

    # w_degree
    log_deg  = torch.log1p(deg)
    w_degree = (log_deg - log_deg.min()) / (log_deg.max() - log_deg.min() + 1e-8)

    # w_boundary
    cross_edge  = (sens[src] != sens[dst]).float()
    cross_count = torch.zeros(N, device=device)
    cross_count.scatter_add_(0, src, cross_edge)
    boundary_ratio = cross_count / (deg + 1e-8)
    w_boundary = (boundary_ratio - boundary_ratio.min()) / \
                 (boundary_ratio.max() - boundary_ratio.min() + 1e-8)

    # w_lhd
    same_edge  = (sens[src] == sens[dst]).float()
    same_count = torch.zeros(N, device=device)
    same_count.scatter_add_(0, src, same_edge)
    local_h  = same_count / (deg + 1e-8)
    global_h = local_h.mean()
    lhd      = torch.abs(local_h - global_h)
    w_lhd    = (lhd - lhd.min()) / (lhd.max() - lhd.min() + 1e-8)

    # 분산 기반 자동 가중치
    vars_ = torch.stack([w_degree.var(), w_boundary.var(), w_lhd.var()])
    vars_ = vars_ / (vars_.sum() + 1e-8)
    alpha, beta, gamma = vars_[0].item(), vars_[1].item(), vars_[2].item()

    sbrs   = alpha * w_degree + beta * w_boundary + gamma * w_lhd
    sbrs_n = (sbrs - sbrs.min()) / (sbrs.max() - sbrs.min() + 1e-8)

    return sbrs_n.detach(), (alpha, beta, gamma)

# =========================================================
# Model Uncertainty (Prediction Entropy)
# =========================================================
@torch.no_grad()
def estimate_model_uncertainty(model, data, task_type="classification"):
    """
    모델 불확실성 추정 (Luo et al. uncertainty-maximization principle 기반).

    분류: 이진 엔트로피 H(p) = -p log p - (1-p) log(1-p)
    회귀: 1 / (|ŷ| + ε)

    1회 forward pass로 계산 (T회 반복 불필요).

    반환: w_unc [N] float tensor in [0, 1] (정규화됨, detached)
    """
    model.eval()
    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(-1)

    if task_type == "classification":
        prob        = torch.sigmoid(out).clamp(1e-6, 1 - 1e-6)
        uncertainty = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    else:
        uncertainty = 1.0 / (out.abs() + 1e-8)

    w_unc = (uncertainty - uncertainty.min()) / \
            (uncertainty.max() - uncertainty.min() + 1e-8)
    return w_unc.detach()


# =========================================================
# FIW Node Risk Weights
# =========================================================
def compute_node_risk_weights(
    data,
    model=None,
    sbrs_threshold=0.5,
    lam=1.0,
    min_weight=0.5,
    max_weight=2.0,
    task_type="classification",
    ablate_sbrs=False,
    ablate_uncertainty=False,
):
    """
    2단계 게이팅 기반 FIW 노드 가중치.

    1단계: SBRS >= sbrs_threshold 인 노드만 개입 대상 선정
    2단계: SBRS × (1 + λ·w_unc) 로 최종 가중치 조정

    model=None이면 SBRS만 사용 (warm-up 중).

    ablation:
        ablate_sbrs=True        → Uncertainty only
        ablate_uncertainty=True → SBRS only

    반환: (weight [N], (alpha, beta, gamma))
    """
    N      = data.x.size(0)
    device = data.x.device

    if not ablate_sbrs:
        sbrs_n, (alpha, beta, gamma) = compute_sbrs(data)
    else:
        sbrs_n = torch.ones(N, device=device)
        alpha = beta = gamma = 1 / 3

    if model is None:
        weight = min_weight + (max_weight - min_weight) * sbrs_n
        return weight.detach(), (alpha, beta, gamma)

    if not ablate_uncertainty:
        w_unc = estimate_model_uncertainty(model, data, task_type=task_type)
    else:
        w_unc = torch.zeros(N, device=device)

    weight    = torch.full((N,), min_weight, device=device)
    gate_mask = sbrs_n >= sbrs_threshold

    if gate_mask.sum() > 0:
        combined   = sbrs_n[gate_mask] * (1.0 + lam * w_unc[gate_mask])
        combined_n = (combined - combined.min()) / \
                     (combined.max() - combined.min() + 1e-8)
        weight[gate_mask] = min_weight + (max_weight - min_weight) * combined_n

    return weight.detach(), (alpha, beta, gamma)


# =========================================================
# Base: BaseMultiLevelFairGNN
# =========================================================
class BaseMultiLevelFairGNN:
    """
    구조(Structure), 표현(Representation), 출력(Output) 세 레벨
    fairness loss + warm-up 기반 자동 스케일 보정 + 단일 lambda_fair.
    """
    def __init__(self, task_type, device, lambda_fair=0.5, warm_up=100):
        assert task_type in ["classification", "regression"]
        self.task_type   = task_type
        self.device      = device
        self.lambda_fair = lambda_fair
        self.warm_up     = warm_up
        self.group_norm  = GroupWiseNorm()
        self._loss_scales = {"struct": 1.0, "rep": 1.0, "out": 1.0}

    # ---------- required override ----------
    def _build_model(self):       raise NotImplementedError
    def _build_criterion(self):   raise NotImplementedError
    def compute_output_loss(self, preds_or_logits, labels, sensitive_attr, idx_fair):
        raise NotImplementedError
    def _compute_val_score(self, val_result): raise NotImplementedError

    # ---------- shared ----------
    def _build_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
        )

    def _get_fair_idx(self, data):
        if hasattr(data, "idx_sens_train") and data.idx_sens_train is not None:
            return data.idx_sens_train
        return data.idx_train

    @staticmethod
    def perturb_edge_index(edge_index, drop_rate=0.15):
        if drop_rate <= 0.0:
            return edge_index
        num_edges = edge_index.size(1)
        device    = edge_index.device
        keep_mask = torch.rand(num_edges, device=device) > drop_rate
        if keep_mask.sum() == 0:
            keep_mask[torch.randint(0, num_edges, (1,), device=device)] = True
        return edge_index[:, keep_mask]

    def compute_structure_loss(self, data, h_orig):
        idx_fair  = self._get_fair_idx(data)
        edge_pert = self.perturb_edge_index(
            data.edge_index, drop_rate=self.drop_edge_rate_struct)
        _, h_pert = self.model(data, edge_index=edge_pert, return_hidden=True)
        dim = h_orig.size(-1)
        return F.mse_loss(h_orig[idx_fair], h_pert[idx_fair]) / dim

    def compute_representation_loss(self, h, sensitive_attr, idx_fair):
        return self.group_norm(h[idx_fair], sensitive_attr[idx_fair])

    # ---------- calibration ----------
    @torch.no_grad()
    def _calibrate_loss_scales(self, data, criterion):
        """warm-up 직후 1회: task_loss 기준으로 각 fairness loss scale 고정"""
        self.model.eval()
        labels         = data.y.float()
        idx_train      = data.idx_train
        idx_fair       = self._get_fair_idx(data)
        sensitive_attr = data.sensitive_attr

        out, h   = self.model(data, return_hidden=True)
        task_val = criterion(out[idx_train], labels[idx_train]).item()

        def _scale(loss_tensor):
            v = loss_tensor.item()
            return task_val / (v + 1e-8)

        scales = {}
        scales["struct"] = _scale(self.compute_structure_loss(data, h)) \
                           if not self.ablate_struct else 1.0
        scales["rep"]    = _scale(
            self.compute_representation_loss(h, sensitive_attr, idx_fair)) \
                           if not self.ablate_rep else 1.0
        scales["out"]    = _scale(
            self.compute_output_loss(out, labels, sensitive_attr, idx_fair)) \
                           if not self.ablate_out else 1.0

        self._loss_scales = scales
        self.model.train()

        print(
            f"[{self.name}] Loss scale calibrated | "
            f"task={task_val:.4f} | "
            f"struct={scales['struct']:.3f} | "
            f"rep={scales['rep']:.3f} | "
            f"out={scales['out']:.3f}"
        )

    # ---------- train step ----------
    def train_step(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        labels         = data.y.float()
        idx_train      = data.idx_train
        idx_fair       = self._get_fair_idx(data)
        sensitive_attr = data.sensitive_attr

        preds_or_logits, h = self.model(data, return_hidden=True)
        task_loss = criterion(preds_or_logits[idx_train], labels[idx_train])

        struct_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_struct:
            struct_loss = self.compute_structure_loss(data, h)

        rep_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_rep:
            rep_loss = self.compute_representation_loss(h, sensitive_attr, idx_fair)

        out_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_out:
            out_loss = self.compute_output_loss(
                preds_or_logits, labels, sensitive_attr, idx_fair)

        total_loss = (
            task_loss
            + self.lambda_fair * (
                self._loss_scales["struct"] * struct_loss
                + self._loss_scales["rep"]  * rep_loss
                + self._loss_scales["out"]  * out_loss
            )
        )

        total_loss.backward()
        optimizer.step()

        return {
            "total_loss":  float(total_loss.item()),
            "task_loss":   float(task_loss.item()),
            "struct_loss": float(struct_loss.item()),
            "rep_loss":    float(rep_loss.item()),
            "out_loss":    float(out_loss.item()),
        }

    # ---------- fit ----------
    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=0.0,
            patience=50, verbose=True, print_interval=50):
        optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        criterion = self._build_criterion()

        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")
        for _ in range(self.warm_up):
            self.train_step(data, optimizer, criterion)

        self._calibrate_loss_scales(data, criterion)

        best_val_score = -float("inf")
        best_state     = copy.deepcopy(self.model.state_dict())
        counter        = 0
        remaining      = epochs - self.warm_up

        if verbose:
            print(f"[{self.name}] Phase 2: main training {remaining} epochs...")

        for epoch in range(remaining):
            train_info = self.train_step(data, optimizer, criterion)
            val_result = evaluate_pyg_model(
                self.model, data, split="val", task_type=self.task_type)
            val_score = self._compute_val_score(val_result)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state     = copy.deepcopy(self.model.state_dict())
                counter        = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                train_result = evaluate_pyg_model(
                    self.model, data, split="train", task_type=self.task_type)
                print(
                    f"[{self.name}] "
                    f"Epoch {epoch + self.warm_up + 1:04d} | "
                    f"Total {train_info['total_loss']:.4f} | "
                    f"Task {train_info['task_loss']:.4f} | "
                    f"Struct {train_info['struct_loss']:.4f} | "
                    f"Rep {train_info['rep_loss']:.4f} | "
                    f"Out {train_info['out_loss']:.4f} | "
                    f"Train {train_result} | "
                    f"Val {val_result} | "
                    f"ValScore {val_score:.4f}"
                )

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch "
                          f"{epoch + self.warm_up + 1}.")
                break

        self.model.load_state_dict(best_state)
        if verbose:
            print(f"[{self.name}] Training finished. "
                  f"Best val score: {best_val_score:.4f}")

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model, data, split=split, task_type=self.task_type)

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        out = self.model(data)
        if isinstance(out, tuple):
            out = out[0]
        return out.view(-1)

    @torch.no_grad()
    def predict_proba(self, data):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only valid for classification.")
        return torch.sigmoid(self.predict(data))


# =========================================================
# FnCGNN / FnRGNN (균일 다단계 공정성)
# =========================================================
class FnCGNN(BaseMultiLevelFairGNN):
    """균일 다단계 공정성 개입 - 분류"""
    def __init__(
        self, in_feats, h_feats, device,
        name="GCN", dropout=0.1, sgc_k=2,
        drop_edge_rate_struct=0.15,
        lambda_fair=0.5, warm_up=100,
        ablate_struct=False, ablate_rep=False, ablate_out=False,
        val_tradeoff_dp=0.5, val_tradeoff_eo=0.5,
    ):
        super().__init__(task_type="classification", device=device,
                         lambda_fair=lambda_fair, warm_up=warm_up)
        assert name in ["GCN", "GraphSAGE", "SGC"]
        self.name                  = f"{name}/FnCGNN"
        self.in_feats              = in_feats
        self.h_feats               = h_feats
        self.backbone_name         = name
        self.dropout               = dropout
        self.sgc_k                 = sgc_k
        self.drop_edge_rate_struct = drop_edge_rate_struct
        self.ablate_struct         = ablate_struct
        self.ablate_rep            = ablate_rep
        self.ablate_out            = ablate_out
        self.val_tradeoff_dp       = val_tradeoff_dp
        self.val_tradeoff_eo       = val_tradeoff_eo
        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(self.backbone_name, self.in_feats,
                              self.h_feats, self.dropout, self.sgc_k)

    def _build_criterion(self):
        return nn.BCEWithLogitsLoss()

    def compute_output_loss(self, logits, labels, sensitive_attr, idx_fair):
        return classification_output_fairness_loss(
            torch.sigmoid(logits), labels, sensitive_attr, idx=idx_fair)

    def _compute_val_score(self, val_result):
        acc = float(val_result.get("acc", 0.0))
        dp  = abs(float(val_result.get("dp", 0.0)))
        eo  = abs(float(val_result.get("eo", 0.0)))
        return acc - self.val_tradeoff_dp * dp - self.val_tradeoff_eo * eo

class FnRGNN(BaseMultiLevelFairGNN):
    """균일 다단계 공정성 개입 - 회귀"""
    def __init__(
        self, in_feats, h_feats, device,
        name="GCN", dropout=0.1, sgc_k=2,
        drop_edge_rate_struct=0.15,
        lambda_fair=0.5, warm_up=100,
        ablate_struct=False, ablate_rep=False, ablate_out=False,
        val_tradeoff_mae=1.0, val_tradeoff_bias=0.5, val_tradeoff_mean_pred=0.5,
    ):
        super().__init__(task_type="regression", device=device,
                         lambda_fair=lambda_fair, warm_up=warm_up)
        assert name in ["GCN", "GraphSAGE", "SGC"]
        self.name                   = f"{name}/FnRGNN"
        self.in_feats               = in_feats
        self.h_feats                = h_feats
        self.backbone_name          = name
        self.dropout                = dropout
        self.sgc_k                  = sgc_k
        self.drop_edge_rate_struct  = drop_edge_rate_struct
        self.ablate_struct          = ablate_struct
        self.ablate_rep             = ablate_rep
        self.ablate_out             = ablate_out
        self.val_tradeoff_mae       = val_tradeoff_mae
        self.val_tradeoff_bias      = val_tradeoff_bias
        self.val_tradeoff_mean_pred = val_tradeoff_mean_pred
        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(self.backbone_name, self.in_feats,
                              self.h_feats, self.dropout, self.sgc_k)

    def _build_criterion(self):
        return nn.MSELoss()

    def compute_output_loss(self, preds, labels, sensitive_attr, idx_fair):
        return regression_output_fairness_loss(
            preds, labels, sensitive_attr, idx=idx_fair)

    def _compute_val_score(self, val_result):
        mae           = float(val_result.get("mae", float("inf")))
        bias_gap      = abs(float(val_result.get("bias_gap", 0.0)))
        mean_pred_gap = abs(float(val_result.get("mean_pred_gap", 0.0)))
        return -(
            self.val_tradeoff_mae       * mae
            + self.val_tradeoff_bias    * bias_gap
            + self.val_tradeoff_mean_pred * mean_pred_gap
        )

# =========================================================
# NodeAwareBase (FIPS 2단계 게이팅)
# ========================================================
class NodeAwareBase(BaseMultiLevelFairGNN):
    """
    FIPS 기반 2단계 게이팅 차등 공정성 개입 베이스 클래스.

    Phase 1 (warm-up): SBRS만으로 균일 학습 → 모델 안정화
    전환점:
        _update_node_weights_fips() → 노드 가중치 고정
        _calibrate_loss_scales()    → loss scale 고정
    Phase 2: FIPS 가중치 + 고정 scale + lambda_fair
    """
    def __init__(
        self, task_type, device,
        lambda_fair=0.5,
        sbrs_threshold=0.5, lam=1.0,
        min_weight=0.5, max_weight=2.0,
        warm_up=100,
        ablate_sbrs=False, ablate_uncertainty=False,
    ):
        super().__init__(task_type=task_type, device=device,
                         lambda_fair=lambda_fair)
        # warm_up은 uncertainty 추정 전용으로 독립 관리
        self.warm_up            = warm_up
        self.sbrs_threshold     = sbrs_threshold
        self.lam                = lam
        self.min_weight         = min_weight
        self.max_weight         = max_weight
        self.ablate_sbrs        = ablate_sbrs
        self.ablate_uncertainty = ablate_uncertainty
        self.group_norm         = WeightedGroupWiseNorm()
        self._node_weight       = None

    def _init_node_weights_sbrs(self, data):
        self._node_weight, (a, b, g) = compute_node_risk_weights(
            data, model=None,
            min_weight=self.min_weight, max_weight=self.max_weight,
            ablate_sbrs=self.ablate_sbrs,
            ablate_uncertainty=self.ablate_uncertainty,
        )
        w = self._node_weight
        print(f"[{self.name}] Phase 1 NodeWeight (SBRS only) | "
              f"alpha={a:.3f} beta={b:.3f} gamma={g:.3f} | "
              f"min={w.min():.3f} max={w.max():.3f} "
              f"mean={w.mean():.3f} std={w.std():.3f}")

    def _update_node_weights_fips(self, data):
        N = data.x.size(0)
        self._node_weight, (a, b, g) = compute_node_risk_weights(
            data, model=self.model,
            sbrs_threshold=self.sbrs_threshold, lam=self.lam,
            min_weight=self.min_weight, max_weight=self.max_weight,
            task_type=self.task_type,
            ablate_sbrs=self.ablate_sbrs,
            ablate_uncertainty=self.ablate_uncertainty,
        )
        w         = self._node_weight
        gate_mask = w > self.min_weight
        print(f"[{self.name}] Phase 2 NodeWeight (FIPS) | "
              f"alpha={a:.3f} beta={b:.3f} gamma={g:.3f} | "
              f"gated={gate_mask.sum().item()}/{N} "
              f"({100*gate_mask.float().mean().item():.1f}%) | "
              f"min={w.min():.3f} max={w.max():.3f} "
              f"mean={w.mean():.3f} std={w.std():.3f}")

    def compute_structure_loss(self, data, h_orig):
        idx_fair  = self._get_fair_idx(data)
        edge_pert = self.perturb_edge_index(
            data.edge_index, drop_rate=self.drop_edge_rate_struct)
        _, h_pert = self.model(data, edge_index=edge_pert, return_hidden=True)
        dim = h_orig.size(-1)
        w   = self._node_weight[idx_fair]
        w   = w / (w.mean() + 1e-8)
        node_mse = ((h_orig[idx_fair] - h_pert[idx_fair]) ** 2).mean(dim=-1)
        return (node_mse * w).mean() / dim

    def compute_representation_loss(self, h, sensitive_attr, idx_fair):
        return self.group_norm(
            h, sensitive_attr, weight=self._node_weight, idx=idx_fair)

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=0.0,
            patience=100, verbose=True, print_interval=50):
        optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        criterion = self._build_criterion()

        # Phase 1: warm-up
        self._init_node_weights_sbrs(data)
        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")
        for _ in range(self.warm_up):
            self.train_step(data, optimizer, criterion)

        # 전환점: 노드 가중치 + loss scale 동시 고정
        if verbose:
            print(f"[{self.name}] Updating FIPS weights & calibrating loss scales...")
        self._update_node_weights_fips(data)
        self._calibrate_loss_scales(data, criterion)
        self.model.train()

        # Phase 2: 본 학습
        best_val_score = -float("inf")
        best_state     = copy.deepcopy(self.model.state_dict())
        counter        = 0
        remaining      = epochs - self.warm_up

        if verbose:
            print(f"[{self.name}] Phase 2: main training {remaining} epochs...")

        for epoch in range(remaining):
            train_info = self.train_step(data, optimizer, criterion)
            val_result = evaluate_pyg_model(
                self.model, data, split="val", task_type=self.task_type)
            val_score = self._compute_val_score(val_result)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state     = copy.deepcopy(self.model.state_dict())
                counter        = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                train_result = evaluate_pyg_model(
                    self.model, data, split="train", task_type=self.task_type)
                print(
                    f"[{self.name}] "
                    f"Epoch {epoch + self.warm_up + 1:04d} | "
                    f"Total {train_info['total_loss']:.4f} | "
                    f"Task {train_info['task_loss']:.4f} | "
                    f"Struct {train_info['struct_loss']:.4f} | "
                    f"Rep {train_info['rep_loss']:.4f} | "
                    f"Out {train_info['out_loss']:.4f} | "
                    f"Train {train_result} | "
                    f"Val {val_result} | "
                    f"ValScore {val_score:.4f}"
                )

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch "
                          f"{epoch + self.warm_up + 1}.")
                break

        self.model.load_state_dict(best_state)
        if verbose:
            print(f"[{self.name}] Training finished. "
                  f"Best val score: {best_val_score:.4f}")


# =========================================================
# SUMMIT-C / SUMMIT-R
# =========================================================
class FairGate_C(NodeAwareBase):
    """
    SUMMIT: Structural risk and Uncertainty-guided
            Multi-level bias MITigation — Classification
    """
    def __init__(
        self, in_feats, h_feats, device,
        name="GCN", dropout=0.1, sgc_k=2,
        lambda_fair=0.5,
        drop_edge_rate_struct=0.1,
        ablate_struct=False, ablate_rep=False, ablate_out=False,
        val_tradeoff_dp=0.5, val_tradeoff_eo=0.5,
        sbrs_threshold=0.5, lam=1.0,
        min_weight=0.5, max_weight=2.0,
        warm_up=100,
        ablate_sbrs=False, ablate_uncertainty=False,
    ):
        super().__init__(
            task_type="classification", device=device,
            lambda_fair=lambda_fair,
            sbrs_threshold=sbrs_threshold, lam=lam,
            min_weight=min_weight, max_weight=max_weight,
            warm_up=warm_up,
            ablate_sbrs=ablate_sbrs, ablate_uncertainty=ablate_uncertainty,
        )
        assert name in ["GCN", "GraphSAGE", "SGC"]
        self.name                  = f"{name}/SUMMIT-C"
        self.in_feats              = in_feats
        self.h_feats               = h_feats
        self.backbone_name         = name
        self.dropout               = dropout
        self.sgc_k                 = sgc_k
        self.drop_edge_rate_struct = drop_edge_rate_struct
        self.ablate_struct         = ablate_struct
        self.ablate_rep            = ablate_rep
        self.ablate_out            = ablate_out
        self.val_tradeoff_dp       = val_tradeoff_dp
        self.val_tradeoff_eo       = val_tradeoff_eo
        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(self.backbone_name, self.in_feats,
                              self.h_feats, self.dropout, self.sgc_k)

    def _build_criterion(self):
        return nn.BCEWithLogitsLoss()

    def compute_output_loss(self, logits, labels, sensitive_attr, idx_fair):
        return weighted_classification_fairness_loss(
            torch.sigmoid(logits), labels, sensitive_attr,
            self._node_weight, idx=idx_fair)

    def _compute_val_score(self, val_result):
        acc = float(val_result.get("acc", 0.0))
        dp  = abs(float(val_result.get("dp", 0.0)))
        eo  = abs(float(val_result.get("eo", 0.0)))
        return acc - self.val_tradeoff_dp * dp - self.val_tradeoff_eo * eo

class FairGate_R(NodeAwareBase):
    """
    SUMMIT: Structural risk and Uncertainty-guided
            Multi-level bias MITigation — Regression
    """
    def __init__(
        self, in_feats, h_feats, device,
        name="GCN", dropout=0.1, sgc_k=2,
        lambda_fair=0.5,
        drop_edge_rate_struct=0.1,
        ablate_struct=False, ablate_rep=False, ablate_out=False,
        val_tradeoff_mae=1.0, val_tradeoff_bias=1.0, val_tradeoff_mean_pred=0.5,
        sbrs_threshold=0.5, lam=1.0,
        min_weight=0.5, max_weight=2.0,
        warm_up=100,
        ablate_sbrs=False, ablate_uncertainty=False,
    ):
        super().__init__(
            task_type="regression", device=device,
            lambda_fair=lambda_fair,
            sbrs_threshold=sbrs_threshold, lam=lam,
            min_weight=min_weight, max_weight=max_weight,
            warm_up=warm_up,
            ablate_sbrs=ablate_sbrs, ablate_uncertainty=ablate_uncertainty,
        )
        assert name in ["GCN", "GraphSAGE", "SGC"]
        self.name                   = f"{name}/SUMMIT-R"
        self.in_feats               = in_feats
        self.h_feats                = h_feats
        self.backbone_name          = name
        self.dropout                = dropout
        self.sgc_k                  = sgc_k
        self.drop_edge_rate_struct  = drop_edge_rate_struct
        self.ablate_struct          = ablate_struct
        self.ablate_rep             = ablate_rep
        self.ablate_out             = ablate_out
        self.val_tradeoff_mae       = val_tradeoff_mae
        self.val_tradeoff_bias      = val_tradeoff_bias
        self.val_tradeoff_mean_pred = val_tradeoff_mean_pred
        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(self.backbone_name, self.in_feats,
                              self.h_feats, self.dropout, self.sgc_k)

    def _build_criterion(self):
        return nn.MSELoss()

    def compute_output_loss(self, preds, labels, sensitive_attr, idx_fair):
        return weighted_regression_fairness_loss(
            preds, labels, sensitive_attr,
            self._node_weight, idx=idx_fair)

    def _compute_val_score(self, val_result):
        mae           = float(val_result.get("mae", float("inf")))
        bias_gap      = abs(float(val_result.get("bias_gap", 0.0)))
        mean_pred_gap = abs(float(val_result.get("mean_pred_gap", 0.0)))
        return -(
            self.val_tradeoff_mae         * mae
            + self.val_tradeoff_bias      * bias_gap
            + self.val_tradeoff_mean_pred * mean_pred_gap
        )
