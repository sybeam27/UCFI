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

    adj = adj.tocoo()
    edge_index = torch.tensor(
        [adj.row, adj.col],
        dtype=torch.long,
        device=device,
    )

    x = dataset_dict["features"].float().to(device)

    if task_type == "classification":
        y = dataset_dict["labels"].long().view(-1).to(device)
    elif task_type == "regression":
        y = dataset_dict["labels"].float().view(-1).to(device)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    sensitive_attr = dataset_dict["sens"].long().view(-1).to(device)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        sensitive_attr=sensitive_attr,
    )
    data.idx_train = dataset_dict["idx_train"].long().to(device)
    data.idx_val = dataset_dict["idx_val"].long().to(device)
    data.idx_test = dataset_dict["idx_test"].long().to(device)

    if "idx_sens_train" in dataset_dict:
        data.idx_sens_train = dataset_dict["idx_sens_train"].long().to(device)

    return data


# =========================================================
# Backbone Models
# =========================================================
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_dim=1, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x = data.x
        edge_index = data.edge_index if edge_index is None else edge_index

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)

        if return_hidden:
            return out, h
        return out

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_dim=1, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x = data.x
        edge_index = data.edge_index if edge_index is None else edge_index

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)

        if return_hidden:
            return out, h
        return out

class SGC(nn.Module):
    def __init__(self, in_feats, out_dim=1, K=2):
        super().__init__()
        self.conv = SGConv(in_feats, out_dim, K=K)

    def forward(self, data, edge_index=None, return_hidden=False):
        x = data.x
        edge_index = data.edge_index if edge_index is None else edge_index

        out = self.conv(x, edge_index).view(-1)

        # hidden representation이 별도로 없으므로 입력 특징을 대체 사용
        if return_hidden:
            return out, x
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


# =========================================================
# Fairness Regularizers
# =========================================================
class GroupWiseNorm(nn.Module):
    """
    그룹 간 평균/분산 차이를 줄이는 differentiable regularizer
    representation / scalar output 둘 다 처리 가능
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, sensitive_attr, idx=None):
        if idx is not None:
            z = z[idx]
            sensitive_attr = sensitive_attr[idx]

        if z.dim() == 1:
            z = z.unsqueeze(1)

        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)

        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return z.new_tensor(0.0)

        z0 = z[mask_0]
        z1 = z[mask_1]

        mean0 = z0.mean(dim=0)
        mean1 = z1.mean(dim=0)

        var0 = z0.var(dim=0, unbiased=False)
        var1 = z1.var(dim=0, unbiased=False)

        mean_diff = torch.abs(mean0 - mean1).mean()
        var_diff = torch.abs(var0 - var1).mean()

        return mean_diff + var_diff


def differentiable_dp_loss(prob, sensitive_attr, idx=None):
    if idx is not None:
        prob = prob[idx]
        sensitive_attr = sensitive_attr[idx]

    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)

    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return prob.new_tensor(0.0)

    p0 = prob[mask_0].mean()
    p1 = prob[mask_1].mean()
    return torch.abs(p0 - p1)

def differentiable_eo_loss(prob, labels, sensitive_attr, idx=None):
    if idx is not None:
        prob = prob[idx]
        labels = labels[idx]
        sensitive_attr = sensitive_attr[idx]

    labels = labels.float()

    mask_0_y1 = (sensitive_attr == 0) & (labels == 1)
    mask_1_y1 = (sensitive_attr == 1) & (labels == 1)

    if mask_0_y1.sum() == 0 or mask_1_y1.sum() == 0:
        return prob.new_tensor(0.0)

    eo0 = prob[mask_0_y1].mean()
    eo1 = prob[mask_1_y1].mean()
    return torch.abs(eo0 - eo1)

def regression_output_fairness_loss(preds, labels, sensitive_attr, idx=None):
    """
    회귀용 output-level fairness surrogate
    - mean_pred_gap
    - bias_gap
    """
    if idx is not None:
        preds = preds[idx]
        labels = labels[idx]
        sensitive_attr = sensitive_attr[idx]

    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)

    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return preds.new_tensor(0.0)

    pred0 = preds[mask_0]
    pred1 = preds[mask_1]
    y0 = labels[mask_0]
    y1 = labels[mask_1]

    mean_pred_gap = torch.abs(pred0.mean() - pred1.mean())

    resid0 = pred0 - y0
    resid1 = pred1 - y1
    bias_gap = torch.abs(resid0.mean() - resid1.mean())

    return mean_pred_gap + bias_gap


# =========================================================
# Base Multi-Level Fair GNN
# =========================================================
class BaseMultiLevelFairGNN:
    """
    공통 설계:
    - structure loss
    - representation loss
    - output loss
    를 공유하고,
    task-specific 부분만 override
    """
    def __init__(self, task_type, device):
        assert task_type in ["classification", "regression"]
        self.task_type = task_type
        self.device = device
        self.group_norm = GroupWiseNorm()

    # ---------- required override ----------
    def _build_model(self):
        raise NotImplementedError

    def _build_criterion(self):
        raise NotImplementedError

    def compute_output_loss(self, preds_or_logits, labels, sensitive_attr, idx_fair):
        raise NotImplementedError

    def _compute_val_score(self, val_result):
        raise NotImplementedError

    # ---------- shared ----------
    def _build_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
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
        device = edge_index.device

        keep_mask = torch.rand(num_edges, device=device) > drop_rate
        if keep_mask.sum() == 0:
            keep_mask[torch.randint(0, num_edges, (1,), device=device)] = True

        return edge_index[:, keep_mask]

    def compute_structure_loss(self, data, h_orig):
        edge_pert = self.perturb_edge_index(
            data.edge_index,
            drop_rate=self.drop_edge_rate_struct,
        )
        _, h_pert = self.model(data, edge_index=edge_pert, return_hidden=True)
        return F.mse_loss(h_orig, h_pert)

    def compute_representation_loss(self, h, sensitive_attr, idx_fair):
        return self.group_norm(h, sensitive_attr, idx=idx_fair)

    def train_step(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        labels = data.y.float()
        idx_train = data.idx_train
        idx_fair = self._get_fair_idx(data)
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
                preds_or_logits,
                labels,
                sensitive_attr,
                idx_fair,
            )

        total_loss = (
            task_loss
            + self.lambda_struct * struct_loss
            + self.lambda_rep * rep_loss
            + self.lambda_out * out_loss
        )

        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "struct_loss": float(struct_loss.item()),
            "rep_loss": float(rep_loss.item()),
            "out_loss": float(out_loss.item()),
        }

    def fit(
        self,
        data,
        epochs=1000,
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
                break

        self.model.load_state_dict(best_state)

        if verbose:
            print(f"[{self.name}] Training finished. Best val score: {best_val_score:.4f}")

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model,
            data,
            split=split,
            task_type=self.task_type,
        )

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
        logits = self.predict(data)
        return torch.sigmoid(logits)


# =========================================================
# FnCGNN: Multi-Level Fairness for Classification
# =========================================================
class FnCGNN(BaseMultiLevelFairGNN):
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        name="GCN",
        dropout=0.1,
        sgc_k=2,
        lambda_struct=0.01,
        lambda_rep=0.05,
        lambda_out=0.05,
        drop_edge_rate_struct=0.15,
        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,
        val_tradeoff_dp=0.5,
        val_tradeoff_eo=0.5,
    ):
        super().__init__(task_type="classification", device=device)

        assert name in ["GCN", "GraphSAGE", "SGC"], "지원하지 않는 모델입니다."
        self.name = f"{name}/FnCGNN"

        self.in_feats = in_feats
        self.h_feats = h_feats
        self.backbone_name = name
        self.dropout = dropout
        self.sgc_k = sgc_k

        self.lambda_struct = lambda_struct
        self.lambda_rep = lambda_rep
        self.lambda_out = lambda_out
        self.drop_edge_rate_struct = drop_edge_rate_struct

        self.ablate_struct = ablate_struct
        self.ablate_rep = ablate_rep
        self.ablate_out = ablate_out

        self.val_tradeoff_dp = val_tradeoff_dp
        self.val_tradeoff_eo = val_tradeoff_eo

        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(
            name=self.backbone_name,
            in_feats=self.in_feats,
            h_feats=self.h_feats,
            dropout=self.dropout,
            sgc_k=self.sgc_k,
        )

    def _build_criterion(self):
        return nn.BCEWithLogitsLoss()

    def compute_output_loss(self, logits, labels, sensitive_attr, idx_fair):
        prob = torch.sigmoid(logits)
        dp_loss = differentiable_dp_loss(prob, sensitive_attr, idx=idx_fair)
        eo_loss = differentiable_eo_loss(prob, labels, sensitive_attr, idx=idx_fair)
        return dp_loss + eo_loss

    def _compute_val_score(self, val_result):
        acc = float(val_result.get("acc", 0.0))
        dp = abs(float(val_result.get("dp", 0.0)))
        eo = abs(float(val_result.get("eo", 0.0)))
        return acc - self.val_tradeoff_dp * dp - self.val_tradeoff_eo * eo


# =========================================================
# FnRGNN: Multi-Level Fairness for Regression
# =========================================================
class FnRGNN(BaseMultiLevelFairGNN):
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        name="GCN",
        dropout=0.1,
        sgc_k=2,
        lambda_struct=0.01,
        lambda_rep=0.05,
        lambda_out=0.05,
        drop_edge_rate_struct=0.15,
        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,
        val_tradeoff_mae=1.0,
        val_tradeoff_bias=0.5,
        val_tradeoff_mean_pred=0.5,
    ):
        super().__init__(task_type="regression", device=device)

        assert name in ["GCN", "GraphSAGE", "SGC"], "지원하지 않는 모델입니다."
        self.name = f"{name}/FnRGNN"

        self.in_feats = in_feats
        self.h_feats = h_feats
        self.backbone_name = name
        self.dropout = dropout
        self.sgc_k = sgc_k

        self.lambda_struct = lambda_struct
        self.lambda_rep = lambda_rep
        self.lambda_out = lambda_out
        self.drop_edge_rate_struct = drop_edge_rate_struct

        self.ablate_struct = ablate_struct
        self.ablate_rep = ablate_rep
        self.ablate_out = ablate_out

        self.val_tradeoff_mae = val_tradeoff_mae
        self.val_tradeoff_bias = val_tradeoff_bias
        self.val_tradeoff_mean_pred = val_tradeoff_mean_pred

        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(
            name=self.backbone_name,
            in_feats=self.in_feats,
            h_feats=self.h_feats,
            dropout=self.dropout,
            sgc_k=self.sgc_k,
        )

    def _build_criterion(self):
        return nn.MSELoss()

    def compute_output_loss(self, preds, labels, sensitive_attr, idx_fair):
        return regression_output_fairness_loss(
            preds,
            labels,
            sensitive_attr,
            idx=idx_fair,
        )

    def _compute_val_score(self, val_result):
        mae = float(val_result.get("mae", float("inf")))
        bias_gap = abs(float(val_result.get("bias_gap", 0.0)))
        mean_pred_gap = abs(float(val_result.get("mean_pred_gap", 0.0)))

        return -(
            self.val_tradeoff_mae * mae
            + self.val_tradeoff_bias * bias_gap
            + self.val_tradeoff_mean_pred * mean_pred_gap
        )


# =========================================================
# Optional Helpers
# =========================================================
def train_fncgnn_model(
    data,
    nfeat,
    hidden_dim=64,
    device="cpu",
    backbone="GCN",
    dropout=0.1,
    sgc_k=2,
    lambda_struct=0.01,
    lambda_rep=0.05,
    lambda_out=0.05,
    drop_edge_rate_struct=0.15,
    lr=1e-3,
    weight_decay=0.0,
    epochs=300,
    patience=50,
    verbose=True,
):
    model = FnCGNN(
        in_feats=nfeat,
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,
        lambda_struct=lambda_struct,
        lambda_rep=lambda_rep,
        lambda_out=lambda_out,
        drop_edge_rate_struct=drop_edge_rate_struct,
    )

    model.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=verbose,
    )

    final_test = model.evaluate(data, split="test")
    return model, final_test

def train_fnrgnn_model(
    data,
    nfeat,
    hidden_dim=64,
    device="cpu",
    backbone="GCN",
    dropout=0.1,
    sgc_k=2,
    lambda_struct=0.01,
    lambda_rep=0.05,
    lambda_out=0.05,
    drop_edge_rate_struct=0.15,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    patience=50,
    verbose=True,
):
    model = FnRGNN(
        in_feats=nfeat,
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,
        lambda_struct=lambda_struct,
        lambda_rep=lambda_rep,
        lambda_out=lambda_out,
        drop_edge_rate_struct=drop_edge_rate_struct,
    )

    model.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=verbose,
    )

    final_test = model.evaluate(data, split="test")
    return model, final_test



# # 나중에 살펴보기
# # =========================================================
# # FIPS Utilities
# # =========================================================
# def compute_static_structural_risk(data):
#     """
#     정적 구조 위험도 계산.
 
#     구성 요소:
#       - degree centrality:  허브 노드일수록 편향 전파력이 큼
#       - boundary score:     이웃 중 다른 민감 속성 그룹 비율
#                             → 그룹 경계 노드일수록 fairness 개입이 필요
 
#     반환: [N] float tensor, 값 범위 [0, 1]
#     """
#     edge_index = data.edge_index
#     sens       = data.sensitive_attr
#     N          = data.x.size(0)
#     device     = data.x.device
 
#     ones = torch.ones(edge_index.size(1), device=device)
 
#     # degree
#     deg = torch.zeros(N, device=device)
#     deg.scatter_add_(0, edge_index[0], ones)
#     deg_norm = (deg - deg.min()) / (deg.max() - deg.min() + 1e-8)
 
#     # cross-group edge 비율 (boundary score)
#     src, dst   = edge_index
#     cross      = (sens[src] != sens[dst]).float()
#     boundary   = torch.zeros(N, device=device)
#     boundary.scatter_add_(0, src, cross)
#     boundary   = boundary / (deg + 1e-8)
 
#     risk = 0.5 * deg_norm + 0.5 * boundary
#     return risk  # [N]
 
 
# # =========================================================
# # FIPS-Guided Multi-Level Fair GNN
# # =========================================================
# class FIPS_MFGNN(BaselineGNN):
#     """
#     FIPS(Fairness Intervention Priority Score) 기반 선택적 fairness 개입 모델.
 
#     핵심 아이디어
#     -------------
#     1. Static Structural Risk  : 그래프 구조(degree, boundary)로 고위험 노드 탐지
#     2. Dynamic Structural Uncertainty : 학습 중 edge perturbation을 T회 반복하여
#                                         representation 분산(= 구조적 불안정성) 추정
#     3. FIPS = α·risk + β·uncertainty  (softmax 정규화 후 노드별 가중치로 변환)
#     4. Multi-level Selective Intervention
#        - Structure level : FIPS 가중 consistency loss (고위험 노드 → 더 강한 안정성 요구)
#        - Representation  : FIPS 가중 hidden representation GroupWiseNorm
#        - Output level    : FIPS 가중 DP/EO surrogate loss
 
#     파라미터
#     --------
#     fips_alpha          : static risk 반영 비중
#     fips_beta           : dynamic uncertainty 반영 비중
#     fips_temperature    : softmax temperature (높을수록 고위험 노드에 더 집중)
#     fips_T              : uncertainty 추정용 perturbation 반복 횟수
#     fips_update_interval: FIPS를 매 N epoch마다 재계산 (계산 비용 절감)
#     lambda_struct/rep/out: 각 level 손실 가중치
#     drop_edge_rate_struct: edge dropout 비율
#     val_tradeoff_sp/eo  : validation composite score에서 fairness 반영 비중
#     ablate_struct/rep/out: 각 level 개입 비활성화 (ablation study용)
#     """
 
#     def __init__(
#         self,
#         in_feats,
#         h_feats,
#         device,
#         name="GCN",
#         dropout=0.1,
#         sgc_k=2,
#         # FIPS hyperparameters
#         fips_alpha=0.5,
#         fips_beta=0.5,
#         fips_temperature=5.0,
#         fips_T=10,
#         fips_update_interval=5,
#         # Multi-level loss weights
#         lambda_struct=0.01,
#         lambda_rep=0.05,
#         lambda_out=0.05,
#         drop_edge_rate_struct=0.15,
#         # Ablation switches
#         ablate_struct=False,
#         ablate_rep=False,
#         ablate_out=False,
#         ablate_fips=False,          # True → uniform weight (기존 MultiFairGNN과 동일)
#         ablate_uncertainty=False,    # True → static risk만 사용 (dynamic uncertainty 제거)
#         ablate_static_risk=False,    # True → dynamic uncertainty만 사용 (static risk 제거)
#         # Validation selection
#         val_tradeoff_sp=0.5,
#         val_tradeoff_eo=0.5,
#     ):
#         super().__init__(
#             in_feats=in_feats,
#             h_feats=h_feats,
#             device=device,
#             name=name,
#             dropout=dropout,
#             sgc_k=sgc_k,
#         )
 
#         # FIPS config
#         self.fips_alpha            = fips_alpha
#         self.fips_beta             = fips_beta
#         self.fips_temperature      = fips_temperature
#         self.fips_T                = fips_T
#         self.fips_update_interval  = fips_update_interval
 
#         # Loss weights
#         self.lambda_struct         = lambda_struct
#         self.lambda_rep            = lambda_rep
#         self.lambda_out            = lambda_out
#         self.drop_edge_rate_struct = drop_edge_rate_struct
 
#         # Ablation flags
#         self.ablate_struct         = ablate_struct
#         self.ablate_rep            = ablate_rep
#         self.ablate_out            = ablate_out
#         self.ablate_fips           = ablate_fips
#         self.ablate_uncertainty    = ablate_uncertainty
#         self.ablate_static_risk    = ablate_static_risk
 
#         # Validation composite score weights
#         self.val_tradeoff_sp       = val_tradeoff_sp
#         self.val_tradeoff_eo       = val_tradeoff_eo
 
#         # Fairness module
#         self.group_norm = FnC_GroupWiseNorm()
 
#         # FIPS cache
#         self._fips_weight: torch.Tensor | None = None
#         self._fips_epoch_counter: int          = 0
 
#         print(
#             f"[Model Init] FIPS_MFGNN | backbone:{name} | "
#             f"struct:{not ablate_struct}, rep:{not ablate_rep}, out:{not ablate_out} | "
#             f"fips:{not ablate_fips} | uncertainty:{not ablate_uncertainty} | "
#             f"static_risk:{not ablate_static_risk} | "
#             f"α={fips_alpha}, β={fips_beta}, T={fips_T}, τ={fips_temperature} | "
#             f"update_interval={fips_update_interval}"
#         )
 
#     # ------------------------------------------------------------------
#     # Edge perturbation helper
#     # ------------------------------------------------------------------
#     @staticmethod
#     def perturb_edge_index(edge_index: torch.Tensor, drop_rate: float = 0.15) -> torch.Tensor:
#         if drop_rate <= 0.0:
#             return edge_index
 
#         num_edges = edge_index.size(1)
#         keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
 
#         if not keep_mask.any():
#             keep_mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
 
#         return edge_index[:, keep_mask]
 
#     # ------------------------------------------------------------------
#     # Dynamic Structural Uncertainty
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _estimate_structural_uncertainty(self, data: Data) -> torch.Tensor:
#         """
#         Edge dropout을 T번 반복 → hidden representation의 분산 → 노드별 불안정성.
 
#         반환: [N] float tensor (정규화 전 raw variance)
#         """
#         self.model.eval()
#         h_list = []
 
#         for _ in range(self.fips_T):
#             edge_pert = self.perturb_edge_index(
#                 data.edge_index, drop_rate=self.drop_edge_rate_struct
#             )
#             _, h = self.model(data, edge_index=edge_pert, return_hidden=True)
#             h_list.append(h)  # [N, d]
 
#         h_stack = torch.stack(h_list, dim=0)          # [T, N, d]
#         uncertainty = h_stack.var(dim=0).mean(dim=-1)  # [N]
#         return uncertainty
 
#     # ------------------------------------------------------------------
#     # Static Structural Risk
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def _compute_static_risk(self, data: Data) -> torch.Tensor:
#         """
#         compute_static_structural_risk 모듈 함수를 래핑.
#         반환: [N] float tensor, [0, 1] 정규화됨
#         """
#         return compute_static_structural_risk(data)
 
#     # ------------------------------------------------------------------
#     # FIPS Score & Weight
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def compute_fips(self, data: Data) -> torch.Tensor:
#         """
#         FIPS = α·static_risk + β·dynamic_uncertainty
 
#         ablation 플래그에 따라 구성 요소를 선택적으로 포함.
#         softmax(τ · FIPS) * N → 합이 N이고 평균이 ≈1인 가중치 벡터 반환.
 
#         반환: [N] float tensor
#         """
#         N      = data.x.size(0)
#         device = data.x.device
 
#         # ── ablate_fips: 모든 노드 동일 가중치 (uniform baseline)
#         if self.ablate_fips:
#             return torch.ones(N, device=device)
 
#         # ── 구성 요소 계산
#         if self.ablate_static_risk:
#             static_risk = torch.zeros(N, device=device)
#         else:
#             static_risk = self._compute_static_risk(data)
 
#         if self.ablate_uncertainty:
#             dyn_unc = torch.zeros(N, device=device)
#         else:
#             dyn_unc = self._estimate_structural_uncertainty(data)
 
#         # ── min-max 정규화
#         def _norm(v: torch.Tensor) -> torch.Tensor:
#             mn, mx = v.min(), v.max()
#             return (v - mn) / (mx - mn + 1e-8)
 
#         static_risk = _norm(static_risk)
#         dyn_unc     = _norm(dyn_unc)
 
#         fips = self.fips_alpha * static_risk + self.fips_beta * dyn_unc  # [N]
 
#         # ── softmax 정규화 → 평균 ≈ 1 스케일
#         fips_weight = torch.softmax(fips * self.fips_temperature, dim=0) * N
#         return fips_weight  # [N]
 
#     def _get_fips_weight(self, data: Data) -> torch.Tensor:
#         """
#         FIPS cache 관리.
#         fips_update_interval 마다 재계산, 그 사이에는 캐시 사용.
#         """
#         if (
#             self._fips_weight is None
#             or self._fips_epoch_counter % self.fips_update_interval == 0
#         ):
#             self._fips_weight = self.compute_fips(data)
#         return self._fips_weight
 
#     # ------------------------------------------------------------------
#     # Loss Components
#     # ------------------------------------------------------------------
#     def _structure_loss(
#         self, data: Data, h_orig: torch.Tensor, fips_weight: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Structure-level: FIPS 가중 representation consistency loss.
#         고위험 노드의 표현이 edge 교란에 더 안정적이어야 한다는 제약.
 
#         L_struct = mean( fips_weight * ||h_orig - h_pert||² )
#         """
#         edge_pert = self.perturb_edge_index(
#             data.edge_index, drop_rate=self.drop_edge_rate_struct
#         )
#         _, h_pert       = self.model(data, edge_index=edge_pert, return_hidden=True)
#         node_sq_err     = (h_orig - h_pert).pow(2).mean(dim=-1)  # [N]
#         weighted_err    = fips_weight * node_sq_err               # [N]
#         return weighted_err.mean()
 
#     def _representation_loss(
#         self,
#         h: torch.Tensor,
#         sensitive_attr: torch.Tensor,
#         idx_train: torch.Tensor,
#         fips_weight: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Representation-level: FIPS 가중 GroupWiseNorm.
#         고위험 노드의 representation이 그룹 분포 정렬에 더 많이 기여.
 
#         h_weighted = h * fips_weight (broadcast)
#         """
#         h_weighted = h * fips_weight.unsqueeze(1)  # [N, d]
#         return self.group_norm(h_weighted, sensitive_attr, idx=idx_train)
 
#     def _output_loss(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         sensitive_attr: torch.Tensor,
#         idx_train: torch.Tensor,
#         fips_weight: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Output-level: FIPS 가중 DP/EO surrogate loss.
#         고위험 노드의 예측값 편향이 더 강하게 규제됨.
 
#         weight_train = fips_weight[idx_train] (train split 한정)
#         """
#         prob         = torch.sigmoid(logits)
#         weight_train = fips_weight[idx_train]  # [|idx_train|]
 
#         dp_loss = differentiable_dp_loss(
#             prob, sensitive_attr, idx=idx_train
#         )
#         eo_loss = differentiable_eo_loss(
#             prob, labels, sensitive_attr, idx=idx_train
#         )
        
#         return dp_loss + eo_loss
 
#     # ------------------------------------------------------------------
#     # Train Step (override)
#     # ------------------------------------------------------------------
#     def train_step(
#         self,
#         data: Data,
#         optimizer: torch.optim.Optimizer,
#         criterion: nn.Module,
#     ) -> dict:
#         # ── 1. FIPS 가중치 (eval 모드로 추정, 캐시 활용)
#         fips_weight = self._get_fips_weight(data)  # [N], detached
#         self._fips_epoch_counter += 1
 
#         # ── 2. Forward (train 모드)
#         self.model.train()
#         optimizer.zero_grad()
 
#         logits, h = self.model(data, return_hidden=True)
#         labels    = data.y.float()
#         idx_train = data.idx_train
#         sensitive = data.sensitive_attr
 
#         # ── 3. Task loss
#         task_loss = criterion(logits[idx_train], labels[idx_train])
 
#         # ── 4. Structure-level loss
#         struct_loss = logits.new_tensor(0.0)
#         if not self.ablate_struct:
#             struct_loss = self._structure_loss(data, h, fips_weight)
 
#         # ── 5. Representation-level loss
#         rep_loss = logits.new_tensor(0.0)
#         if not self.ablate_rep:
#             rep_loss = self._representation_loss(h, sensitive, idx_train, fips_weight)
 
#         # ── 6. Output-level loss
#         out_loss = logits.new_tensor(0.0)
#         if not self.ablate_out:
#             out_loss = self._output_loss(logits, labels, sensitive, idx_train, fips_weight)
 
#         # ── 7. Total loss
#         total_loss = (
#             task_loss
#             + self.lambda_struct * struct_loss
#             + self.lambda_rep    * rep_loss
#             + self.lambda_out    * out_loss
#         )
 
#         total_loss.backward()
#         optimizer.step()
 
#         return {
#             "total_loss":  float(total_loss.item()),
#             "task_loss":   float(task_loss.item()),
#             "struct_loss": float(struct_loss.item()) if not self.ablate_struct else 0.0,
#             "rep_loss":    float(rep_loss.item())    if not self.ablate_rep    else 0.0,
#             "out_loss":    float(out_loss.item())    if not self.ablate_out    else 0.0,
#         }
 
#     # ------------------------------------------------------------------
#     # Validation Composite Score
#     # ------------------------------------------------------------------
#     def _compute_val_score(self, val_result: dict) -> tuple:
#         """
#         val_score = acc - α·|SP| - β·|EO|   (높을수록 좋음)
#         """
#         val_acc = float(val_result.get("acc", 0.0))
#         val_sp  = abs(float(val_result.get("global_sp", 0.0)))
#         val_eo  = abs(float(val_result.get("global_eo", 0.0)))
 
#         val_score = (
#             val_acc
#             - self.val_tradeoff_sp * val_sp
#             - self.val_tradeoff_eo * val_eo
#         )
#         return val_score, val_acc, val_sp, val_eo
 
#     # ------------------------------------------------------------------
#     # Fit (override: composite early stopping + FIPS cache 초기화)
#     # ------------------------------------------------------------------
#     def fit(
#         self,
#         data: Data,
#         epochs: int = 1000,
#         lr: float = 1e-3,
#         weight_decay: float = 0.0,
#         patience: int = 50,
#         verbose: bool = True,
#     ) -> None:
#         # FIPS cache 초기화
#         self._fips_weight         = None
#         self._fips_epoch_counter  = 0
 
#         optimizer      = self._build_optimizer(lr=lr, weight_decay=weight_decay)
#         criterion      = nn.BCEWithLogitsLoss()
#         best_val_score = -float("inf")
#         best_state     = copy.deepcopy(self.model.state_dict())
#         counter        = 0
 
#         for epoch in range(epochs):
#             train_info = self.train_step(data, optimizer, criterion)
#             val_result = evaluate_pyg_model(self.model, data, split="val")
#             val_score, val_acc, val_sp, val_eo = self._compute_val_score(val_result)
 
#             if val_score > best_val_score:
#                 best_val_score = val_score
#                 best_state     = copy.deepcopy(self.model.state_dict())
#                 counter        = 0
#             else:
#                 counter += 1
 
#             if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
#                 train_result = evaluate_pyg_model(self.model, data, split="train")
#                 print(
#                     f"[FIPS_MFGNN] Epoch {epoch+1:04d} | "
#                     f"Loss {train_info['total_loss']:.4f} "
#                     f"(task={train_info['task_loss']:.4f} "
#                     f"str={train_info['struct_loss']:.4f} "
#                     f"rep={train_info['rep_loss']:.4f} "
#                     f"out={train_info['out_loss']:.4f}) | "
#                     f"Train Acc {train_result.get('acc', 0.0):.4f} | "
#                     f"Val Acc {val_acc:.4f} | "
#                     f"Val SP {val_sp:.4f} | "
#                     f"Val EO {val_eo:.4f} | "
#                     f"Val Score {val_score:.4f}"
#                 )
 
#             if counter >= patience:
#                 if verbose:
#                     print(f"[FIPS_MFGNN] Early stopping at epoch {epoch+1}.")
#                 break
 
#         self.model.load_state_dict(best_state)
#         if verbose:
#             print(
#                 f"[FIPS_MFGNN] Training finished. "
#                 f"Best composite val score: {best_val_score:.4f}"
#             )
 
#     # ------------------------------------------------------------------
#     # FIPS 분석 유틸리티
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def get_fips_analysis(self, data: Data) -> dict:
#         """
#         FIPS 구성 요소와 최종 가중치를 반환.
#         분석 및 시각화용 메서드.
 
#         반환: {
#             'static_risk':   [N] tensor,
#             'uncertainty':   [N] tensor,
#             'fips_raw':      [N] tensor (softmax 이전),
#             'fips_weight':   [N] tensor (softmax 이후),
#         }
#         """
#         N      = data.x.size(0)
#         device = data.x.device
 
#         static_risk = (
#             torch.zeros(N, device=device)
#             if self.ablate_static_risk
#             else self._compute_static_risk(data)
#         )
#         dyn_unc = (
#             torch.zeros(N, device=device)
#             if self.ablate_uncertainty
#             else self._estimate_structural_uncertainty(data)
#         )
 
#         def _norm(v):
#             mn, mx = v.min(), v.max()
#             return (v - mn) / (mx - mn + 1e-8)
 
#         static_risk_n = _norm(static_risk)
#         dyn_unc_n     = _norm(dyn_unc)
 
#         fips_raw    = self.fips_alpha * static_risk_n + self.fips_beta * dyn_unc_n
#         fips_weight = torch.softmax(fips_raw * self.fips_temperature, dim=0) * N
 
#         # model을 train 모드로 복귀
#         self.model.train()
 
#         return {
#             "static_risk":  static_risk_n.cpu(),
#             "uncertainty":  dyn_unc_n.cpu(),
#             "fips_raw":     fips_raw.cpu(),
#             "fips_weight":  fips_weight.cpu(),
#         }
 