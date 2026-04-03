import copy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, SGConv
from utils.metrics import evaluate_pyg_model




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
        var0  = z0.var(dim=0, unbiased=False)
        var1  = z1.var(dim=0, unbiased=False)

        mean_diff = torch.abs(mean0 - mean1).mean()
        var_diff  = torch.abs(var0  - var1 ).mean()

        # ↓ 추가: dim 평균이므로 이미 정규화됨, 추가 /dim 불필요
        # mean_diff, var_diff는 이미 dim 평균 → 스케일 안정적
        return mean_diff + var_diff  # 변경 없음 — 이미 정상

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
        device = edge_index.device

        keep_mask = torch.rand(num_edges, device=device) > drop_rate
        if keep_mask.sum() == 0:
            keep_mask[torch.randint(0, num_edges, (1,), device=device)] = True

        return edge_index[:, keep_mask]
    
    def compute_structure_loss(self, data, h_orig):
        idx_fair  = self._get_fair_idx(data)
        edge_pert = self.perturb_edge_index(
            data.edge_index,
            drop_rate=self.drop_edge_rate_struct,
        )
        _, h_pert = self.model(data, edge_index=edge_pert, return_hidden=True)
        
        # hidden dim으로 나눠서 스케일 정규화
        # MSE 평균이 dim에 비례하므로 dim으로 나누면 항상 ~O(1) 스케일
        dim = h_orig.size(-1)
        return F.mse_loss(h_orig[idx_fair], h_pert[idx_fair]) / dim

    def compute_representation_loss(self, h, sensitive_attr, idx_fair):
        h_train   = h[idx_fair]                      # train 노드만 명시적으로 분리
        sens_train = sensitive_attr[idx_fair]        # 대응하는 sensitive attr도 분리
        return self.group_norm(h_train, sens_train)  # idx 인자 불필요

    def train_step(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        labels        = data.y.float()
        idx_train     = data.idx_train
        idx_fair      = self._get_fair_idx(data)
        sensitive_attr = data.sensitive_attr

        preds_or_logits, h = self.model(data, return_hidden=True)
        task_loss = criterion(preds_or_logits[idx_train], labels[idx_train])

        struct_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_struct:
            struct_loss = self.compute_structure_loss(data, h)
            # 별도 스케일링 불필요 — compute_structure_loss 내부에서 처리

        rep_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_rep:
            rep_loss = self.compute_representation_loss(h, sensitive_attr, idx_fair)
            # rep loss (GroupWiseNorm)도 dim 정규화 적용

        out_loss = preds_or_logits.new_tensor(0.0)
        if not self.ablate_out:
            out_loss = self.compute_output_loss(
                preds_or_logits, labels, sensitive_attr, idx_fair
            )

        total_loss = (
            task_loss
            + self.lambda_struct * struct_loss
            + self.lambda_rep    * rep_loss
            + self.lambda_out    * out_loss
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
        mae           = float(val_result.get("mae", float("inf")))
        bias_gap      = abs(float(val_result.get("bias_gap", 0.0)))
        mean_pred_gap = abs(float(val_result.get("mean_pred_gap", 0.0)))
        return -(
            self.val_tradeoff_mae      * mae
            + self.val_tradeoff_bias   * bias_gap
            + self.val_tradeoff_mean_pred * mean_pred_gap  # 주석 해제
        )





# =========================================================
# Structural Bias Risk Score (SBRS) Computation
# =========================================================
def compute_sbrs(data, alpha=0.34, beta=0.33, gamma=0.33):
    """
    그래프 구조 기반 Structural Bias Risk Score (SBRS) 계산.

    GNN 편향은 노드의 구조적 위치에 따라 세 가지 독립적인 경로로 전파됨:

      [경로 1] Propagation Influence (Dong et al., AAAI 2023):
        허브 노드는 L-hop 전파에서 influence score가 높아
        편향된 representation을 더 많은 이웃에 전파함.
        → w_degree = normalize(log(1 + deg(v)))

      [경로 2] Cross-group Information Leakage (Dai & Wang, WSDM 2021; Dong et al., WWW 2022):
        경계 노드는 두 sensitive group 사이에서 편향 정보를 중개함.
        → w_boundary = normalize(boundary_ratio(v))

      [경로 3] Local Homophily Deviation (Loveland et al., SDM 2025):
        Global homophily와 크게 다른 local homophily를 가진 노드는
        OOD 취약성으로 인해 편향 예측에 노출됨.
        → w_lhd = normalize(|local_h(v) - global_h|)

    SBRS(v) = alpha * w_degree(v) + beta * w_boundary(v) + gamma * w_lhd(v)

    반환: sbrs_n [N] float tensor in [0, 1] (detached, 정규화됨)
    """
    edge_index = data.edge_index
    sens       = data.sensitive_attr
    N          = data.x.size(0)
    device     = data.x.device

    ones     = torch.ones(edge_index.size(1), device=device)
    src, dst = edge_index

    # ── degree
    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, ones)

    # ── [경로 1] Propagation Influence
    log_deg  = torch.log1p(deg)
    w_degree = (log_deg - log_deg.min()) / (log_deg.max() - log_deg.min() + 1e-8)

    # ── [경로 2] Cross-group Leakage
    cross_edge     = (sens[src] != sens[dst]).float()
    cross_count    = torch.zeros(N, device=device)
    cross_count.scatter_add_(0, src, cross_edge)
    boundary_ratio = cross_count / (deg + 1e-8)
    w_boundary     = (boundary_ratio - boundary_ratio.min()) / \
                     (boundary_ratio.max() - boundary_ratio.min() + 1e-8)

    # ── [경로 3] Local Homophily Deviation
    same_edge   = (sens[src] == sens[dst]).float()
    same_count  = torch.zeros(N, device=device)
    same_count.scatter_add_(0, src, same_edge)
    local_h     = same_count / (deg + 1e-8)
    global_h    = local_h.mean()
    lhd         = torch.abs(local_h - global_h)
    w_lhd       = (lhd - lhd.min()) / (lhd.max() - lhd.min() + 1e-8)

    # ── SBRS 정규화
    sbrs   = alpha * w_degree + beta * w_boundary + gamma * w_lhd
    sbrs_n = (sbrs - sbrs.min()) / (sbrs.max() - sbrs.min() + 1e-8)

    return sbrs_n.detach()


# =========================================================
# Structural Uncertainty Estimation
# =========================================================
@torch.no_grad()
def estimate_structural_uncertainty(model, data, T=10, drop_rate=0.1,
                                    task_type="classification"):
    """
    Edge perturbation을 T회 반복하여 각 노드의 예측 확률 분산을
    structural uncertainty로 추정.

    "구조가 조금 흔들려도 예측이 크게 변하는가"를 측정.
    warm-up 학습 후 1회 호출하여 캐시.

    반환: w_unc [N] float tensor in [0, 1] (정규화됨, detached)
    """
    model.eval()
    prob_list = []

    for _ in range(T):
        edge_index = data.edge_index
        num_edges  = edge_index.size(1)
        keep_mask  = torch.rand(num_edges, device=edge_index.device) > drop_rate
        if keep_mask.sum() == 0:
            keep_mask[0] = True
        edge_pert = edge_index[:, keep_mask]

        out = model(data, edge_index=edge_pert)
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(-1)

        if task_type == "classification":
            prob = torch.sigmoid(out)
        else:
            prob = out   # 회귀: 예측값 자체의 분산

        prob_list.append(prob)

    prob_stack  = torch.stack(prob_list, dim=0)   # [T, N]
    uncertainty = prob_stack.var(dim=0)            # [N]

    # [0, 1] 정규화
    w_unc = (uncertainty - uncertainty.min()) / \
            (uncertainty.max() - uncertainty.min() + 1e-8)

    return w_unc.detach()


# =========================================================
# 2단계 게이팅 기반 최종 노드 가중치 계산
# =========================================================
def compute_node_risk_weights(
    data,
    model=None,
    alpha=0.34,
    beta=0.33,
    gamma=0.33,
    sbrs_threshold=0.5,   # 1단계 게이팅 임계값 (SBRS 정규화 기준)
    lam=1.0,              # 2단계 uncertainty 반영 강도
    min_weight=0.5,
    max_weight=2.0,
    T=10,
    drop_rate=0.1,
    task_type="classification",
):
    """
    2단계 게이팅 기반 FIPS(Fairness Intervention Priority Score) 노드 가중치.

    1단계 — 구조적 위험 노드 식별 (SBRS 기반):
      SBRS(v) ≥ sbrs_threshold 인 노드만 fairness 개입 대상으로 선정.
      구조적 조건이 없는 노드는 uncertainty가 높아도 개입 불필요.

    2단계 — 불확실성으로 최종 가중치 조정:
      개입 대상 노드: weight(v) = scale(SBRS(v) × (1 + λ·w_unc(v)))
      비대상 노드:    weight(v) = min_weight

    model=None이면 SBRS만 사용 (Phase 1 warm-up 중).
    model이 주어지면 uncertainty를 추가 반영 (Phase 2 진입 시).

    반환: weight [N] float tensor (detached)
    """
    N      = data.x.size(0)
    device = data.x.device

    # ── 1단계: SBRS 계산
    sbrs_n = compute_sbrs(data, alpha=alpha, beta=beta, gamma=gamma)

    # ── model 없으면 SBRS만으로 가중치 반환 (warm-up용)
    if model is None:
        weight = min_weight + (max_weight - min_weight) * sbrs_n
        return weight.detach()

    # ── 2단계: structural uncertainty 추정
    w_unc = estimate_structural_uncertainty(
        model, data, T=T, drop_rate=drop_rate, task_type=task_type
    )

    # ── 게이팅 적용
    weight    = torch.full((N,), min_weight, device=device)
    gate_mask = sbrs_n >= sbrs_threshold   # 1단계 통과 노드

    if gate_mask.sum() > 0:
        # SBRS × (1 + λ·uncertainty) → 정규화 → 스케일
        combined   = sbrs_n[gate_mask] * (1.0 + lam * w_unc[gate_mask])
        combined_n = (combined - combined.min()) / \
                     (combined.max() - combined.min() + 1e-8)
        weight[gate_mask] = min_weight + (max_weight - min_weight) * combined_n

    return weight.detach()


# =========================================================
# Weighted Fairness Loss Functions
# =========================================================
class WeightedGroupWiseNorm(nn.Module):
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

def weighted_dp_loss(prob, sensitive_attr, weight, idx=None):
    if idx is not None:
        prob           = prob[idx]
        sensitive_attr = sensitive_attr[idx]
        weight         = weight[idx]

    weight = weight / (weight.mean() + 1e-8)
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)

    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return prob.new_tensor(0.0)

    p0 = (prob[mask_0] * weight[mask_0]).sum() / (weight[mask_0].sum() + 1e-8)
    p1 = (prob[mask_1] * weight[mask_1]).sum() / (weight[mask_1].sum() + 1e-8)
    return torch.abs(p0 - p1)

def weighted_eo_loss(prob, labels, sensitive_attr, weight, idx=None):
    if idx is not None:
        prob           = prob[idx]
        labels         = labels[idx]
        sensitive_attr = sensitive_attr[idx]
        weight         = weight[idx]

    labels = labels.float()
    weight = weight / (weight.mean() + 1e-8)

    mask_0_y1 = (sensitive_attr == 0) & (labels == 1)
    mask_1_y1 = (sensitive_attr == 1) & (labels == 1)

    if mask_0_y1.sum() == 0 or mask_1_y1.sum() == 0:
        return prob.new_tensor(0.0)

    eo0 = (prob[mask_0_y1] * weight[mask_0_y1]).sum() / (weight[mask_0_y1].sum() + 1e-8)
    eo1 = (prob[mask_1_y1] * weight[mask_1_y1]).sum() / (weight[mask_1_y1].sum() + 1e-8)
    return torch.abs(eo0 - eo1)

def weighted_regression_fairness_loss(preds, labels, sensitive_attr, weight, idx=None):
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

    mean_pred0    = (pred0 * w0).sum() / (w0.sum() + 1e-8)
    mean_pred1    = (pred1 * w1).sum() / (w1.sum() + 1e-8)
    mean_pred_gap = torch.abs(mean_pred0 - mean_pred1)

    bias0    = ((pred0 - y0) * w0).sum() / (w0.sum() + 1e-8)
    bias1    = ((pred1 - y1) * w1).sum() / (w1.sum() + 1e-8)
    bias_gap = torch.abs(bias0 - bias1)

    return mean_pred_gap + bias_gap


# =========================================================
# NodeAwareBase: FIPS 기반 2단계 게이팅 베이스 클래스
# =========================================================
class NodeAwareBase(BaseMultiLevelFairGNN):
    """
    BaseMultiLevelFairGNN을 상속하여
    FIPS(Fairness Intervention Priority Score) 기반
    2단계 게이팅 차등 fairness 개입을 추가한 베이스 클래스.

    설계 원칙:
      Phase 1 (warm-up): SBRS만으로 균일하게 학습하여 모델 안정화
      Phase 2 (main):    SBRS × structural uncertainty 2단계 게이팅으로
                         fairness-critical 노드에 집중 개입

    2단계 게이팅:
      1단계 — SBRS ≥ threshold: 구조적 위험 노드만 개입 대상 선정
      2단계 — SBRS × (1 + λ·uncertainty): 실제 불안정한 노드 우선

    변경된 부분 (BaseMultiLevelFairGNN 대비):
      - fit(): warm-up → uncertainty 추정 → 가중치 갱신 → 본 학습
      - compute_structure_loss: FIPS weighted MSE
      - compute_representation_loss: WeightedGroupWiseNorm
      - compute_output_loss: subclass에서 FIPS weighted loss 사용
    """
    def __init__(
        self,
        task_type,
        device,
        # SBRS 하이퍼파라미터
        alpha=0.34,           # 경로 1: Propagation Influence 비중
        beta=0.33,            # 경로 2: Cross-group Leakage 비중
        gamma=0.33,           # 경로 3: Local Homophily Deviation 비중
        # 2단계 게이팅 하이퍼파라미터
        sbrs_threshold=0.5,   # 1단계 임계값 (SBRS 정규화 기준, 0.5=상위50%)
        lam=1.0,              # 2단계 uncertainty 반영 강도
        # 가중치 범위
        min_weight=0.5,
        max_weight=2.0,
        # uncertainty 추정 설정
        warm_up=100,          # uncertainty 추정 전 선행 학습 epoch
        unc_T=10,             # perturbation 반복 횟수
        unc_drop=0.1,         # edge dropout 비율
    ):
        super().__init__(task_type=task_type, device=device)

        # SBRS
        self.alpha          = alpha
        self.beta           = beta
        self.gamma          = gamma
        # 게이팅
        self.sbrs_threshold = sbrs_threshold
        self.lam            = lam
        # 가중치 범위
        self.min_weight     = min_weight
        self.max_weight     = max_weight
        # warm-up / uncertainty
        self.warm_up        = warm_up
        self.unc_T          = unc_T
        self.unc_drop       = unc_drop

        # weighted group norm
        self.group_norm     = WeightedGroupWiseNorm()

        # 노드 가중치 캐시
        self._node_weight: torch.Tensor | None = None

    # ── Phase 1: SBRS만으로 초기 가중치 설정 (warm-up용)
    def _init_node_weights_sbrs(self, data):
        self._node_weight = compute_node_risk_weights(
            data,
            model      = None,   # uncertainty 없이 SBRS만
            alpha      = self.alpha,
            beta       = self.beta,
            gamma      = self.gamma,
            min_weight = self.min_weight,
            max_weight = self.max_weight,
        )
        w = self._node_weight
        print(
            f"[{self.name}] Phase 1 NodeWeight (SBRS only) | "
            f"min={w.min():.3f} max={w.max():.3f} "
            f"mean={w.mean():.3f} std={w.std():.3f}"
        )

    # ── Phase 2: SBRS × uncertainty 2단계 게이팅으로 가중치 갱신
    def _update_node_weights_fips(self, data):
        N      = data.x.size(0)
        device = data.x.device

        self._node_weight = compute_node_risk_weights(
            data,
            model           = self.model,
            alpha           = self.alpha,
            beta            = self.beta,
            gamma           = self.gamma,
            sbrs_threshold  = self.sbrs_threshold,
            lam             = self.lam,
            min_weight      = self.min_weight,
            max_weight      = self.max_weight,
            T               = self.unc_T,
            drop_rate       = self.unc_drop,
            task_type       = self.task_type,
        )

        w         = self._node_weight
        gate_mask = w > self.min_weight
        print(
            f"[{self.name}] Phase 2 NodeWeight (FIPS) | "
            f"gated={gate_mask.sum().item()}/{N} "
            f"({100*gate_mask.float().mean().item():.1f}%) | "
            f"min={w.min():.3f} max={w.max():.3f} "
            f"mean={w.mean():.3f} std={w.std():.3f}"
        )

    # ── Structure loss override (FIPS weighted MSE)
    def compute_structure_loss(self, data, h_orig):
        idx_fair  = self._get_fair_idx(data)
        edge_pert = self.perturb_edge_index(
            data.edge_index,
            drop_rate=self.drop_edge_rate_struct,
        )
        _, h_pert = self.model(data, edge_index=edge_pert, return_hidden=True)

        dim      = h_orig.size(-1)
        w        = self._node_weight[idx_fair]
        w        = w / (w.mean() + 1e-8)
        node_mse = ((h_orig[idx_fair] - h_pert[idx_fair]) ** 2).mean(dim=-1)
        return (node_mse * w).mean() / dim

    # ── Representation loss override (WeightedGroupWiseNorm)
    def compute_representation_loss(self, h, sensitive_attr, idx_fair):
        return self.group_norm(
            h, sensitive_attr,
            weight=self._node_weight,
            idx=idx_fair,
        )

    # ── fit() override: 2단계 학습 구조
    def fit(
        self,
        data,
        epochs=1000,
        lr=1e-3,
        weight_decay=0.0,
        patience=100,
        verbose=True,
        print_interval=50,
    ):
        optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        criterion = self._build_criterion()

        # ── Phase 1: warm-up (SBRS만으로 학습)
        self._init_node_weights_sbrs(data)

        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")

        for epoch in range(self.warm_up):
            self.train_step(data, optimizer, criterion)

        # ── uncertainty 추정 및 FIPS 가중치 갱신
        if verbose:
            print(
                f"[{self.name}] Estimating structural uncertainty "
                f"(T={self.unc_T}, drop={self.unc_drop})..."
            )
        self._update_node_weights_fips(data)
        self.model.train()

        # ── Phase 2: 본 학습 (FIPS 가중치 적용)
        best_val_score = -float("inf")
        best_state     = copy.deepcopy(self.model.state_dict())
        counter        = 0
        remaining      = epochs - self.warm_up

        if verbose:
            print(f"[{self.name}] Phase 2: main training {remaining} epochs...")

        for epoch in range(remaining):
            train_info = self.train_step(data, optimizer, criterion)
            val_result = evaluate_pyg_model(
                self.model, data, split="val", task_type=self.task_type
            )
            val_score = self._compute_val_score(val_result)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state     = copy.deepcopy(self.model.state_dict())
                counter        = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                train_result = evaluate_pyg_model(
                    self.model, data, split="train", task_type=self.task_type
                )
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
                    print(
                        f"[{self.name}] Early stopping at epoch "
                        f"{epoch + self.warm_up + 1}."
                    )
                break

        self.model.load_state_dict(best_state)
        if verbose:
            print(
                f"[{self.name}] Training finished. "
                f"Best val score: {best_val_score:.4f}"
            )


# =========================================================
# NAFnCGNN: Node-Aware Fair Classification GNN
# =========================================================
class NAFnCGNN(NodeAwareBase):
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        name="GCN",
        dropout=0.1,
        sgc_k=2,
        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,
        drop_edge_rate_struct=0.1,
        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,
        val_tradeoff_dp=0.3,
        val_tradeoff_eo=0.3,
        # SBRS 하이퍼파라미터
        alpha=0.34,           # 경로 1: Propagation Influence (degree) 비중
        beta=0.33,            # 경로 2: Cross-group Leakage (boundary ratio) 비중
        gamma=0.33,           # 경로 3: Local Homophily Deviation 비중
        # 2단계 게이팅 하이퍼파라미터
        sbrs_threshold=0.5,   # 1단계 임계값
        lam=1.0,              # uncertainty 반영 강도
        # 가중치 범위
        min_weight=0.5,
        max_weight=2.0,
        # warm-up / uncertainty 설정
        warm_up=100,
        unc_T=10,
        unc_drop=0.1,
    ):
        super().__init__(
            task_type       = "classification",
            device          = device,
            alpha           = alpha,
            beta            = beta,
            gamma           = gamma,
            sbrs_threshold  = sbrs_threshold,
            lam             = lam,
            min_weight      = min_weight,
            max_weight      = max_weight,
            warm_up         = warm_up,
            unc_T           = unc_T,
            unc_drop        = unc_drop,
        )

        assert name in ["GCN", "GraphSAGE", "SGC"], "지원하지 않는 모델입니다."
        self.name = f"{name}/NAFnCGNN"

        self.in_feats             = in_feats
        self.h_feats              = h_feats
        self.backbone_name        = name
        self.dropout              = dropout
        self.sgc_k                = sgc_k
        self.lambda_struct        = lambda_struct
        self.lambda_rep           = lambda_rep
        self.lambda_out           = lambda_out
        self.drop_edge_rate_struct = drop_edge_rate_struct
        self.ablate_struct        = ablate_struct
        self.ablate_rep           = ablate_rep
        self.ablate_out           = ablate_out
        self.val_tradeoff_dp      = val_tradeoff_dp
        self.val_tradeoff_eo      = val_tradeoff_eo

        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(
            name     = self.backbone_name,
            in_feats = self.in_feats,
            h_feats  = self.h_feats,
            dropout  = self.dropout,
            sgc_k    = self.sgc_k,
        )

    def _build_criterion(self):
        return nn.BCEWithLogitsLoss()

    def compute_output_loss(self, logits, labels, sensitive_attr, idx_fair):
        prob    = torch.sigmoid(logits)
        dp_loss = weighted_dp_loss(prob, sensitive_attr, self._node_weight, idx=idx_fair)
        eo_loss = weighted_eo_loss(prob, labels, sensitive_attr, self._node_weight, idx=idx_fair)
        return dp_loss + eo_loss

    def _compute_val_score(self, val_result):
        acc = float(val_result.get("acc", 0.0))
        dp  = abs(float(val_result.get("dp", 0.0)))
        eo  = abs(float(val_result.get("eo", 0.0)))
        return acc - self.val_tradeoff_dp * dp - self.val_tradeoff_eo * eo

# =========================================================
# NAFnRGNN: Node-Aware Fair Regression GNN
# =========================================================
class NAFnRGNN(NodeAwareBase):
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        name="GCN",
        dropout=0.1,
        sgc_k=2,
        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,
        drop_edge_rate_struct=0.1,
        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,
        val_tradeoff_mae=1.0,
        val_tradeoff_bias=1.0,
        val_tradeoff_mean_pred=0.5,
        # SBRS 하이퍼파라미터
        alpha=0.34,           # 경로 1: Propagation Influence (degree) 비중
        beta=0.33,            # 경로 2: Cross-group Leakage (boundary ratio) 비중
        gamma=0.33,           # 경로 3: Local Homophily Deviation 비중
        # 2단계 게이팅 하이퍼파라미터
        sbrs_threshold=0.5,
        lam=1.0,
        # 가중치 범위
        min_weight=0.5,
        max_weight=2.0,
        # warm-up / uncertainty 설정
        warm_up=100,
        unc_T=10,
        unc_drop=0.1,
    ):
        super().__init__(
            task_type       = "regression",
            device          = device,
            alpha           = alpha,
            beta            = beta,
            gamma           = gamma,
            sbrs_threshold  = sbrs_threshold,
            lam             = lam,
            min_weight      = min_weight,
            max_weight      = max_weight,
            warm_up         = warm_up,
            unc_T           = unc_T,
            unc_drop        = unc_drop,
        )

        assert name in ["GCN", "GraphSAGE", "SGC"], "지원하지 않는 모델입니다."
        self.name = f"{name}/NAFnRGNN"

        self.in_feats              = in_feats
        self.h_feats               = h_feats
        self.backbone_name         = name
        self.dropout               = dropout
        self.sgc_k                 = sgc_k
        self.lambda_struct         = lambda_struct
        self.lambda_rep            = lambda_rep
        self.lambda_out            = lambda_out
        self.drop_edge_rate_struct = drop_edge_rate_struct
        self.ablate_struct         = ablate_struct
        self.ablate_rep            = ablate_rep
        self.ablate_out            = ablate_out
        self.val_tradeoff_mae      = val_tradeoff_mae
        self.val_tradeoff_bias     = val_tradeoff_bias
        self.val_tradeoff_mean_pred = val_tradeoff_mean_pred

        self.model = self._build_model().to(device)

    def _build_model(self):
        return build_backbone(
            name     = self.backbone_name,
            in_feats = self.in_feats,
            h_feats  = self.h_feats,
            dropout  = self.dropout,
            sgc_k    = self.sgc_k,
        )

    def _build_criterion(self):
        return nn.MSELoss()

    def compute_output_loss(self, preds, labels, sensitive_attr, idx_fair):
        return weighted_regression_fairness_loss(
            preds, labels, sensitive_attr, self._node_weight, idx=idx_fair
        )

    def _compute_val_score(self, val_result):
        mae           = float(val_result.get("mae", float("inf")))
        bias_gap      = abs(float(val_result.get("bias_gap", 0.0)))
        mean_pred_gap = abs(float(val_result.get("mean_pred_gap", 0.0)))
        return -(
            self.val_tradeoff_mae       * mae
            + self.val_tradeoff_bias    * bias_gap
            + self.val_tradeoff_mean_pred * mean_pred_gap
        )





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
 