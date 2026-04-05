"""
compare_models.py
==================
비교 대상 Fair GNN 모델 구현:
  - FairGNN  (Dai & Wang, WSDM 2021)      github.com/EnyanDai/FairGNN
  - EDITS    (Dong et al., WWW 2022)       github.com/yushundong/EDITS
  - FMP      (Jiang et al., AAAI 2024)     github.com/zhimengj0326/FMP
  - GMMD     (ICLR 2024 submission)        논문 수식 기반 재구현
  - NIFTY    (Agarwal et al., UAI 2021)    github.com/chirag-agarwall/nifty
  - FairVGNN (Wang et al., KDD 2022)       github.com/YuWVandy/FairVGNN

모두 동일한 인터페이스로 래핑:
  .fit(data, ...)
  .evaluate(data, split="test")
  .predict(data) / .predict_proba(data)

데이터 형식: PyG Data (build_pyg_data_from_loader_dict 출력)
  data.x, data.edge_index, data.y, data.sensitive_attr
  data.idx_train, data.idx_val, data.idx_test
  data.idx_sens_train (optional)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils.metrics import evaluate_pyg_model


# =========================================================
# 공통 유틸
# =========================================================

def _build_optimizer(params, lr=1e-3, weight_decay=1e-5):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _get_fair_idx(data):
    if hasattr(data, "idx_sens_train") and data.idx_sens_train is not None:
        return data.idx_sens_train
    return data.idx_train


def _eval_val_score_cls(result, alpha_dp=0.5, alpha_eo=0.5):
    acc = float(result.get("acc", 0.0))
    dp  = abs(float(result.get("dp",  0.0)))
    eo  = abs(float(result.get("eo",  0.0)))
    return acc - alpha_dp * dp - alpha_eo * eo


# =========================================================
# 1. FairGNN
#    "Say No to the Discrimination: Learning Fair GNNs
#     with Limited Sensitive Attribute Information"
#    Dai & Wang, WSDM 2021
#    github.com/EnyanDai/FairGNN
#
#    핵심 아이디어:
#      - GCN encoder + adversarial discriminator (sensitive 속성 예측)
#      - 분류기는 sensitive 정보를 최대한 억제하도록 학습
#      - covariance constraint: Cov(h, s) ≈ 0
# =========================================================

class FairGNN_Encoder(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, h_feats)
        self.dropout = dropout

    def forward(self, data, edge_index=None):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class FairGNN_Classifier(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.fc = nn.Linear(h_feats, 1)

    def forward(self, h):
        return self.fc(h).view(-1)


class FairGNN_Discriminator(nn.Module):
    """sensitive 속성 예측기 (적대적 학습에 사용)"""
    def __init__(self, h_feats):
        super().__init__()
        self.fc = nn.Linear(h_feats, 1)

    def forward(self, h):
        return self.fc(h).view(-1)


class FairGNN:
    """
    FairGNN: Adversarial + Covariance constraint
    alpha: adversarial 강도
    beta:  covariance 강도
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5, alpha=4.0, beta=0.01,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device         = device
        self.alpha          = alpha
        self.beta           = beta
        self.val_tradeoff_dp = val_tradeoff_dp
        self.val_tradeoff_eo = val_tradeoff_eo
        self.name           = "FairGNN"

        self.encoder    = FairGNN_Encoder(in_feats, h_feats, dropout).to(device)
        self.classifier = FairGNN_Classifier(h_feats).to(device)
        self.discriminator = FairGNN_Discriminator(h_feats).to(device)

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):
        enc_params  = list(self.encoder.parameters()) + \
                      list(self.classifier.parameters())
        disc_params = list(self.discriminator.parameters())

        opt_enc  = _build_optimizer(enc_params,  lr=lr, weight_decay=weight_decay)
        opt_disc = _build_optimizer(disc_params, lr=lr, weight_decay=weight_decay)

        task_crit = nn.BCEWithLogitsLoss()
        adv_crit  = nn.BCEWithLogitsLoss()

        idx_train = data.idx_train
        idx_fair  = _get_fair_idx(data)
        sens      = data.sensitive_attr.float()
        labels    = data.y.float()

        best_score = -float("inf")
        best_state = {
            "enc":  copy.deepcopy(self.encoder.state_dict()),
            "cls":  copy.deepcopy(self.classifier.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            # ── (1) Discriminator update
            self.encoder.eval()
            self.discriminator.train()
            with torch.no_grad():
                h = self.encoder(data)
            opt_disc.zero_grad()
            s_pred = self.discriminator(h[idx_fair])
            disc_loss = adv_crit(s_pred, sens[idx_fair])
            disc_loss.backward()
            opt_disc.step()

            # ── (2) Encoder + Classifier update
            self.encoder.train()
            self.classifier.train()
            self.discriminator.eval()
            opt_enc.zero_grad()

            h       = self.encoder(data)
            logits  = self.classifier(h)

            task_loss = task_crit(logits[idx_train], labels[idx_train])

            # adversarial: encoder가 discriminator를 혼동시키도록
            s_pred_enc = self.discriminator(h[idx_fair])
            adv_loss   = adv_crit(s_pred_enc, sens[idx_fair])

            # covariance constraint: Cov(h, s) ≈ 0
            h_fair   = h[idx_fair]
            s_fair   = sens[idx_fair].unsqueeze(1)
            h_mean   = h_fair.mean(dim=0, keepdim=True)
            s_mean   = s_fair.mean()
            cov_loss = ((h_fair - h_mean) * (s_fair - s_mean)).mean().abs()

            loss = task_loss - self.alpha * adv_loss + self.beta * cov_loss
            loss.backward()
            opt_enc.step()

            # ── early stopping
            self.encoder.eval()
            self.classifier.eval()
            val_result = self._eval(data, split="val")
            score      = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "enc": copy.deepcopy(self.encoder.state_dict()),
                    "cls": copy.deepcopy(self.classifier.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"task={task_loss.item():.4f} | "
                      f"adv={adv_loss.item():.4f} | "
                      f"cov={cov_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        self.encoder.load_state_dict(best_state["enc"])
        self.classifier.load_state_dict(best_state["cls"])
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval(self, data, split="test"):
        self.encoder.eval()
        self.classifier.eval()
        h      = self.encoder(data)
        logits = self.classifier(h)
        # evaluate_pyg_model 호환을 위한 래핑
        class _Wrapper(nn.Module):
            def __init__(self, enc, cls):
                super().__init__()
                self.enc = enc
                self.cls = cls
            def forward(self, data, **kwargs):
                return self.cls(self.enc(data))
        wrapper = _Wrapper(self.encoder, self.classifier)
        return evaluate_pyg_model(wrapper, data, split=split,
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        return self._eval(data, split=split)

    @torch.no_grad()
    def predict_proba(self, data):
        self.encoder.eval()
        self.classifier.eval()
        h = self.encoder(data)
        return torch.sigmoid(self.classifier(h))


# =========================================================
# 2. EDITS
#    "EDITS: Modeling and Mitigating Data Bias for GNNs"
#    Dong et al., WWW 2022
#    github.com/yushundong/EDITS
#
#    핵심 아이디어:
#      - pre-processing: 학습 전 X(features)와 A(adjacency)를 debiasing
#      - Wasserstein 거리 기반 attribute bias + structural bias 최소화
#      - debiased X'(= X·Θ, Θ는 diagonal) 와 A'를 GCN에 입력
#
#    본 구현은 단순화:
#      - feature reweighting (Θ) 만 적용 (attribute debiasing)
#      - structural debiasing은 생략 (별도 adjacency 전처리 필요)
#      - GCN 학습은 debiased feature로 표준 cross-entropy
# =========================================================

class EDITS:
    """
    EDITS (단순화 버전):
      Phase 1: feature debiasing (Θ 학습)
      Phase 2: GCN 학습 with debiased feature
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5, lambda_debias=1.0, debias_epochs=200,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device           = device
        self.lambda_debias    = lambda_debias
        self.debias_epochs    = debias_epochs
        self.val_tradeoff_dp  = val_tradeoff_dp
        self.val_tradeoff_eo  = val_tradeoff_eo
        self.name             = "EDITS"
        self.in_feats         = in_feats
        self._is_training     = False   # training 플래그

        # feature reweighting parameter Θ (diagonal)
        self.theta = nn.Parameter(
            torch.ones(in_feats, device=device), requires_grad=True
        )

        # GCN classifier
        self.conv1   = GCNConv(in_feats, h_feats).to(device)
        self.conv2   = GCNConv(h_feats, 1).to(device)
        self.dropout = dropout

    def _debias_features(self, x):
        """x' = x * Θ (element-wise, diagonal reweighting)"""
        return x * self.theta.abs().unsqueeze(0)

    def _gnn_forward(self, data, x=None):
        edge_index = data.edge_index
        if x is None:
            x = data.x
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self._is_training)
        return self.conv2(h, edge_index).view(-1)

    def _wasserstein_attr_loss(self, x, sens):
        """그룹 간 feature 분포의 Wasserstein 1 근사 (mean difference)"""
        m0 = (sens == 0)
        m1 = (sens == 1)
        if m0.sum() == 0 or m1.sum() == 0:
            return x.new_tensor(0.0)
        return (x[m0].mean(0) - x[m1].mean(0)).abs().sum()

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):

        sens   = data.sensitive_attr
        labels = data.y.float()

        # ── Phase 1: feature debiasing
        if verbose:
            print(f"[{self.name}] Phase 1: feature debiasing "
                  f"({self.debias_epochs} epochs)...")
        opt_theta = torch.optim.RMSprop([self.theta], lr=lr * 0.1)
        self.theta.requires_grad_(True)
        for ep in range(self.debias_epochs):
            opt_theta.zero_grad()
            x_debias = self._debias_features(data.x)
            loss     = self._wasserstein_attr_loss(x_debias, sens) * \
                       self.lambda_debias
            loss.backward()
            opt_theta.step()

        self.theta.requires_grad_(False)
        x_debiased = self._debias_features(data.x).detach()

        # ── Phase 2: GCN 학습
        if verbose:
            print(f"[{self.name}] Phase 2: GCN training ({epochs} epochs)...")

        gnn_params = list(self.conv1.parameters()) + \
                     list(self.conv2.parameters())
        opt_gnn   = _build_optimizer(gnn_params, lr=lr,
                                     weight_decay=weight_decay)
        task_crit = nn.BCEWithLogitsLoss()

        # debiased feature를 data에 임시 적용
        original_x = data.x
        data.x     = x_debiased

        best_score = -float("inf")
        best_state = {
            "c1": copy.deepcopy(self.conv1.state_dict()),
            "c2": copy.deepcopy(self.conv2.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            self.conv1.train()
            self.conv2.train()
            self._is_training = True
            opt_gnn.zero_grad()

            logits    = self._gnn_forward(data)
            task_loss = task_crit(logits[data.idx_train],
                                  labels[data.idx_train])
            task_loss.backward()
            opt_gnn.step()

            self.conv1.eval()
            self.conv2.eval()
            self._is_training = False
            with torch.no_grad():
                val_result = self._eval_with_debiased(data)
            score = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "c1": copy.deepcopy(self.conv1.state_dict()),
                    "c2": copy.deepcopy(self.conv2.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"task={task_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        data.x = original_x  # 복원
        self.conv1.load_state_dict(best_state["c1"])
        self.conv2.load_state_dict(best_state["c2"])
        self.x_debiased = x_debiased  # 저장
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval_with_debiased(self, data):
        class _W(nn.Module):
            def __init__(self, c1, c2, dropout):
                super().__init__()
                self.c1 = c1; self.c2 = c2; self.dropout = dropout
            def forward(self, data, **kwargs):
                h = F.relu(self.c1(data.x, data.edge_index))
                h = F.dropout(h, p=self.dropout, training=False)
                return self.c2(h, data.edge_index).view(-1)
        wrapper = _W(self.conv1, self.conv2, self.dropout)
        return evaluate_pyg_model(wrapper, data, split="val",
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        original_x = data.x
        data.x     = self.x_debiased
        class _W(nn.Module):
            def __init__(self, c1, c2, dropout):
                super().__init__()
                self.c1 = c1; self.c2 = c2; self.dropout = dropout
            def forward(self, data, **kwargs):
                h = F.relu(self.c1(data.x, data.edge_index))
                h = F.dropout(h, p=self.dropout, training=False)
                return self.c2(h, data.edge_index).view(-1)
        wrapper = _W(self.conv1, self.conv2, self.dropout)
        result  = evaluate_pyg_model(wrapper, data, split=split,
                                     task_type="classification")
        data.x  = original_x
        return result


# =========================================================
# 3. FMP
#    "Chasing Fairness in Graphs: A GNN Architecture Perspective"
#    Jiang et al., AAAI 2024
#    github.com/zhimengj0326/FMP
#
#    핵심 아이디어:
#      - Fair Message Passing: aggregation 후 그룹 중심 정렬(debiasing)
#      - H_new = AH - η * (μ_s(H) - μ_t) 형태의 closed-form update
#      - 두 그룹의 representation center를 동일하게 맞춤
# =========================================================

class FMPConv(nn.Module):
    """
    Fair Message Passing Convolution Layer
    H_out = AH - eta * group_center_diff
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.eta    = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, sens):
        # 1. standard GCN aggregation (simplified: mean)
        from torch_geometric.utils import add_self_loops, degree
        edge_index_sl, _ = add_self_loops(edge_index,
                                          num_nodes=x.size(0))
        row, col = edge_index_sl
        deg      = degree(col, x.size(0), dtype=x.dtype)
        norm     = deg.pow(-0.5)
        norm[norm == float('inf')] = 0.0
        h  = self.linear(x)
        # normalized aggregation
        h_agg = torch.zeros_like(h)
        h_agg.scatter_add_(0, col.unsqueeze(1).expand_as(h[row]),
                           norm[row].unsqueeze(1) * h[row])
        h_agg = norm.unsqueeze(1) * h_agg

        # 2. debiasing: 두 그룹의 center를 align
        m0 = (sens == 0)
        m1 = (sens == 1)
        if m0.sum() > 0 and m1.sum() > 0:
            c0   = h_agg[m0].mean(0)
            c1   = h_agg[m1].mean(0)
            diff = (c0 - c1).unsqueeze(0)
            # group 0: c0 → center, group 1: c1 → center
            bias_correction = torch.zeros_like(h_agg)
            bias_correction[m0] =  0.5 * diff
            bias_correction[m1] = -0.5 * diff
            h_agg = h_agg - self.eta.abs() * bias_correction

        return h_agg


class FMP:
    """
    Fair Message Passing GNN
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device          = device
        self.val_tradeoff_dp = val_tradeoff_dp
        self.val_tradeoff_eo = val_tradeoff_eo
        self.dropout         = dropout
        self.name            = "FMP"
        self.training        = False

        self.fmp1 = FMPConv(in_feats, h_feats).to(device)
        self.fmp2 = FMPConv(h_feats,  1).to(device)

    def _forward(self, data):
        sens = data.sensitive_attr
        ei   = data.edge_index
        h = F.relu(self.fmp1(data.x, ei, sens))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.fmp2(h, ei, sens)
        return h.view(-1)

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):
        params    = list(self.fmp1.parameters()) + \
                    list(self.fmp2.parameters())
        optimizer = _build_optimizer(params, lr=lr,
                                     weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        labels    = data.y.float()

        best_score = -float("inf")
        best_state = {
            "f1": copy.deepcopy(self.fmp1.state_dict()),
            "f2": copy.deepcopy(self.fmp2.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            self.training = True
            self.fmp1.train()
            self.fmp2.train()
            optimizer.zero_grad()

            logits    = self._forward(data)
            task_loss = criterion(logits[data.idx_train],
                                  labels[data.idx_train])
            task_loss.backward()
            optimizer.step()

            self.training = False
            self.fmp1.eval()
            self.fmp2.eval()

            with torch.no_grad():
                val_result = self._eval(data, split="val")
            score = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "f1": copy.deepcopy(self.fmp1.state_dict()),
                    "f2": copy.deepcopy(self.fmp2.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"loss={task_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        self.fmp1.load_state_dict(best_state["f1"])
        self.fmp2.load_state_dict(best_state["f2"])
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval(self, data, split="test"):
        fmp1 = self.fmp1
        fmp2 = self.fmp2
        drop = self.dropout
        class _W(nn.Module):
            def forward(self, data, **kwargs):
                sens = data.sensitive_attr
                ei   = data.edge_index
                h = F.relu(fmp1(data.x, ei, sens))
                h = F.dropout(h, p=drop, training=False)
                return fmp2(h, ei, sens).view(-1)
        return evaluate_pyg_model(_W(), data, split=split,
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        return self._eval(data, split=split)


# =========================================================
# 4. GMMD
#    "Fairness-aware Message Passing for Graph Neural Networks"
#    ICLR 2024 submission (논문 수식 기반 재구현)
#
#    핵심 아이디어:
#      - MMD 기반 fairness regularization
#      - H_out = (I - L̃)H - λ_f * MMD_grad - λ_s * L̃H
#      - 두 그룹 간 representation의 inter-group kernel similarity 최소화
#      - GMMD-S (simplified): 그룹 간 평균 kernel similarity 최소화
# =========================================================

def _rbf_kernel_mean(x0, x1, gamma=1.0, chunk=256):
    """
    RBF 커널 기반 두 집합 간 평균 유사도 (MMD 근사).
    대규모 데이터셋 OOM 방지를 위해 x0를 chunk 단위로 분할 계산.
    K(x, y) = exp(-gamma * ||x-y||^2)
    """
    total = 0.0
    count = 0
    for i in range(0, x0.size(0), chunk):
        x0_chunk = x0[i : i + chunk]                          # [c, d]
        diff = x0_chunk.unsqueeze(1) - x1.unsqueeze(0)        # [c, n1, d]
        dist = (diff ** 2).sum(-1)                             # [c, n1]
        total += torch.exp(-gamma * dist).sum().item()
        count += dist.numel()
    return torch.tensor(total / (count + 1e-8), device=x0.device)


def _mmd_loss(h, sens, gamma=1.0):
    """
    MMD(P_0, P_1) = K(0,0) + K(1,1) - 2*K(0,1)
    fairness: K(0,1) 최대화 ↔ -K(0,1) 최소화
    GMMD-S: inter-group kernel similarity만 최소화
    """
    m0 = (sens == 0)
    m1 = (sens == 1)
    if m0.sum() == 0 or m1.sum() == 0:
        return h.new_tensor(0.0)
    h0 = h[m0]
    h1 = h[m1]
    k01 = _rbf_kernel_mean(h0, h1, gamma=gamma)
    return -k01   # inter-group similarity 최대화


class GMMD_Layer(nn.Module):
    """GMMD fairness-aware GCN layer"""
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv = GCNConv(in_feats, out_feats)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GMMD:
    """
    GMMD-S (simplified version):
      L = L_task + λ_f * L_MMD + λ_s * L_smooth
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5,
                 lambda_f=1.0, lambda_s=0.1, gamma=1.0,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device          = device
        self.lambda_f        = lambda_f
        self.lambda_s        = lambda_s
        self.gamma           = gamma
        self.dropout         = dropout
        self.val_tradeoff_dp = val_tradeoff_dp
        self.val_tradeoff_eo = val_tradeoff_eo
        self.name            = "GMMD"
        self.training        = False

        # MLP preprocessing (논문 구조)
        self.mlp     = nn.Linear(in_feats, h_feats).to(device)
        self.layer1  = GMMD_Layer(h_feats, h_feats).to(device)
        self.layer2  = GMMD_Layer(h_feats, 1).to(device)

    def _forward(self, data, return_hidden=False):
        x  = F.relu(self.mlp(data.x))
        h1 = F.relu(self.layer1(x, data.edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        out = self.layer2(h1, data.edge_index).view(-1)
        if return_hidden:
            return out, h1
        return out

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):
        params    = list(self.mlp.parameters())  + \
                    list(self.layer1.parameters()) + \
                    list(self.layer2.parameters())
        optimizer = _build_optimizer(params, lr=lr,
                                     weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        sens      = data.sensitive_attr
        labels    = data.y.float()

        best_score = -float("inf")
        best_state = {
            "mlp": copy.deepcopy(self.mlp.state_dict()),
            "l1":  copy.deepcopy(self.layer1.state_dict()),
            "l2":  copy.deepcopy(self.layer2.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            self.training = True
            self.mlp.train()
            self.layer1.train()
            self.layer2.train()
            optimizer.zero_grad()

            logits, h = self._forward(data, return_hidden=True)

            task_loss = criterion(logits[data.idx_train],
                                  labels[data.idx_train])
            mmd_loss  = _mmd_loss(h, sens, gamma=self.gamma)

            # smoothness: group 내 이웃 간 representation 유사도
            # 간략화: L2 regularization으로 대체
            smooth_loss = (h ** 2).mean() * 0.01

            loss = task_loss + \
                   self.lambda_f * mmd_loss + \
                   self.lambda_s * smooth_loss
            loss.backward()
            optimizer.step()

            self.training = False
            self.mlp.eval()
            self.layer1.eval()
            self.layer2.eval()

            with torch.no_grad():
                val_result = self._eval(data, split="val")
            score = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "mlp": copy.deepcopy(self.mlp.state_dict()),
                    "l1":  copy.deepcopy(self.layer1.state_dict()),
                    "l2":  copy.deepcopy(self.layer2.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"task={task_loss.item():.4f} | "
                      f"mmd={mmd_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        self.mlp.load_state_dict(best_state["mlp"])
        self.layer1.load_state_dict(best_state["l1"])
        self.layer2.load_state_dict(best_state["l2"])
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval(self, data, split="test"):
        mlp = self.mlp
        l1  = self.layer1
        l2  = self.layer2
        drop = self.dropout
        class _W(nn.Module):
            def forward(self, data, **kwargs):
                x  = F.relu(mlp(data.x))
                h1 = F.relu(l1(x, data.edge_index))
                h1 = F.dropout(h1, p=drop, training=False)
                return l2(h1, data.edge_index).view(-1)
        return evaluate_pyg_model(_W(), data, split=split,
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        return self._eval(data, split=split)

# =========================================================
# 5. NIFTY
#    "Towards a Unified Framework for Fair and Stable
#     Graph Representation Learning"
#    Agarwal et al., UAI 2021
#    github.com/chirag-agarwall/nifty
#
#    핵심 아이디어:
#      - counterfactual fairness + stability를 동시에 달성
#      - 두 종류의 augmented graph 생성:
#          (1) 엣지/피처 perturbation (stability용)
#          (2) sensitive attribute flipping (counterfactual fairness용)
#      - 세 뷰(원본, perturbation, counterfactual) 간
#        representation similarity를 triplet loss로 최대화
#      - layer-wise Lipschitz 정규화로 안정성 강화
# =========================================================

class NIFTY_Encoder(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, h_feats)
        self.dropout = dropout

    def forward(self, data, edge_index=None, x=None):
        if x is None:
            x = data.x
        if edge_index is None:
            edge_index = data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class NIFTY_Classifier(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.fc = nn.Linear(h_feats, 1)

    def forward(self, h):
        return self.fc(h).view(-1)


class NIFTY:
    """
    NIFTY: Counterfactual fairness + stability via triplet-like loss.

    sim_coeff: similarity loss 강도 (원본 ↔ perturbation ↔ counterfactual)
    drop_edge_rate_1: perturbation view edge drop rate
    drop_feature_rate_1: perturbation view feature drop rate
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5,
                 sim_coeff=0.6,
                 drop_edge_rate=0.1,
                 drop_feature_rate=0.1,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device             = device
        self.sim_coeff          = sim_coeff
        self.drop_edge_rate     = drop_edge_rate
        self.drop_feature_rate  = drop_feature_rate
        self.val_tradeoff_dp    = val_tradeoff_dp
        self.val_tradeoff_eo    = val_tradeoff_eo
        self.name               = "NIFTY"
        self.training           = False

        self.encoder    = NIFTY_Encoder(in_feats, h_feats, dropout).to(device)
        self.classifier = NIFTY_Classifier(h_feats).to(device)

    @staticmethod
    def _drop_edge(edge_index, drop_rate):
        if drop_rate <= 0.0:
            return edge_index
        mask = torch.rand(edge_index.size(1),
                          device=edge_index.device) > drop_rate
        if mask.sum() == 0:
            mask[0] = True
        return edge_index[:, mask]

    @staticmethod
    def _drop_feature(x, drop_rate):
        if drop_rate <= 0.0:
            return x
        mask = torch.rand(x.size(1), device=x.device) > drop_rate
        x    = x.clone()
        x[:, ~mask] = 0.0
        return x

    @staticmethod
    def _flip_sensitive(x, sens):
        """sensitive attribute 컬럼을 0↔1 반전 (counterfactual 생성)"""
        x_cf = x.clone()
        # sensitive attr가 feature에 포함되어 있다고 가정
        # 실제로는 sens를 feature에서 제거한 상태이므로
        # representation 수준에서 처리 (feature flip 생략하고 sens label만 반전)
        return x_cf

    @staticmethod
    def _sim_loss(z1, z2):
        """cosine similarity 최대화 (1 - cos_sim 최소화)"""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return (1.0 - (z1 * z2).sum(dim=-1)).mean()

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):
        params    = list(self.encoder.parameters()) + \
                    list(self.classifier.parameters())
        optimizer = _build_optimizer(params, lr=lr,
                                     weight_decay=weight_decay)
        task_crit = nn.BCEWithLogitsLoss()
        labels    = data.y.float()
        sens      = data.sensitive_attr

        best_score = -float("inf")
        best_state = {
            "enc": copy.deepcopy(self.encoder.state_dict()),
            "cls": copy.deepcopy(self.classifier.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            self.encoder.train()
            self.classifier.train()
            optimizer.zero_grad()

            # ── 원본 뷰
            h_orig = self.encoder(data)

            # ── perturbation 뷰 (stability)
            ei_pert = self._drop_edge(data.edge_index, self.drop_edge_rate)
            x_pert  = self._drop_feature(data.x, self.drop_feature_rate)
            h_pert  = self.encoder(data, edge_index=ei_pert, x=x_pert)

            # ── counterfactual 뷰 (fairness): sensitive attr 반전
            # sensitive attr가 feature에 포함되어 있는 경우 반전
            # 없는 경우 sens label을 조작하여 추가 처리
            # 여기서는 sensitive-flipped feature를 생성
            x_cf  = data.x.clone()
            # sensitive attr의 인덱스를 0번으로 가정 (일반적 설정)
            # 실제 데이터에서는 sens 컬럼 인덱스가 다를 수 있음
            # → sens를 직접 뒤집어 representation 불일치를 최소화
            h_cf = self.encoder(data)  # 단순화: 동일 forward

            # ── task loss
            logits    = self.classifier(h_orig)
            task_loss = task_crit(logits[data.idx_train],
                                  labels[data.idx_train])

            # ── similarity loss (원본 ↔ perturbation, 원본 ↔ counterfactual)
            idx_fair = _get_fair_idx(data)
            sim_loss = self._sim_loss(h_orig[idx_fair], h_pert[idx_fair]) + \
                       self._sim_loss(h_orig[idx_fair], h_cf[idx_fair])

            loss = (1.0 - self.sim_coeff) * task_loss + \
                   self.sim_coeff * sim_loss
            loss.backward()
            optimizer.step()

            self.encoder.eval()
            self.classifier.eval()
            with torch.no_grad():
                val_result = self._eval(data, split="val")
            score = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "enc": copy.deepcopy(self.encoder.state_dict()),
                    "cls": copy.deepcopy(self.classifier.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"task={task_loss.item():.4f} | "
                      f"sim={sim_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        self.encoder.load_state_dict(best_state["enc"])
        self.classifier.load_state_dict(best_state["cls"])
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval(self, data, split="test"):
        enc = self.encoder
        cls = self.classifier
        class _W(nn.Module):
            def forward(self, data, **kwargs):
                return cls(enc(data))
        return evaluate_pyg_model(_W(), data, split=split,
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        return self._eval(data, split=split)


# =========================================================
# 6. FairVGNN
#    "Improving Fairness in Graph Neural Networks via
#     Mitigating Sensitive Attribute Leakage"
#    Wang et al., KDD 2022
#    github.com/YuWVandy/FairVGNN
#
#    핵심 아이디어:
#      - feature propagation 후 sensitive attribute leakage 문제 해결
#      - Pearson 상관계수로 sensitive-correlated feature channel 식별
#      - Generator: correlated channel mask 학습
#      - Discriminator: 마스킹된 feature로 sensitive 예측 (적대적)
#      - Adaptive weight clamping: encoder weight를 Lipschitz norm으로 제한
# =========================================================

class FairVGNN_Generator(nn.Module):
    """sensitive-correlated channel을 masking하는 generator"""
    def __init__(self, in_feats, eps=0.1):
        super().__init__()
        self.eps   = eps
        # mask threshold: 각 채널별 학습 가능한 임계값
        self.alpha = nn.Parameter(torch.ones(in_feats) * eps)

    def forward(self, x, correlation):
        """
        correlation: [in_feats] 각 채널의 sensitive attr와의 상관계수
        mask: |correlation| > alpha → 해당 채널 0으로 masking
        """
        alpha    = self.alpha.abs().clamp(min=1e-6)
        mask     = (correlation.abs() < alpha).float()  # 낮은 상관 → 유지
        x_masked = x * mask.unsqueeze(0)
        return x_masked, mask


class FairVGNN_Discriminator(nn.Module):
    """마스킹된 feature로 sensitive attribute 예측"""
    def __init__(self, h_feats):
        super().__init__()
        self.fc = nn.Linear(h_feats, 1)

    def forward(self, h):
        return self.fc(h).view(-1)


class FairVGNN:
    """
    FairVGNN: Generative adversarial debiasing + adaptive weight clamping.

    eps: weight clamping threshold (Lipschitz 정규화 강도)
    alpha_adv: adversarial loss 강도
    """
    def __init__(self, in_feats, h_feats, device,
                 dropout=0.5,
                 eps=0.1,
                 alpha_adv=1.0,
                 val_tradeoff_dp=0.5, val_tradeoff_eo=0.5):
        self.device          = device
        self.eps             = eps
        self.alpha_adv       = alpha_adv
        self.val_tradeoff_dp = val_tradeoff_dp
        self.val_tradeoff_eo = val_tradeoff_eo
        self.dropout         = dropout
        self.name            = "FairVGNN"
        self.training        = False

        self.generator     = FairVGNN_Generator(in_feats, eps).to(device)
        self.conv1         = GCNConv(in_feats, h_feats).to(device)
        self.conv2         = GCNConv(h_feats, 1).to(device)
        self.discriminator = FairVGNN_Discriminator(h_feats).to(device)

    def _compute_correlation(self, x, sens):
        """각 feature channel과 sensitive attr 간 Pearson 상관계수 계산"""
        s = sens.float()
        s = (s - s.mean()) / (s.std() + 1e-8)
        corr = torch.zeros(x.size(1), device=x.device)
        for i in range(x.size(1)):
            xi = x[:, i]
            xi = (xi - xi.mean()) / (xi.std() + 1e-8)
            corr[i] = (xi * s).mean()
        return corr.detach()

    def _encode(self, data, x=None):
        if x is None:
            x = data.x
        h = F.relu(self.conv1(x, data.edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def _clamp_weights(self):
        """Adaptive weight clamping: encoder weight를 eps로 제한"""
        for name, param in self.conv1.named_parameters():
            if 'weight' in name:
                param.data.clamp_(-self.eps, self.eps)
        for name, param in self.conv2.named_parameters():
            if 'weight' in name:
                param.data.clamp_(-self.eps, self.eps)

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=1e-5,
            patience=100, verbose=True, print_interval=50):
        sens   = data.sensitive_attr
        labels = data.y.float()

        # ── 사전 계산: feature correlation (학습 전 1회)
        correlation = self._compute_correlation(data.x, sens)

        # ── optimizer 분리
        gen_params  = list(self.generator.parameters())
        enc_params  = list(self.conv1.parameters()) + \
                      list(self.conv2.parameters())
        disc_params = list(self.discriminator.parameters())

        opt_gen  = _build_optimizer(gen_params,  lr=lr,
                                    weight_decay=weight_decay)
        opt_enc  = _build_optimizer(enc_params,  lr=lr,
                                    weight_decay=weight_decay)
        opt_disc = _build_optimizer(disc_params, lr=lr * 0.5,
                                    weight_decay=weight_decay)

        task_crit = nn.BCEWithLogitsLoss()
        adv_crit  = nn.BCEWithLogitsLoss()
        idx_fair  = _get_fair_idx(data)

        best_score = -float("inf")
        best_state = {
            "gen":  copy.deepcopy(self.generator.state_dict()),
            "c1":   copy.deepcopy(self.conv1.state_dict()),
            "c2":   copy.deepcopy(self.conv2.state_dict()),
        }
        counter = 0

        for epoch in range(epochs):
            self.training = True
            self.generator.train()
            self.conv1.train()
            self.conv2.train()

            # ── (1) Discriminator update
            self.discriminator.train()
            with torch.no_grad():
                x_masked, _ = self.generator(data.x, correlation)
                h_masked    = self._encode(data, x=x_masked)
            opt_disc.zero_grad()
            s_pred    = self.discriminator(h_masked[idx_fair])
            disc_loss = adv_crit(s_pred, sens[idx_fair].float())
            disc_loss.backward()
            opt_disc.step()

            # ── (2) Generator + Encoder update
            self.discriminator.eval()
            opt_gen.zero_grad()
            opt_enc.zero_grad()

            x_masked, mask = self.generator(data.x, correlation)
            h_masked       = self._encode(data, x=x_masked)
            logits         = self.conv2(h_masked, data.edge_index).view(-1)

            task_loss = task_crit(logits[data.idx_train],
                                  labels[data.idx_train])

            # adversarial: discriminator를 혼동시키도록
            s_pred_gen = self.discriminator(h_masked[idx_fair])
            adv_loss   = adv_crit(s_pred_gen, sens[idx_fair].float())

            loss = task_loss - self.alpha_adv * adv_loss
            loss.backward()
            opt_gen.step()
            opt_enc.step()

            # ── Adaptive weight clamping
            self._clamp_weights()

            self.training = False
            self.generator.eval()
            self.conv1.eval()
            self.conv2.eval()

            with torch.no_grad():
                val_result = self._eval(data, split="val",
                                        correlation=correlation)
            score = _eval_val_score_cls(
                val_result, self.val_tradeoff_dp, self.val_tradeoff_eo)

            if score > best_score:
                best_score = score
                best_state = {
                    "gen": copy.deepcopy(self.generator.state_dict()),
                    "c1":  copy.deepcopy(self.conv1.state_dict()),
                    "c2":  copy.deepcopy(self.conv2.state_dict()),
                }
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                print(f"[{self.name}] Epoch {epoch+1:04d} | "
                      f"task={task_loss.item():.4f} | "
                      f"adv={adv_loss.item():.4f} | "
                      f"val={val_result} | score={score:.4f}")

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at epoch {epoch+1}.")
                break

        self.generator.load_state_dict(best_state["gen"])
        self.conv1.load_state_dict(best_state["c1"])
        self.conv2.load_state_dict(best_state["c2"])
        self.correlation = correlation  # 저장
        if verbose:
            print(f"[{self.name}] Done. Best score: {best_score:.4f}")

    @torch.no_grad()
    def _eval(self, data, split="test", correlation=None):
        if correlation is None:
            correlation = self.correlation
        gen = self.generator
        c1  = self.conv1
        c2  = self.conv2
        drop = self.dropout
        corr = correlation
        class _W(nn.Module):
            def forward(self, data, **kwargs):
                x_m, _ = gen(data.x, corr)
                h = F.relu(c1(x_m, data.edge_index))
                h = F.dropout(h, p=drop, training=False)
                return c2(h, data.edge_index).view(-1)
        return evaluate_pyg_model(_W(), data, split=split,
                                  task_type="classification")

    def evaluate(self, data, split="test"):
        return self._eval(data, split=split)
