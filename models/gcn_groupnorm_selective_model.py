import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GroupWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_prob, sensitive_attr):
        """
        pred_prob: [N, 1] or [N]
        sensitive_attr: [N], binary {0,1}
        """
        if pred_prob.dim() == 2 and pred_prob.size(1) == 1:
            pred_prob = pred_prob.squeeze(1)

        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)

        pred_0 = pred_prob[mask_0]
        pred_1 = pred_prob[mask_1]

        zero = torch.tensor(0.0, device=pred_prob.device)

        if pred_0.numel() == 0 or pred_1.numel() == 0:
            return zero

        mean_diff = torch.abs(pred_0.mean() - pred_1.mean())
        var_diff = torch.abs(
            pred_0.var(unbiased=False) - pred_1.var(unbiased=False)
        )

        return mean_diff + var_diff


class StructuralUncertaintyHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return F.softplus(self.fc(h)).squeeze(-1)   # [N], positive


class GCNGroupNormSelective(nn.Module):
    """
    Baseline GCN + GroupWiseNorm fairness loss
    applied selectively on high-priority nodes only.

    Priority is computed from:
      - static structural risk
      - dynamic uncertainty
    """
    def __init__(
        self,
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
        pos_weight=None,
    ):
        super().__init__()

        self.gcn1 = GCNConv(nfeat, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

        self.unc_head = StructuralUncertaintyHead(hidden_dim)
        self.group_norm = GroupWiseNorm()

        self.dropout = dropout
        self.lambda_dist = lambda_dist
        self.lambda_unc = lambda_unc

        self.num_perturbations = num_perturbations
        self.drop_edge_rate = drop_edge_rate
        self.risk_weights = risk_weights
        self.priority_exponents = priority_exponents
        self.priority_mode = priority_mode
        self.priority_k_frac = priority_k_frac
        self.priority_threshold = priority_threshold

        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight), dtype=torch.float32)

        self.pos_weight = pos_weight

        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.gcn2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        y_logit = self.classifier(h)
        uncertainty = self.unc_head(h)

        return {
            "logits": y_logit,
            "h": h,
            "uncertainty": uncertainty,
        }

    @staticmethod
    def perturb_edge_index(edge_index, drop_edge_rate=0.1):
        if drop_edge_rate <= 0.0:
            return edge_index

        num_edges = edge_index.size(1)
        device = edge_index.device

        keep_mask = torch.rand(num_edges, device=device) > drop_edge_rate

        if keep_mask.sum() == 0:
            rand_idx = torch.randint(0, num_edges, (1,), device=device)
            keep_mask[rand_idx] = True

        return edge_index[:, keep_mask]

    @torch.no_grad()
    def compute_instability_teacher(
        self,
        features,
        edge_index,
        num_perturbations=4,
        drop_edge_rate=0.1,
        use_original_center=False,
    ):
        was_training = self.training
        self.eval()

        out_orig = self.forward_minibackbone(features, edge_index)
        h_orig = out_orig["h"]

        perturbed_h_list = []
        for _ in range(num_perturbations):
            pert_edge_index = self.perturb_edge_index(edge_index, drop_edge_rate=drop_edge_rate)
            out_pert = self.forward_minibackbone(features, pert_edge_index)
            perturbed_h_list.append(out_pert["h"])

        h_stack = torch.stack(perturbed_h_list, dim=0)  # [T, N, D]

        if use_original_center:
            center = h_orig.unsqueeze(0)
        else:
            center = h_stack.mean(dim=0, keepdim=True)

        sq_diff = (h_stack - center).pow(2).sum(dim=-1)  # [T, N]
        teacher_u = sq_diff.mean(dim=0)                  # [N]

        if was_training:
            self.train()

        return teacher_u

    def forward_minibackbone(self, features, edge_index):
        """
        Helper used for instability teacher computation.
        Dropout is controlled externally by train/eval mode.
        """
        h = self.gcn1(features, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.gcn2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        return {"h": h}

    @staticmethod
    def uncertainty_loss(pred_uncertainty, teacher_uncertainty, index=None):
        if index is not None:
            pred_uncertainty = pred_uncertainty[index]
            teacher_uncertainty = teacher_uncertainty[index]
        return F.mse_loss(pred_uncertainty, teacher_uncertainty)

    @staticmethod
    def compute_out_degree(edge_index, num_nodes):
        src = edge_index[0]
        deg = torch.bincount(src, minlength=num_nodes).float().to(edge_index.device)
        return deg

    def compute_normalized_boundary_score(self, edge_index, sensitive_attr, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        device = edge_index.device

        diff_neighbor = (sensitive_attr[src] != sensitive_attr[dst]).float()

        diff_count = torch.zeros(num_nodes, device=device)
        diff_count.index_add_(0, src, diff_neighbor)

        deg = self.compute_out_degree(edge_index, num_nodes).clamp(min=1.0)
        boundary_score = diff_count / deg
        return boundary_score

    def compute_self_group_exposure(self, edge_index, sensitive_attr, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        device = edge_index.device

        same_neighbor = (sensitive_attr[src] == sensitive_attr[dst]).float()

        same_count = torch.zeros(num_nodes, device=device)
        same_count.index_add_(0, src, same_neighbor)

        deg = self.compute_out_degree(edge_index, num_nodes).clamp(min=1.0)
        self_exposure = same_count / deg
        return self_exposure

    def compute_self_group_exposure_deficit(self, edge_index, sensitive_attr, num_nodes):
        self_exposure = self.compute_self_group_exposure(edge_index, sensitive_attr, num_nodes)
        return 1.0 - self_exposure

    def compute_influence_proxy(self, edge_index, sensitive_attr, num_nodes):
        deg = self.compute_out_degree(edge_index, num_nodes)
        boundary_score = self.compute_normalized_boundary_score(edge_index, sensitive_attr, num_nodes)

        deg_norm = deg / (deg.max() + 1e-8)
        influence_proxy = deg_norm * boundary_score
        return influence_proxy

    @staticmethod
    def minmax_normalize(x, eps=1e-8):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)

    def compute_static_structural_risk(
        self,
        edge_index,
        sensitive_attr,
        num_nodes,
        alpha_boundary=1.0,
        alpha_exposure=1.0,
        alpha_influence=1.0,
        normalize_each=True,
    ):
        boundary = self.compute_normalized_boundary_score(edge_index, sensitive_attr, num_nodes)
        exposure_deficit = self.compute_self_group_exposure_deficit(edge_index, sensitive_attr, num_nodes)
        influence = self.compute_influence_proxy(edge_index, sensitive_attr, num_nodes)

        if normalize_each:
            boundary_n = self.minmax_normalize(boundary)
            exposure_n = self.minmax_normalize(exposure_deficit)
            influence_n = self.minmax_normalize(influence)
        else:
            boundary_n = boundary
            exposure_n = exposure_deficit
            influence_n = influence

        risk = (
            alpha_boundary * boundary_n
            + alpha_exposure * exposure_n
            + alpha_influence * influence_n
        )
        risk = self.minmax_normalize(risk)

        details = {
            "boundary_score": boundary,
            "exposure_deficit": exposure_deficit,
            "influence_proxy": influence,
            "boundary_score_norm": boundary_n,
            "exposure_deficit_norm": exposure_n,
            "influence_proxy_norm": influence_n,
        }

        return risk, details

    def compute_priority_score(
        self,
        static_risk,
        dynamic_uncertainty,
        alpha=1.0,
        beta=1.0,
        normalize_inputs=True,
        normalize_output=True,
        eps=1e-8,
    ):
        if normalize_inputs:
            risk_n = self.minmax_normalize(static_risk)
            unc_n = self.minmax_normalize(dynamic_uncertainty)
        else:
            risk_n = static_risk
            unc_n = dynamic_uncertainty

        priority = (risk_n.clamp(min=eps) ** alpha) * (unc_n.clamp(min=eps) ** beta)

        if normalize_output:
            priority = self.minmax_normalize(priority)

        aux = {
            "risk_norm": risk_n,
            "uncertainty_norm": unc_n,
        }
        return priority, aux

    @staticmethod
    def make_priority_mask(priority, mode="topk", k_frac=0.2, threshold=None):
        num_nodes = priority.size(0)
        device = priority.device

        if mode == "topk":
            k = max(1, int(num_nodes * k_frac))
            topk_idx = torch.topk(priority, k=k, largest=True).indices
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            mask[topk_idx] = True
            return mask

        elif mode == "threshold":
            if threshold is None:
                raise ValueError("threshold must be provided when mode='threshold'")
            return priority >= threshold

        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def compute_risk_and_priority(self, data):
        num_nodes = data.x.size(0)

        was_training = self.training
        self.eval()

        out = self.forward(data)
        pred_uncertainty = out["uncertainty"]

        static_risk, risk_details = self.compute_static_structural_risk(
            edge_index=data.edge_index,
            sensitive_attr=data.sensitive_attr,
            num_nodes=num_nodes,
            alpha_boundary=self.risk_weights[0],
            alpha_exposure=self.risk_weights[1],
            alpha_influence=self.risk_weights[2],
            normalize_each=True,
        )

        teacher_uncertainty = self.compute_instability_teacher(
            features=data.x,
            edge_index=data.edge_index,
            num_perturbations=self.num_perturbations,
            drop_edge_rate=self.drop_edge_rate,
            use_original_center=False,
        )

        priority_pred, priority_aux_pred = self.compute_priority_score(
            static_risk=static_risk,
            dynamic_uncertainty=pred_uncertainty,
            alpha=self.priority_exponents[0],
            beta=self.priority_exponents[1],
            normalize_inputs=True,
            normalize_output=True,
        )

        if was_training:
            self.train()

        return {
            "static_risk": static_risk,
            "risk_details": risk_details,
            "pred_uncertainty": pred_uncertainty,
            "teacher_uncertainty": teacher_uncertainty,
            "priority_pred": priority_pred,
            "priority_aux_pred": priority_aux_pred,
        }

    def optimize(self, data):
        labels = data.y.float()
        idx_train = data.idx_train
        idx_sens_train = data.idx_sens_train
        sensitive_attr = data.sensitive_attr

        self.train()
        self.optimizer.zero_grad()

        out = self.forward(data)
        y_logit = out["logits"]
        pred_unc = out["uncertainty"]

        # 1) task classification loss
        task_loss = self.criterion(
            y_logit[idx_train],
            labels[idx_train].unsqueeze(1)
        )

        # 2) uncertainty supervision
        teacher_unc = self.compute_instability_teacher(
            features=data.x,
            edge_index=data.edge_index,
            num_perturbations=self.num_perturbations,
            drop_edge_rate=self.drop_edge_rate,
            use_original_center=False,
        )

        unc_loss = self.uncertainty_loss(
            pred_uncertainty=pred_unc,
            teacher_uncertainty=teacher_unc,
            index=idx_train
        )

        # 3) priority computation
        static_risk, _ = self.compute_static_structural_risk(
            edge_index=data.edge_index,
            sensitive_attr=data.sensitive_attr,
            num_nodes=data.x.size(0),
            alpha_boundary=self.risk_weights[0],
            alpha_exposure=self.risk_weights[1],
            alpha_influence=self.risk_weights[2],
            normalize_each=True,
        )

        priority_pred, _ = self.compute_priority_score(
            static_risk=static_risk,
            dynamic_uncertainty=pred_unc.detach(),
            alpha=self.priority_exponents[0],
            beta=self.priority_exponents[1],
            normalize_inputs=True,
            normalize_output=True,
        )

        priority_mask = self.make_priority_mask(
            priority_pred,
            mode=self.priority_mode,
            k_frac=self.priority_k_frac,
            threshold=self.priority_threshold,
        )

        # selective fairness subset:
        # sensitive-known nodes ∩ high-priority nodes
        idx_fair = idx_sens_train[priority_mask[idx_sens_train]]

        if idx_fair.numel() > 0:
            y_prob_fair = torch.sigmoid(y_logit[idx_fair])
            s_fair = sensitive_attr[idx_fair]
            dist_loss = self.group_norm(y_prob_fair, s_fair)
        else:
            dist_loss = torch.tensor(0.0, device=y_logit.device)

        total_loss = task_loss + self.lambda_unc * unc_loss + self.lambda_dist * dist_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "unc_loss": unc_loss.item(),
            "dist_loss": dist_loss.item(),
            "selected_fair_count": int(idx_fair.numel()),
            "priority_mean": float(priority_pred.mean().item()),
            "priority_max": float(priority_pred.max().item()),
        }