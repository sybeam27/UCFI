import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class WeightedGroupWiseNorm(nn.Module):
    """
    Weighted version of GroupWiseNorm.

    For each sensitive group g in {0,1}, compute:
        weighted mean  = sum_i w_i p_i / sum_i w_i
        weighted var   = sum_i w_i (p_i - mean_g)^2 / sum_i w_i

    Final fairness distance:
        |mean_0 - mean_1| + |var_0 - var_1|
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _weighted_mean_var(self, x, w):
        w_sum = w.sum().clamp(min=self.eps)
        mean = (w * x).sum() / w_sum
        var = (w * (x - mean).pow(2)).sum() / w_sum
        return mean, var

    def forward(self, pred_prob, sensitive_attr, weights):
        """
        pred_prob: [N, 1] or [N]
        sensitive_attr: [N], binary {0,1}
        weights: [N], nonnegative
        """
        if pred_prob.dim() == 2 and pred_prob.size(1) == 1:
            pred_prob = pred_prob.squeeze(1)

        weights = weights.view(-1).clamp(min=0.0)

        mask_0 = (sensitive_attr == 0)
        mask_1 = (sensitive_attr == 1)

        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return torch.tensor(0.0, device=pred_prob.device)

        pred_0 = pred_prob[mask_0]
        pred_1 = pred_prob[mask_1]
        w_0 = weights[mask_0]
        w_1 = weights[mask_1]

        if w_0.sum() <= self.eps or w_1.sum() <= self.eps:
            return torch.tensor(0.0, device=pred_prob.device)

        mean_0, var_0 = self._weighted_mean_var(pred_0, w_0)
        mean_1, var_1 = self._weighted_mean_var(pred_1, w_1)

        mean_diff = torch.abs(mean_0 - mean_1)
        var_diff = torch.abs(var_0 - var_1)

        return mean_diff + var_diff


class StructuralUncertaintyHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return F.softplus(self.fc(h)).squeeze(-1)  # [N], positive


class GCNGroupNormSoftWeighted(nn.Module):
    """
    GCN + GroupNorm + soft-weighted fairness intervention.

    Difference from hard selective version:
    - do NOT select only top-k nodes
    - apply fairness loss on all idx_sens_train nodes
    - but assign node-wise weights using priority score

    This is usually more stable than hard masking.
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
        weight_transform="linear",          # ["linear", "power", "sigmoid", "softmax"]
        weight_power=1.0,
        weight_temperature=1.0,
        min_fair_weight=0.05,               # avoid exact zero weights
        normalize_fair_weights="mean1",     # ["none", "mean1", "sum1"]
        detach_fair_weights=True,
        lr=1e-3,
        weight_decay=1e-5,
        pos_weight=None,
    ):
        super().__init__()

        self.gcn1 = GCNConv(nfeat, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

        self.unc_head = StructuralUncertaintyHead(hidden_dim)
        self.group_norm = WeightedGroupWiseNorm()

        self.dropout = dropout
        self.lambda_dist = lambda_dist
        self.lambda_unc = lambda_unc

        self.num_perturbations = num_perturbations
        self.drop_edge_rate = drop_edge_rate
        self.risk_weights = risk_weights
        self.priority_exponents = priority_exponents

        self.weight_transform = weight_transform
        self.weight_power = weight_power
        self.weight_temperature = weight_temperature
        self.min_fair_weight = min_fair_weight
        self.normalize_fair_weights = normalize_fair_weights
        self.detach_fair_weights = detach_fair_weights

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
            "logits": y_logit,           # [N, 1]
            "h": h,                      # [N, D]
            "uncertainty": uncertainty,  # [N]
        }

    def backbone_forward(self, features, edge_index):
        h = self.gcn1(features, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.gcn2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        return {"h": h}

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

        h_orig = self.backbone_forward(features, edge_index)["h"]

        perturbed_h_list = []
        for _ in range(num_perturbations):
            pert_edge_index = self.perturb_edge_index(edge_index, drop_edge_rate)
            h_pert = self.backbone_forward(features, pert_edge_index)["h"]
            perturbed_h_list.append(h_pert)

        h_stack = torch.stack(perturbed_h_list, dim=0)  # [T, N, D]

        if use_original_center:
            center = h_orig.unsqueeze(0)
        else:
            center = h_stack.mean(dim=0, keepdim=True)

        sq_diff = (h_stack - center).pow(2).sum(dim=-1)  # [T, N]
        teacher_u = sq_diff.mean(dim=0)
        teacher_u = torch.log1p(teacher_u)

        if was_training:
            self.train()

        return teacher_u

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

    def priority_to_weights(self, priority_subset):
        """
        Convert normalized priority [0,1] into positive fairness weights.
        """
        if priority_subset.numel() == 0:
            return priority_subset

        p = priority_subset.clamp(min=0.0, max=1.0)
        eps = 1e-8

        if self.weight_transform == "linear":
            w = p

        elif self.weight_transform == "power":
            w = p.pow(self.weight_power)

        elif self.weight_transform == "sigmoid":
            # center at 0.5, temp controls steepness
            temp = max(self.weight_temperature, eps)
            w = torch.sigmoid((p - 0.5) / temp)

        elif self.weight_transform == "softmax":
            temp = max(self.weight_temperature, eps)
            w = torch.softmax(p / temp, dim=0)

        else:
            raise ValueError(
                "weight_transform must be one of ['linear', 'power', 'sigmoid', 'softmax']"
            )

        # avoid exact zeros
        w = self.min_fair_weight + (1.0 - self.min_fair_weight) * w

        if self.normalize_fair_weights == "mean1":
            w = w / (w.mean().clamp(min=eps))

        elif self.normalize_fair_weights == "sum1":
            w = w / (w.sum().clamp(min=eps))

        elif self.normalize_fair_weights == "none":
            pass

        else:
            raise ValueError(
                "normalize_fair_weights must be one of ['none', 'mean1', 'sum1']"
            )

        return w

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

        # 1) Task loss
        task_loss = self.criterion(
            y_logit[idx_train],
            labels[idx_train].unsqueeze(1)
        )

        # 2) Uncertainty supervision
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

        # 3) Static risk + priority
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
            dynamic_uncertainty=pred_unc,
            alpha=self.priority_exponents[0],
            beta=self.priority_exponents[1],
            normalize_inputs=True,
            normalize_output=True,
        )

        # 4) Soft fairness weights over ALL idx_sens_train
        sens_priority = priority_pred[idx_sens_train]

        if self.detach_fair_weights:
            sens_priority = sens_priority.detach()

        fair_weights = self.priority_to_weights(sens_priority)

        # 5) Soft-weighted fairness loss
        if idx_sens_train.numel() > 1:
            y_prob_fair = torch.sigmoid(y_logit[idx_sens_train])
            s_fair = sensitive_attr[idx_sens_train]
            dist_loss = self.group_norm(y_prob_fair, s_fair, fair_weights)
        else:
            dist_loss = torch.tensor(0.0, device=y_logit.device)

        total_loss = task_loss + self.lambda_unc * unc_loss + self.lambda_dist * dist_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()

        # logging helpers
        avg_fair_weight = float(fair_weights.mean().item()) if fair_weights.numel() > 0 else 0.0
        max_fair_weight = float(fair_weights.max().item()) if fair_weights.numel() > 0 else 0.0
        min_fair_weight = float(fair_weights.min().item()) if fair_weights.numel() > 0 else 0.0

        # Kish effective sample size: tells how concentrated the weights are
        if fair_weights.numel() > 0:
            w_sum = fair_weights.sum()
            w_sq_sum = fair_weights.pow(2).sum().clamp(min=1e-8)
            eff_n = float((w_sum.pow(2) / w_sq_sum).item())
            eff_ratio = eff_n / float(fair_weights.numel())
        else:
            eff_n = 0.0
            eff_ratio = 0.0

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "unc_loss": unc_loss.item(),
            "dist_loss": dist_loss.item(),
            "fair_weight_avg": avg_fair_weight,
            "fair_weight_min": min_fair_weight,
            "fair_weight_max": max_fair_weight,
            "fair_weight_effective_n": eff_n,
            "fair_weight_effective_ratio": eff_ratio,
            "priority_mean_all": float(priority_pred.mean().item()),
            "priority_mean_sens": float(priority_pred[idx_sens_train].mean().item()) if idx_sens_train.numel() > 0 else 0.0,
            "priority_max_sens": float(priority_pred[idx_sens_train].max().item()) if idx_sens_train.numel() > 0 else 0.0,
        }