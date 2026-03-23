import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GroupWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_prob, sensitive_attr):
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


class GCNGroupNorm(nn.Module):
    def __init__(
        self,
        nfeat,
        hidden_dim=64,
        dropout=0.5,
        lambda_dist=0.05,
        lr=1e-3,
        weight_decay=1e-5,
        pos_weight=None,
    ):
        super().__init__()
        self.gcn1 = GCNConv(nfeat, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.lambda_dist = lambda_dist
        self.group_norm = GroupWiseNorm()

        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight), dtype=torch.float32)

        self.pos_weight = pos_weight
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        return y_logit, h

    def optimize(self, data):
        labels = data.y.float()
        idx_train = data.idx_train
        idx_fair = data.idx_sens_train
        sensitive_attr = data.sensitive_attr

        self.train()
        self.optimizer.zero_grad()

        y_logit, _ = self.forward(data)

        task_loss = self.criterion(
            y_logit[idx_train],
            labels[idx_train].unsqueeze(1)
        )

        if idx_fair.numel() > 0:
            y_prob_fair = torch.sigmoid(y_logit[idx_fair])
            s_fair = sensitive_attr[idx_fair]
            dist_loss = self.group_norm(y_prob_fair, s_fair)
        else:
            dist_loss = torch.tensor(0.0, device=y_logit.device)

        total_loss = task_loss + self.lambda_dist * dist_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "dist_loss": dist_loss.item(),
        }