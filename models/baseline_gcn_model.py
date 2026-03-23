import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BaselineGCN(nn.Module):
    def __init__(
        self,
        nfeat,
        hidden_dim=64,
        dropout=0.5,
        lr=1e-3,
        weight_decay=1e-5,
        pos_weight=None,
    ):
        super().__init__()
        self.gcn1 = GCNConv(nfeat, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

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

        self.train()
        self.optimizer.zero_grad()

        y_logit, _ = self.forward(data)

        loss = self.criterion(
            y_logit[idx_train],
            labels[idx_train].unsqueeze(1)
        )

        loss.backward()
        self.optimizer.step()

        return {"total_loss": loss.item()}