import random
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def scipy_adj_to_edge_index(adj: sp.spmatrix):
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    row = torch.LongTensor(adj.row)
    col = torch.LongTensor(adj.col)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def binarize_sensitive_attribute(sens: torch.Tensor, positive_value=None):
    sens_np = sens.detach().cpu().numpy()
    uniq = np.unique(sens_np)

    if len(uniq) < 2:
        raise ValueError("민감속성 고유값이 2개 미만입니다.")

    if positive_value is None:
        base = np.min(uniq)
        sens_bin = (sens_np != base).astype(np.float32)
    else:
        sens_bin = (sens_np == positive_value).astype(np.float32)

    return torch.FloatTensor(sens_bin)

def prepare_pyg_data(data_dict, device="cpu", sens_binary=True):
    x = data_dict["features"].float()

    y = data_dict["labels"].clone().long()
    y[y > 1] = 1

    sensitive_attr = data_dict["sens"].clone().float()
    if sens_binary:
        sensitive_attr = binarize_sensitive_attribute(sensitive_attr)
    sensitive_attr = sensitive_attr.long()

    edge_index = scipy_adj_to_edge_index(data_dict["adj"])

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.sensitive_attr = sensitive_attr
    data.idx_train = data_dict["idx_train"].long()
    data.idx_val = data_dict["idx_val"].long()
    data.idx_test = data_dict["idx_test"].long()
    data.idx_sens_train = data_dict["idx_sens_train"].long()

    return data.to(device)