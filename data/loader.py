from pathlib import Path
from typing import Optional, Dict, Any
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


DATASET_CONFIGS = {
    "nba": {
        "csv_file": "nba.csv",
        "edge_file": "nba_relationship.txt",
        "id_col": "user_id",
        "default_sens_attr": "country",
        "default_predict_attr": "SALARY",
    },
    "pokec_z": {
        "csv_file": "region_job.csv",
        "edge_file": "region_job_relationship.txt",
        "id_col": "user_id",
        "default_sens_attr": "region",
        "default_predict_attr": "I_am_working_in_field",
    },
    "pokec_n": {
        "csv_file": "region_job_2.csv",
        "edge_file": "region_job_2_relationship.txt",
        "id_col": "user_id",
        "default_sens_attr": "region",
        "default_predict_attr": "I_am_working_in_field",
    },
    "german": {
        "csv_file": "german.csv",
        "edge_file": "german_edges.txt",
        "id_col": None,
        "default_sens_attr": "Gender",
        "default_predict_attr": "GoodCustomer",
    },
}


def _build_symmetric_adj(num_nodes: int, edges: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    return adj.tocoo()


def _load_edges(edge_path: Path, node_ids: np.ndarray, id_col: Optional[str]) -> np.ndarray:
    edges_unordered = np.genfromtxt(edge_path, dtype=int)
    if edges_unordered.ndim == 1:
        edges_unordered = edges_unordered.reshape(-1, 2)

    if id_col is None:
        edges = edges_unordered
    else:
        idx_map = {node_id: i for i, node_id in enumerate(node_ids)}
        mapped = list(map(idx_map.get, edges_unordered.flatten()))
        if any(v is None for v in mapped):
            raise ValueError(f"edge file {edge_path} 안에 CSV의 id와 매칭되지 않는 값이 있습니다.")
        edges = np.array(mapped, dtype=int).reshape(edges_unordered.shape)

    return edges


def _encode_sensitive_column(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.values
    cat = pd.Categorical(series)
    return cat.codes.astype(np.float32)


def load_fair_graph_dataset(
    dataset_name: str,
    root: str,
    sens_attr: Optional[str] = None,
    predict_attr: Optional[str] = None,
    label_number: Optional[int] = 1000,
    sens_number: Optional[int] = 500,
    use_all_sensitive: bool = False,
    sens_ratio: Optional[float] = None,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    seed: int = 19,
    test_idx_as_val: bool = False,
    feature_drop_cols: Optional[list] = None,
) -> Dict[str, Any]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"지원하지 않는 dataset_name: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]
    root = Path(root)

    csv_path = root / cfg["csv_file"]
    edge_path = root / cfg["edge_file"] if cfg.get("edge_file") else None

    sens_attr = sens_attr or cfg["default_sens_attr"]
    predict_attr = predict_attr or cfg["default_predict_attr"]
    id_col = cfg["id_col"]

    df = pd.read_csv(csv_path)

    if sens_attr not in df.columns:
        raise ValueError(f"sens_attr '{sens_attr}'가 없습니다.")
    if predict_attr not in df.columns:
        raise ValueError(f"predict_attr '{predict_attr}'가 없습니다.")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}'가 없습니다.")

    drop_cols = [sens_attr, predict_attr]
    if id_col is not None:
        drop_cols.append(id_col)
    if feature_drop_cols:
        drop_cols.extend(feature_drop_cols)

    feature_cols = [c for c in df.columns if c not in set(drop_cols)]

    feature_df = df[feature_cols].copy()
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    features = sp.csr_matrix(feature_df.values, dtype=np.float32)

    labels = np.asarray(df[predict_attr].values)
    sens = _encode_sensitive_column(df[sens_attr])

    if id_col is not None:
        node_ids = np.array(df[id_col], dtype=int)
    else:
        node_ids = np.arange(len(df), dtype=int)

    if edge_path is not None and edge_path.exists():
        edges = _load_edges(edge_path, node_ids=node_ids, id_col=id_col)
        adj = _build_symmetric_adj(num_nodes=len(df), edges=edges)
    else:
        adj = sp.eye(len(df), dtype=np.float32).tocoo()

    features = torch.FloatTensor(np.asarray(features.todense()))
    labels = torch.LongTensor(labels)
    sens = torch.FloatTensor(sens)

    rng = random.Random(seed)
    label_idx = np.where(labels.numpy() >= 0)[0].tolist()
    rng.shuffle(label_idx)

    n_total = len(label_idx)
    n_train_full = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    n_train = n_train_full if label_number is None else min(n_train_full, label_number)

    idx_train = label_idx[:n_train]
    idx_val = label_idx[n_train_full:n_train_full + n_val]

    if test_idx_as_val:
        idx_test = label_idx[n_train:]
        idx_val = idx_test
    else:
        idx_test = label_idx[n_train_full + n_val:]

    sens_idx = set(np.where(sens.numpy() >= 0)[0].tolist())
    idx_test = np.asarray(list(sens_idx & set(idx_test)), dtype=int)

    candidate_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    rng.shuffle(candidate_sens_train)

    if use_all_sensitive:
        selected_sens_train = candidate_sens_train
    else:
        if sens_ratio is not None:
            k = int(len(candidate_sens_train) * sens_ratio)
        elif sens_number is not None:
            k = min(sens_number, len(candidate_sens_train))
        else:
            k = len(candidate_sens_train)
        selected_sens_train = candidate_sens_train[:k]

    return {
        "adj": adj,
        "features": features,
        "labels": labels,
        "sens": sens,
        "idx_train": torch.LongTensor(idx_train),
        "idx_val": torch.LongTensor(idx_val),
        "idx_test": torch.LongTensor(idx_test),
        "idx_sens_train": torch.LongTensor(selected_sens_train),
        "feature_names": list(feature_df.columns),
        "df": df,
    }