from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


# =========================================================
# Dataset configs
# =========================================================
CLASSIFICATION_DATASET_CONFIGS = {
    "pokec_z": {
        "csv_file": "region_job.csv",
        "edge_file": "region_job_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "I_am_working_in_field",
        "sens_attrs": ["region", "gender"],
        "dataset_type": "pokec_style",
        "test_idx": False,
        "label_number": 500,
        "sens_number": 200,
        "task_type": "classification",
        "label_binarize": None,
        "dn": "Pokec_z_Classification",
    },

    "pokec_n": {
        "csv_file": "region_job_2.csv",
        "edge_file": "region_job_2_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "I_am_working_in_field",
        "sens_attrs": ["region", "gender"],
        "dataset_type": "pokec_style",
        "test_idx": False,
        "label_number": 500,
        "sens_number": 200,
        "task_type": "classification",
        "label_binarize": None,
        "dn": "Pokec_n_Classification",
    },

    "german": {
        "csv_file": "german.csv",
        "edge_file": "german_edges.txt",
        "id_col": None,
        "predict_attr": "GoodCustomer",
        "sens_attrs": ["Gender"],
        "dataset_type": "german_style",
        "test_idx": None,
        "label_number": 100,
        "sens_number": 100,
        "task_type": "classification",
        "label_binarize": None,
        "dn": "German_Classification",
    },

    "nba": {
        "csv_file": "nba.csv",
        "edge_file": "nba_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "SALARY",
        "sens_attrs": ["country"],
        "dataset_type": "pokec_style",
        "test_idx": True,
        "label_number": 100,
        "sens_number": 50,
        "task_type": "classification",
        "label_binarize": "greater_than_zero",   # SALARY > 0 -> 1 else 0
        "dn": "NBA_SalaryBinary_Classification",
    },
}


REGRESSION_DATASET_CONFIGS = {
    "pokec_z": {
        "csv_file": "region_job.csv",
        "edge_file": "region_job_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "completion_percentage",
        "sens_attrs": ["region", "gender"],
        "dataset_type": "pokec_style",
        "test_idx": False,
        "label_number": 500,
        "sens_number": 200,
        "task_type": "regression",
        "description": "Pokec_z profile completion regression",
    },

    "pokec_n": {
        "csv_file": "region_job_2.csv",
        "edge_file": "region_job_2_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "completion_percentage",
        "sens_attrs": ["region", "gender"],
        "dataset_type": "pokec_style",
        "test_idx": False,
        "label_number": 500,
        "sens_number": 200,
        "task_type": "regression",
        "description": "Pokec_n profile completion regression",
    },

    "nba": {
        "csv_file": "nba.csv",
        "edge_file": "nba_relationship.txt",
        "id_col": "user_id",
        "predict_attr": "PIE",   # 필요하면 호출 시 predict_attr="MPG"
        "sens_attrs": ["country"],
        "dataset_type": "pokec_style",
        "test_idx": True,
        "label_number": 100,
        "sens_number": 50,
        "task_type": "regression",
        "description": "NBA player performance regression",
    },
    
    "german": {
        "csv_file": "german.csv",
        "edge_file": "german_edges.txt",
        "id_col": None,
        "predict_attr": "LoanAmount",
        "sens_attrs": [
            "Gender",
            "ForeignWorker",
            "Single",
            "HasTelephone",
            "OwnsHouse",
            "Unemployed",
        ],
        "dataset_type": "german_style",
        "test_idx": None,
        "label_number": None,
        "sens_number": None,
        "task_type": "regression",
        "description": "German loan amount regression",
    },
}


# =========================================================
# Config helpers
# =========================================================
def _get_dataset_config(dataset_name: str, task_type: str) -> Dict[str, Any]:
    if task_type == "classification":
        configs = CLASSIFICATION_DATASET_CONFIGS
    elif task_type == "regression":
        configs = REGRESSION_DATASET_CONFIGS
    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type}")

    if dataset_name not in configs:
        raise ValueError(f"Unsupported dataset_name '{dataset_name}' for task_type='{task_type}'")

    return configs[dataset_name]


# =========================================================
# Graph utils
# =========================================================
def _build_symmetric_adj(num_nodes: int, edges: np.ndarray) -> sp.coo_matrix:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges shape must be (E, 2), got {edges.shape}")

    valid_mask = (
        (edges[:, 0] >= 0) &
        (edges[:, 1] >= 0) &
        (edges[:, 0] < num_nodes) &
        (edges[:, 1] < num_nodes)
    )
    edges = edges[valid_mask]

    if edges.shape[0] == 0:
        return sp.eye(num_nodes, dtype=np.float32).tocoo()

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    return adj.tocoo()

def _load_edges(
    edge_path: Path,
    node_ids: np.ndarray,
    id_col: Optional[str],
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    edges_unordered = np.genfromtxt(edge_path, dtype=float)

    if edges_unordered.size == 0:
        raise ValueError(f"{edge_path} is empty.")

    if edges_unordered.ndim == 1:
        if edges_unordered.shape[0] != 2:
            raise ValueError(f"{edge_path} has invalid 1D shape: {edges_unordered.shape}")
        edges_unordered = edges_unordered.reshape(1, 2)

    if edges_unordered.shape[1] != 2:
        raise ValueError(f"{edge_path} edge file shape is invalid: {edges_unordered.shape}")

    finite_mask = np.isfinite(edges_unordered).all(axis=1)
    edges_unordered = edges_unordered[finite_mask]

    if edges_unordered.shape[0] == 0:
        raise ValueError(f"{edge_path} has no finite edge rows.")

    edges_unordered = edges_unordered.astype(np.int64)

    if id_col is None:
        edges = edges_unordered
    else:
        idx_map = {node_id: i for i, node_id in enumerate(node_ids)}
        mapped = [idx_map.get(v, -1) for v in edges_unordered.flatten()]
        edges = np.array(mapped, dtype=np.int64).reshape(edges_unordered.shape)

    valid_mask = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    if num_nodes is not None:
        valid_mask &= (edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)

    edges = edges[valid_mask]

    if edges.shape[0] == 0:
        raise ValueError(f"{edge_path}에서 유효한 edge를 하나도 찾지 못했습니다.")

    return edges


# =========================================================
# Leakage rules
# =========================================================
def _get_pokec_completion_leakage_cols(df_columns: List[str]) -> Set[str]:
    """
    completion_percentage를 target으로 쓸 때,
    프로필 완성도를 직접 구성하거나 거의 그대로 반영하는 컬럼들을 제거.
    너무 공격적으로 제거하는 편이 안전하다.
    """
    cols = set(df_columns)

    exact = {
        "public",
        "completion_percentage",
        "spoken_languages_indicator",
        "hobbies_indicator",
        "I_most_enjoy_good_food_indicator",
        "pets_indicator",
        "body_type_indicator",
        "eye_color_indicator",
        "hair_color_indicator",
        "hair_type_indicator",
        "completed_level_of_education_indicator",
        "favourite_color_indicator",
        "relation_to_smoking_indicator",
        "relation_to_alcohol_indicator",
        "on_pokec_i_am_looking_for_indicator",
        "love_is_for_me_indicator",
        "relation_to_casual_sex_indicator",
        "my_partner_should_be_indicator",
        "marital_status_indicator",
        "relation_to_children_indicator",
        "I_like_movies_indicator",
        "I_like_watching_movie_indicator",
        "I_like_music_indicator",
        "I_mostly_like_listening_to_music_indicator",
        "the_idea_of_good_evening_indicator",
        "I_like_specialties_from_kitchen_indicator",
        "I_am_going_to_concerts_indicator",
        "my_active_sports_indicator",
        "my_passive_sports_indicator",
        "I_like_books_indicator",
    }

    # profile field 그 자체인 세부 컬럼들. completion을 사실상 직접 반영할 가능성이 큼
    prefix_or_keyword_based = {
        "anglicky", "nemecky", "rusky", "francuzsky", "spanielsky", "taliansky",
        "slovensky", "japonsky",
    }

    drop = set()
    drop |= (exact & cols)
    drop |= (prefix_or_keyword_based & cols)

    # 안전하게 indicator 류는 전부 제거
    for c in cols:
        if c.endswith("_indicator"):
            drop.add(c)

    return drop

def _get_dataset_leakage_drop_cols(
    dataset_name: str,
    task_type: str,
    predict_attr: str,
    df_columns: List[str],
) -> List[str]:
    """
    데이터셋/태스크/타겟에 따라 자동 제거할 leakage feature 정의.
    너무 보수적으로 가는 편이 낫다.
    """
    cols = set(df_columns)
    drop: Set[str] = set()

    # -------------------------
    # POKEC
    # -------------------------
    if dataset_name in {"pokec_z", "pokec_n"}:
        if task_type == "regression" and predict_attr == "completion_percentage":
            drop |= _get_pokec_completion_leakage_cols(df_columns)

    # -------------------------
    # NBA
    # -------------------------
    if dataset_name == "nba":
        if task_type == "classification" and predict_attr == "SALARY":
            # salary 분류에서 salary와 매우 직접적인 보상/성과 proxy 제거
            # 데이터셋 컬럼에 없으면 자동 무시됨
            nba_salary_leakage = {
                "SALARY",
                "salary",
                "salary_rank",
                "salary_bucket",
                "contract_value",
                "contract",
                "pay",
                "PIE",
                "MPG",
            }
            drop |= (nba_salary_leakage & cols)

        if task_type == "regression" and predict_attr == "PIE":
            # target 자신 및 극단적인 동일/파생 표현 방지
            nba_pie_leakage = {
                "PIE",
                "pie",
                "PIE_rank",
                "PIE_bucket",
            }
            drop |= (nba_pie_leakage & cols)

        if task_type == "regression" and predict_attr == "MPG":
            nba_mpg_leakage = {
                "MPG",
                "mpg",
                "MIN",
                "Minutes",
                "minutes",
                "minutes_per_game",
            }
            drop |= (nba_mpg_leakage & cols)

    # -------------------------
    # German
    # -------------------------
    if dataset_name == "german":
        if task_type == "classification" and predict_attr == "GoodCustomer":
            german_cls_leakage = {
                "GoodCustomer",
                "Risk",
                "RiskPerformance",
                "Label",
                "Target",
                "Default",
                "BadCustomer",
            }
            drop |= (german_cls_leakage & cols)

        if task_type == "regression" and predict_attr == "LoanAmount":
            german_reg_leakage = {
                "LoanAmount",
                "CreditAmount",
                "Credit",
                "GoodCustomer",
                "Risk",
                "RiskPerformance",
                "LoanRateAsPercentOfIncome",
            }
            drop |= (german_reg_leakage & cols)

    return sorted(drop)


# =========================================================
# Feature utils
# =========================================================
def _onehot_features(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in set(drop_cols)]
    feature_df = df[feature_cols].copy()
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    return feature_df

def _encode_numeric_or_categorical(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()
    return pd.Categorical(series).codes.astype(np.int64)

def _encode_sensitive_attribute(series: pd.Series, dataset_type: str) -> np.ndarray:
    if dataset_type == "german_style":
        s = series.copy()
        if s.dtype == object:
            s = s.replace({"Female": 1, "Male": 0})
        if pd.api.types.is_numeric_dtype(s):
            s = s.to_numpy()
        else:
            s = pd.Categorical(s).codes.astype(np.int64)
        s = np.asarray(s, dtype=np.int64)
        s[s > 0] = 1
        return s

    s = _encode_numeric_or_categorical(series).astype(np.int64)
    s[s > 0] = 1
    return s

def _encode_classification_labels(
    series: pd.Series,
    dataset_type: str,
    label_binarize: Optional[str] = None,
) -> np.ndarray:
    if dataset_type == "german_style":
        y = series.to_numpy().copy()
        y[y == -1] = 0
        return y.astype(np.int64)

    y = _encode_numeric_or_categorical(series)

    if label_binarize == "greater_than_zero":
        y = (np.asarray(y) > 0).astype(np.int64)
        return y

    y = np.asarray(y).astype(np.int64)
    y[y > 1] = 1
    return y


def _encode_regression_labels(series: pd.Series) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"Regression target '{series.name}' must be numeric, "
            f"but got dtype={series.dtype}"
        )
    return series.to_numpy(dtype=np.float32)


# =========================================================
# Split utils
# =========================================================
def _pokec_style_split(
    labels: np.ndarray,
    sens: np.ndarray,
    label_number: int,
    sens_number: int,
    seed: int,
    test_idx: bool,
):
    rng = random.Random(seed)

    label_idx = np.where(~pd.isna(labels))[0].tolist()
    rng.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]

    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens_idx = set(np.where(sens >= 0)[0].tolist())
    idx_test = np.asarray(list(sens_idx & set(idx_test)), dtype=int)

    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    rng.shuffle(idx_sens_train)
    idx_sens_train = idx_sens_train[:sens_number]

    return (
        torch.LongTensor(idx_train),
        torch.LongTensor(idx_val),
        torch.LongTensor(idx_test),
        torch.LongTensor(idx_sens_train),
    )

def _german_style_balanced_split(
    labels: np.ndarray,
    label_number: int,
    sens: np.ndarray,
    sens_number: int,
    seed: int,
):
    rng = random.Random(seed)

    label_idx_0 = np.where(labels == 0)[0].tolist()
    label_idx_1 = np.where(labels == 1)[0].tolist()
    rng.shuffle(label_idx_0)
    rng.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)],
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):],
    )

    sens_idx = set(np.where(sens >= 0)[0].tolist())
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    rng.shuffle(idx_sens_train)
    idx_sens_train = idx_sens_train[:sens_number]

    return (
        torch.LongTensor(idx_train),
        torch.LongTensor(idx_val),
        torch.LongTensor(idx_test),
        torch.LongTensor(idx_sens_train),
    )

def _regression_random_split(
    num_nodes: int,
    sens: np.ndarray,
    seed: int,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    sens_number: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    rng = np.random.RandomState(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    sens_idx = set(np.where(sens >= 0)[0].tolist())
    idx_sens_train = list(sens_idx & set(idx_train.tolist()))
    rng.shuffle(idx_sens_train)

    if sens_number is not None:
        idx_sens_train = idx_sens_train[:sens_number]

    return (
        torch.LongTensor(idx_train),
        torch.LongTensor(idx_val),
        torch.LongTensor(idx_test),
        torch.LongTensor(idx_sens_train),
    )


# =========================================================
# Main loader
# =========================================================
def load_fairness_dataset(
    dataset_name: str,
    root: str,
    task_type: str,
    sens_attr: Optional[str] = None,
    predict_attr: Optional[str] = None,
    label_number: Optional[int] = None,
    sens_number: Optional[int] = None,
    seed: int = 20,
    feature_drop_cols: Optional[List[str]] = None,
    remove_leakage: bool = True,
    fallback_to_identity_if_edge_error: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    분류/회귀 공정성 실험용 통합 데이터셋 로더.

    Args:
        dataset_name: pokec_z, pokec_n, nba, german
        root: dataset directory
        task_type: 'classification' or 'regression'
        sens_attr: 사용할 sensitive attribute
        predict_attr: 사용할 label/target column
        feature_drop_cols: 사용자가 추가로 제거할 feature 목록
        remove_leakage: 데이터셋별 leakage feature 자동 제거 여부
    """
    cfg = _get_dataset_config(dataset_name, task_type)
    root = Path(root)

    csv_path = root / cfg["csv_file"]
    edge_path = root / cfg["edge_file"]
    id_col = cfg["id_col"]
    dataset_type = cfg["dataset_type"]

    if sens_attr is None:
        sens_attr = cfg["sens_attrs"][0]
    if sens_attr not in cfg["sens_attrs"]:
        raise ValueError(
            f"sens_attr '{sens_attr}' is not allowed for {dataset_name}/{task_type}. "
            f"Available: {cfg['sens_attrs']}"
        )

    predict_attr = predict_attr or cfg["predict_attr"]
    label_number = cfg["label_number"] if label_number is None else label_number
    sens_number = cfg["sens_number"] if sens_number is None else sens_number

    df = pd.read_csv(csv_path)

    if sens_attr not in df.columns:
        raise ValueError(f"sens_attr '{sens_attr}'가 없습니다.")
    if predict_attr not in df.columns:
        raise ValueError(f"predict_attr '{predict_attr}'가 없습니다.")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}'가 없습니다.")

    # -------------------------
    # leakage-aware drop columns
    # -------------------------
    drop_cols: Set[str] = {sens_attr, predict_attr}
    if id_col is not None:
        drop_cols.add(id_col)

    auto_leakage_cols: List[str] = []
    if remove_leakage:
        auto_leakage_cols = _get_dataset_leakage_drop_cols(
            dataset_name=dataset_name,
            task_type=task_type,
            predict_attr=predict_attr,
            df_columns=list(df.columns),
        )
        drop_cols.update(auto_leakage_cols)

    if feature_drop_cols:
        drop_cols.update(feature_drop_cols)

    if verbose and auto_leakage_cols:
        print(f"[INFO] Auto-removed leakage columns for {dataset_name}/{task_type}/{predict_attr}:")
        print(sorted(auto_leakage_cols))

    feature_df = _onehot_features(df, drop_cols=sorted(drop_cols))
    features = sp.csr_matrix(feature_df.values, dtype=np.float32)

    if id_col is not None:
        node_ids = np.array(df[id_col], dtype=int)
    else:
        node_ids = np.arange(len(df), dtype=int)

    if edge_path.exists():
        try:
            edges = _load_edges(
                edge_path=edge_path,
                node_ids=node_ids,
                id_col=id_col,
                num_nodes=len(df),
            )
            adj = _build_symmetric_adj(num_nodes=len(df), edges=edges)
        except Exception as e:
            if not fallback_to_identity_if_edge_error:
                raise
            if verbose:
                print(f"[WARN] edge file load failed for {dataset_name}: {e}")
                print("[WARN] Falling back to identity adjacency.")
            adj = sp.eye(len(df), dtype=np.float32).tocoo()
    else:
        if verbose:
            print(f"[WARN] edge file not found for {dataset_name}: {edge_path}")
            print("[WARN] Falling back to identity adjacency.")
        adj = sp.eye(len(df), dtype=np.float32).tocoo()

    sens = _encode_sensitive_attribute(df[sens_attr], dataset_type=dataset_type)

    if task_type == "classification":
        labels = _encode_classification_labels(
            df[predict_attr],
            dataset_type=dataset_type,
            label_binarize=cfg.get("label_binarize"),
        )

        if dataset_type == "german_style":
            idx_train, idx_val, idx_test, idx_sens_train = _german_style_balanced_split(
                labels=labels,
                label_number=label_number,
                sens=sens,
                sens_number=sens_number,
                seed=seed,
            )
        else:
            idx_train, idx_val, idx_test, idx_sens_train = _pokec_style_split(
                labels=labels,
                sens=sens,
                label_number=label_number,
                sens_number=sens_number,
                seed=seed,
                test_idx=cfg["test_idx"],
            )

        labels_tensor = torch.LongTensor(labels)

    elif task_type == "regression":
        labels = _encode_regression_labels(df[predict_attr])

        idx_train, idx_val, idx_test, idx_sens_train = _regression_random_split(
            num_nodes=len(df),
            sens=sens,
            seed=seed,
            train_ratio=0.5,
            val_ratio=0.25,
            sens_number=sens_number,
        )

        labels_tensor = torch.FloatTensor(labels)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return {
        "adj": adj,
        "features": torch.FloatTensor(np.asarray(features.todense())),
        "labels": labels_tensor,
        "sens": torch.LongTensor(sens),
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "idx_sens_train": idx_sens_train,
        "feature_names": list(feature_df.columns),
        "df": df,
        "task_type": task_type,
        "dataset_name": dataset_name,
        "predict_attr": predict_attr,
        "sens_attr": sens_attr,
        "dropped_feature_cols": sorted(drop_cols),
        "auto_leakage_cols": auto_leakage_cols,
    }


def get_dataset_root(dataset_name: str) -> str:
    if dataset_name in ["pokec_z", "pokec_n"]:
        return "./data/pokec"
    elif dataset_name == "nba":
        return "./data/NBA"
    elif dataset_name == "german":
        return "./data/NIFTY"
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_default_sens_attrs(dataset_name: str):
    if dataset_name in ["pokec_z", "pokec_n"]:
        return ["region", "gender"]
    elif dataset_name == "nba":
        return ["country"]
    elif dataset_name == "german":
        return ["Gender"]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def load_dataset_from_args(
    dataset_name: str,
    task_type: str,
    sens_attr: str,
    remove_leakage: bool = True,
):
    """
    Returns:
        dataset_dict
    """
    root = get_dataset_root(dataset_name)

    dataset_dict = load_fairness_dataset(
        dataset_name=dataset_name,
        root=root,
        task_type=task_type,
        sens_attr=sens_attr,
        remove_leakage=remove_leakage,
    )
    return dataset_dict