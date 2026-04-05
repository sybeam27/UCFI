"""
SUMMIT Preliminary Analysis 실행 스크립트
==========================================
Pokec-z, Pokec-n, NBA, German 데이터셋에 대해
Q1~Q4 분석을 수행하고 결과를 CSV로 저장.

실행:
    python run_analysis.py
    python run_analysis.py --datasets pokec_z nba
    python run_analysis.py --device cuda:0 --seed 42
"""

import argparse
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils.loader import load_dataset_from_args, build_pyg_data_from_loader_dict
from utils.analysis import run_full_analysis


# =========================================================
# 데이터셋 설정
# =========================================================

DATASET_SETTINGS = {
    "pokec_z": {
        "root":          "dataset/pokec",
        "task_type":     "classification",
        "sens_attr":     "region",
        "predict_attr":  "I_am_working_in_field",
        "label_number":  500,
        "sens_ratio":    0.2,
        "is_pokec":      True,
    },
    "pokec_n": {
        "root":          "dataset/pokec",
        "task_type":     "classification",
        "sens_attr":     "region",
        "predict_attr":  "I_am_working_in_field",
        "label_number":  500,
        "sens_ratio":    0.2,
        "is_pokec":      True,
    },
    "nba": {
        "root":          "dataset/NBA",
        "task_type":     "classification",
        "sens_attr":     "country",
        "predict_attr":  "SALARY",
        "label_number":  100,
        "sens_ratio":    0.2,
        "is_pokec":      False,
    },
    "german": {
        "root":          "dataset/NIFTY",
        "task_type":     "classification",
        "sens_attr":     "Gender",
        "predict_attr":  "GoodCustomer",
        "label_number":  None,
        "sens_ratio":    None,
        "is_pokec":      False,
    },
}


# =========================================================
# Baseline GCN
# =========================================================

class BaselineGCN(nn.Module):
    def __init__(self, in_feats, h_feats=64, out_dim=1, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, out_dim)
        self.dropout = dropout

    def forward(self, data, edge_index=None):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index).view(-1)


def train_baseline(data, task_type, device,
                   hidden_dim=64, dropout=0.5,
                   lr=1e-3, weight_decay=1e-5,
                   epochs=1000, patience=100,
                   verbose=True):
    """
    공정성 제약 없는 baseline GCN 학습.
    분류: BCEWithLogitsLoss / 회귀: MSELoss
    """
    model     = BaselineGCN(data.x.size(1), hidden_dim, 1, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" \
                else nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    counter       = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out   = model(data)
        label = data.y.float()
        loss  = criterion(out[data.idx_train], label[data.idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(data)
            val_loss = criterion(val_out[data.idx_val],
                                 label[data.idx_val]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            counter       = 0
        else:
            counter += 1

        if verbose and (epoch == 0 or (epoch + 1) % 200 == 0):
            print(f"  Epoch {epoch+1:04d} | train_loss={loss.item():.4f} "
                  f"| val_loss={val_loss:.4f}")

        if counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f}")
    return model


# =========================================================
# 결과 저장
# =========================================================

def save_results(results, dataset_name, task_type, save_dir="results/analysis"):
    os.makedirs(save_dir, exist_ok=True)

    tables = {
        "q1_group_stats":      results["q1"]["group_stats"],
        "q1_concentration":    results["q1"]["concentration_df"],
        "q2_uncertainty":      results["q2"]["highrisk_uncertainty_df"],
        "q3_selective":        results["q3"]["selective_df"],
        "q4_boundary":         results["q4_boundary"]["intersection_df"],
        "q4_lhd":              results["q4_lhd"]["intersection_df"],
    }

    for name, df in tables.items():
        path = os.path.join(save_dir, f"{dataset_name}_{task_type}_{name}.csv")
        df.round(4).to_csv(path, index=False)
        print(f"  Saved: {path}")


# =========================================================
# 단일 데이터셋 분석 실행
# =========================================================

def run_dataset_analysis(dataset_name, seed, device, hidden_dim,
                          save_dir, verbose):
    
    cfg       = DATASET_SETTINGS[dataset_name]
    task_type = cfg["task_type"]
    sens_attr = cfg["sens_attr"]
    label_number = cfg["label_number"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Task: {task_type}")
    print(f"{'='*60}")

    # ── 데이터 로드
    # ── load_dataset_from_args는 키워드 인자로 직접 전달
    dataset_dict = load_dataset_from_args(
        dataset_name   = dataset_name,
        task_type      = task_type,
        sens_attr      = sens_attr,
        remove_leakage = True,
    )

    if task_type == "classification":
        dataset_dict = prepare_classification_dataset(
            dataset_dict,
            train_per_class=label_number,
            seed=seed,
        )

    # ── PyG Data 변환
    data = build_pyg_data_from_loader_dict(
        dataset_dict, device=device, task_type=task_type
    )

    # ── Baseline GCN 학습
    print(f"\n[{dataset_name}] Training baseline GCN ...")
    model = train_baseline(
        data       = data,
        task_type  = task_type,
        device     = device,
        hidden_dim = hidden_dim,
        verbose    = verbose,
    )

    # ── Q1~Q4 분석
    print(f"\n[{dataset_name}] Running Q1~Q4 analysis ...")
    results = run_full_analysis(
        model     = model,
        data      = data,
        task_type = task_type,
        split     = "test",
        verbose   = True,
    )

    # ── 결과 저장
    print(f"\n[{dataset_name}] Saving results ...")
    save_results(results, dataset_name, task_type, save_dir=save_dir)

    return results


# =========================================================
# Main
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_classification_dataset(dataset_dict, train_per_class=500, seed=42):
    labels = dataset_dict["labels"].clone()
    labels[labels > 1] = 1
    dataset_dict["labels"] = labels

    valid_mask  = labels >= 0
    labeled_idx = torch.where(valid_mask)[0]

    idx_0 = labeled_idx[labels[labeled_idx] == 0]
    idx_1 = labeled_idx[labels[labeled_idx] == 1]

    rng = torch.Generator()
    rng.manual_seed(seed)

    def shuffle(idx):
        return idx[torch.randperm(len(idx), generator=rng)]

    idx_0 = shuffle(idx_0)
    idx_1 = shuffle(idx_1)

    n_train_0 = min(train_per_class, int(len(idx_0) * 0.5))
    n_train_1 = min(train_per_class, int(len(idx_1) * 0.5))

    train_0, rest_0 = idx_0[:n_train_0], idx_0[n_train_0:]
    train_1, rest_1 = idx_1[:n_train_1], idx_1[n_train_1:]

    idx_train = torch.cat([train_0, train_1])

    rest = shuffle(torch.cat([rest_0, rest_1]))
    mid  = len(rest) // 2
    idx_val  = rest[:mid]
    idx_test = rest[mid:]

    dataset_dict["idx_train"]      = idx_train
    dataset_dict["idx_val"]        = idx_val
    dataset_dict["idx_test"]       = idx_test
    dataset_dict["idx_sens_train"] = idx_train

    print(f"[prepare] binarize 완료 | class 0: {len(idx_0)}개, class 1: {len(idx_1)}개")
    print(f"[prepare] train={len(idx_train)} (0:{n_train_0}, 1:{n_train_1}) | "
          f"val={len(idx_val)} | test={len(idx_test)}")

    return dataset_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pokec_z", "pokec_n", "nba", "german"],
                        choices=list(DATASET_SETTINGS.keys()))
    parser.add_argument("--device",     type=str,  default="cpu")
    parser.add_argument("--seed",       type=int,  default=27)
    parser.add_argument("--hidden_dim", type=int,  default=64)
    parser.add_argument("--save_dir",   type=str,  default="outputs/analysis")
    parser.add_argument("--verbose",    action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = run_dataset_analysis(
            dataset_name = ds,
            seed         = args.seed,
            device       = args.device,
            hidden_dim   = args.hidden_dim,
            save_dir     = args.save_dir,
            verbose      = args.verbose,
        )

    print(f"\n{'='*60}")
    print("All analyses complete.")
    print(f"Results saved to: {args.save_dir}/")
