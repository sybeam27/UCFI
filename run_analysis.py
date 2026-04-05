"""
SUMMIT Preliminary Analysis 실행 스크립트
==========================================
Pokec-z, Pokec-n 데이터셋에 대해 Q1~Q4 분석 수행.
multiple runs(다중 시드)로 통계적 신뢰성 확보.
accuracy, AUC, Mann-Whitney U test 결과 포함.

실행:
    python run_analysis.py
    python run_analysis.py --datasets pokec_z --runs 10
    python run_analysis.py --device cuda:0 --runs 10
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
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp

from utils.loader import load_dataset_from_args, build_pyg_data_from_loader_dict
from utils.analysis import run_full_analysis


# =========================================================
# 데이터셋 설정
# =========================================================

DATASET_SETTINGS = {
    "pokec_z": {
        "task_type":    "classification",
        "sens_attr":    "region",
        "label_number": 500,
    },
    "pokec_n": {
        "task_type":    "classification",
        "sens_attr":    "region",
        "label_number": 500,
    },
}


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_baseline(data, device,
                   hidden_dim=64, dropout=0.5,
                   lr=1e-3, weight_decay=1e-5,
                   epochs=1000, patience=100,
                   verbose=False):
    model     = BaselineGCN(data.x.size(1), hidden_dim, 1, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

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
            val_loss = criterion(
                model(data)[data.idx_val], label[data.idx_val]
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            counter       = 0
        else:
            counter += 1

        if verbose and (epoch == 0 or (epoch + 1) % 200 == 0):
            print(f"    Epoch {epoch+1:04d} | train={loss.item():.4f} "
                  f"| val={val_loss:.4f}")

        if counter >= patience:
            break

    model.load_state_dict(best_state)
    return model


# =========================================================
# Pokec 분류 전처리
# =========================================================

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

    idx_0, idx_1 = shuffle(idx_0), shuffle(idx_1)
    n0 = min(train_per_class, int(len(idx_0) * 0.5))
    n1 = min(train_per_class, int(len(idx_1) * 0.5))

    idx_train = torch.cat([idx_0[:n0], idx_1[:n1]])
    rest      = shuffle(torch.cat([idx_0[n0:], idx_1[n1:]]))
    mid       = len(rest) // 2

    dataset_dict["idx_train"]      = idx_train
    dataset_dict["idx_val"]        = rest[:mid]
    dataset_dict["idx_test"]       = rest[mid:]
    dataset_dict["idx_sens_train"] = idx_train
    return dataset_dict


# =========================================================
# 단일 run 실행
# =========================================================

def run_single(dataset_name, seed, device, hidden_dim, verbose=False):
    cfg       = DATASET_SETTINGS[dataset_name]
    task_type = cfg["task_type"]
    set_seed(seed)

    dataset_dict = load_dataset_from_args(
        dataset_name   = dataset_name,
        task_type      = task_type,
        sens_attr      = cfg["sens_attr"],
        remove_leakage = True,
    )
    dataset_dict = prepare_classification_dataset(
        dataset_dict, train_per_class=cfg["label_number"], seed=seed
    )
    data = build_pyg_data_from_loader_dict(
        dataset_dict, device=device, task_type=task_type
    )
    model = train_baseline(data, device=device, hidden_dim=hidden_dim,
                           verbose=verbose)
    results = run_full_analysis(
        model=model, data=data, task_type=task_type,
        split="test", verbose=False,
    )
    return results


# =========================================================
# 수치 컬럼 추출
# =========================================================

def extract_scalars(results: dict) -> dict:
    flat = {}

    # ── accuracy, AUC
    flat["accuracy"] = results.get("accuracy", np.nan)
    flat["auc"]      = results.get("auc",      np.nan)

    # ── Q1: overall s1 ratio (t-test 기준값)
    flat["q1_overall_s1_ratio"] = results["q1"]["overall_s1_ratio"]

    # ── Q1: overall fairness
    for k, v in results["q1"]["overall_fairness"].items():
        flat[f"q1_overall_{k}"] = v

    # ── Q1: concentration
    for _, row in results["q1"]["concentration_df"].iterrows():
        m = row["metric"]
        flat[f"q1_{m}_enrichment"]   = row.get("enrichment",    np.nan)
        flat[f"q1_{m}_top_s1_ratio"] = row.get("top_s1_ratio",  np.nan)
        flat[f"q1_{m}_parity_gap"]   = row.get("parity_gap",    np.nan)
        flat[f"q1_{m}_equality_gap"] = row.get("equality_gap",  np.nan)
        flat[f"q1_{m}_mw_p"]         = row.get("mannwhitney_p", np.nan)

    # ── Q2: uncertainty diff
    for _, row in results["q2"]["highrisk_uncertainty_df"].iterrows():
        m = row["metric"]
        flat[f"q2_{m}_high_unc"] = row.get("high_unc_mean",      np.nan)
        flat[f"q2_{m}_low_unc"]  = row.get("remaining_unc_mean", np.nan)
        flat[f"q2_{m}_diff"]     = row.get("diff",               np.nan)
        flat[f"q2_{m}_mw_p"]     = row.get("mannwhitney_p",      np.nan)

    # ── Q3: selective
    for _, row in results["q3"]["selective_df"].iterrows():
        s = (row["score"]
             .replace(" ", "_")
             .replace("×", "x")
             .replace("+", "p"))
        flat[f"q3_{s}_error_rate"]   = row.get("error_rate",   np.nan)
        flat[f"q3_{s}_parity_gap"]   = row.get("parity_gap",   np.nan)
        flat[f"q3_{s}_equality_gap"] = row.get("equality_gap", np.nan)

    # ── Q4 boundary
    for _, row in results["q4_boundary"]["intersection_df"].iterrows():
        g = (row["group"]
             .replace(" ", "_")
             .replace("+", "p"))
        flat[f"q4b_{g}_parity_gap"]   = row.get("parity_gap",   np.nan)
        flat[f"q4b_{g}_equality_gap"] = row.get("equality_gap", np.nan)
        flat[f"q4b_{g}_s1_ratio"]     = row.get("s1_ratio",     np.nan)
        flat[f"q4b_{g}_error_rate"]   = row.get("error_rate",   np.nan)

    # ── Q4 lhd
    for _, row in results["q4_lhd"]["intersection_df"].iterrows():
        g = (row["group"]
             .replace(" ", "_")
             .replace("+", "p"))
        flat[f"q4l_{g}_parity_gap"]   = row.get("parity_gap",   np.nan)
        flat[f"q4l_{g}_equality_gap"] = row.get("equality_gap", np.nan)
        flat[f"q4l_{g}_s1_ratio"]     = row.get("s1_ratio",     np.nan)
        flat[f"q4l_{g}_error_rate"]   = row.get("error_rate",   np.nan)

    return flat


# =========================================================
# 통계 집계
# =========================================================

def aggregate_runs(scalar_list: list) -> pd.DataFrame:
    df   = pd.DataFrame(scalar_list)
    mean = df.mean()
    std  = df.std()
    n    = df.notna().sum()
    ci95 = 1.96 * std / np.sqrt(n)
    return pd.DataFrame({
        "mean": mean.round(4),
        "std":  std.round(4),
        "ci95": ci95.round(4),
        "n":    n,
    })


# =========================================================
# 통계 검정
# =========================================================

def cross_run_tests(scalar_list: list) -> pd.DataFrame:
    df   = pd.DataFrame(scalar_list)
    rows = []

    def _wilcoxon(col_a, col_b, label, alternative="less"):
        a = df[col_a].dropna().values if col_a in df else np.array([])
        b = df[col_b].dropna().values if col_b in df else np.array([])
        n = min(len(a), len(b))
        if n < 3:
            return {"test": label, "statistic": np.nan, "p_value": np.nan,
                    "interpretation": "insufficient data (need >= 3 runs)"}
        d = a[:n] - b[:n]
        if np.all(d == 0):
            return {"test": label, "statistic": np.nan, "p_value": np.nan,
                    "interpretation": "all differences are zero"}
        try:
            stat, p = wilcoxon(d, alternative=alternative)
            interp = "significant (p<0.05)" if p < 0.05 else "not significant"
        except Exception as e:
            stat, p, interp = np.nan, np.nan, str(e)
        return {"test": label,
                "statistic": round(float(stat), 4) if not np.isnan(stat) else np.nan,
                "p_value":   round(float(p), 4)    if not np.isnan(p)    else np.nan,
                "interpretation": interp}

    # ── F1: top_s1_ratio > overall_s1_ratio (one-sample t-test)
    for m in ["w_degree", "w_boundary", "w_lhd"]:
        col_top = f"q1_{m}_top_s1_ratio"
        col_ovr = "q1_overall_s1_ratio"
        if col_top not in df.columns or col_ovr not in df.columns:
            continue
        top_vals = df[col_top].dropna().values
        ovr_mean = df[col_ovr].dropna().mean()
        if len(top_vals) < 3:
            rows.append({"finding": "F1",
                         "test": f"top_s1 > overall_s1 ({m})",
                         "statistic": np.nan, "p_value": np.nan,
                         "interpretation": "insufficient data"})
            continue
        stat, p = ttest_1samp(top_vals, popmean=ovr_mean, alternative="greater")
        interp  = "significant (p<0.05)" if p < 0.05 else "not significant"
        rows.append({"finding": "F1",
                     "test": f"top_s1 > overall_s1 ({m})",
                     "statistic": round(float(stat), 4),
                     "p_value":   round(float(p), 4),
                     "interpretation": interp})

    # ── F3: unc_only_parity < overall_parity
    rows.append({"finding": "F3",
                 **_wilcoxon("q3_Uncertainty_only_parity_gap",
                             "q3_Overall_parity_gap",
                             "unc_only_parity < overall_parity",
                             alternative="less")})

    # ── F3: unc_only_error > overall_error
    rows.append({"finding": "F3",
                 **_wilcoxon("q3_Uncertainty_only_error_rate",
                             "q3_Overall_error_rate",
                             "unc_only_error > overall_error",
                             alternative="greater")})

    # ── F4: high_both_equality > high_struct_only_equality
    rows.append({"finding": "F4",
                 **_wilcoxon(
                     "q4b_High_w_boundary_p_High_Uncertainty_equality_gap",
                     "q4b_High_w_boundary_only_equality_gap",
                     "high_both_equality > high_struct_only_equality",
                     alternative="greater")})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["finding", "test", "statistic", "p_value", "interpretation"])


# =========================================================
# 결과 저장
# =========================================================

def save_all(results_list, scalar_list, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    last   = results_list[-1]
    tables = {
        "q1_group_stats":   last["q1"]["group_stats"],
        "q1_concentration": last["q1"]["concentration_df"],
        "q2_uncertainty":   last["q2"]["highrisk_uncertainty_df"],
        "q3_selective":     last["q3"]["selective_df"],
        "q4_boundary":      last["q4_boundary"]["intersection_df"],
        "q4_lhd":           last["q4_lhd"]["intersection_df"],
    }
    for name, df in tables.items():
        path = os.path.join(save_dir, f"{dataset_name}_{name}.csv")
        df.round(4).to_csv(path, index=False)
        print(f"  [table]  {path}")

    raw_path = os.path.join(save_dir, f"{dataset_name}_runs_raw.csv")
    pd.DataFrame(scalar_list).round(4).to_csv(raw_path, index=False)
    print(f"  [raw]    {raw_path}")

    agg      = aggregate_runs(scalar_list)
    agg_path = os.path.join(save_dir, f"{dataset_name}_runs_summary.csv")
    agg.to_csv(agg_path)
    print(f"  [agg]    {agg_path}")

    test_df   = cross_run_tests(scalar_list)
    test_path = os.path.join(save_dir, f"{dataset_name}_stat_tests.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  [tests]  {test_path}")

    return agg, test_df


# =========================================================
# 핵심 Finding 출력
# =========================================================

def print_key_findings(agg: pd.DataFrame, test_df: pd.DataFrame,
                       dataset_name: str):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Key Findings: {dataset_name.upper()}  (mean ± std)")
    print(sep)

    def _get(key):
        if key in agg.index:
            r = agg.loc[key]
            return f"{r['mean']:.4f} ± {r['std']:.4f}"
        return "N/A"

    def _p(label_substr):
        if test_df is None or test_df.empty:
            return ""
        rows = test_df[test_df["test"].str.contains(label_substr,
                                                     regex=False, na=False)]
        if rows.empty:
            return ""
        p   = rows.iloc[0]["p_value"]
        sig = "*" if (not np.isnan(p) and p < 0.05) else ""
        return f"  p={p:.4f}{sig}"

    print(f"\n[Baseline]")
    print(f"  Accuracy = {_get('accuracy')}  |  AUC = {_get('auc')}")

    print(f"\n[Finding 1] 구조적 위험의 집중성")
    for m in ["w_degree", "w_boundary", "w_lhd"]:
        print(f"  {m:12s} | enrichment={_get(f'q1_{m}_enrichment')} | "
              f"equality={_get(f'q1_{m}_equality_gap')} | "
              f"MW_p={_get(f'q1_{m}_mw_p')}{_p(f'overall_s1 ({m})')}")

    print(f"\n[Finding 2] 구조 위험 vs 불확실성")
    for m in ["w_degree", "w_boundary", "w_lhd"]:
        print(f"  {m:12s} | high={_get(f'q2_{m}_high_unc')} | "
              f"low={_get(f'q2_{m}_low_unc')} | "
              f"diff={_get(f'q2_{m}_diff')} | MW_p={_get(f'q2_{m}_mw_p')}")

    print(f"\n[Finding 3] 불확실성 단독 선택의 한계")
    print(f"  Uncertainty only : error={_get('q3_Uncertainty_only_error_rate')} | "
          f"parity={_get('q3_Uncertainty_only_parity_gap')}"
          f"{_p('unc_only_parity')}")
    print(f"  Overall          : error={_get('q3_Overall_error_rate')} | "
          f"parity={_get('q3_Overall_parity_gap')}")
    print(f"  [검정] error rate 차이: {_p('unc_only_error')}")

    print(f"\n[Finding 4] 구조 × 불확실성 교차 (w_boundary 기준)")
    groups = [
        ("High_w_boundary_p_High_Uncertainty", "High+High       "),
        ("High_w_boundary_only",               "High struct only"),
        ("High_Uncertainty_only",              "High unc only   "),
        ("Other",                              "Other           "),
    ]
    for key_suffix, label in groups:
        p  = _get(f"q4b_{key_suffix}_parity_gap")
        e  = _get(f"q4b_{key_suffix}_equality_gap")
        s1 = _get(f"q4b_{key_suffix}_s1_ratio")
        print(f"  {label} | parity={p} | equality={e} | s1={s1}")
    print(f"  [검정] High+High > High struct only (equality): "
          f"{_p('high_both_equality')}")

    if test_df is not None and not test_df.empty:
        print(f"\n[통계 검정 요약]")
        print(test_df.to_string(index=False))


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pokec_z", "pokec_n"],
                        choices=list(DATASET_SETTINGS.keys()))
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--seed",       type=int, default=27,
                        help="Base seed. run i uses seed+i")
    parser.add_argument("--runs",       type=int, default=10,
                        help="Number of runs (>= 10 recommended)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--save_dir",   type=str, default="outputs/analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.runs < 3:
        print("[WARNING] runs < 3: statistical tests require >= 3 runs.")

    for dataset_name in args.datasets:
        print(f"\n{'='*65}")
        print(f"Dataset: {dataset_name.upper()}  |  Runs: {args.runs}  "
              f"|  Base seed: {args.seed}")
        print(f"{'='*65}")

        results_list = []
        scalar_list  = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            print(f"\n  [Run {run_idx+1}/{args.runs}] seed={seed} ...",
                  end="", flush=True)

            results = run_single(
                dataset_name = dataset_name,
                seed         = seed,
                device       = args.device,
                hidden_dim   = args.hidden_dim,
            )

            scalars = extract_scalars(results)
            results_list.append(results)
            scalar_list.append(scalars)

            q1 = results["q1"]["overall_fairness"]
            q4 = results["q4_boundary"]["intersection_df"]
            mask = (q4["group"].str.contains("High Uncertainty") &
                    q4["group"].str.contains("High w_boundary"))
            pb = q4.loc[mask, "parity_gap"].values
            pb = pb[0] if len(pb) > 0 else float("nan")

            print(f" acc={results['accuracy']:.4f} | "
                  f"auc={results['auc']:.4f} | "
                  f"parity={q1.get('parity_gap', float('nan')):.3f} | "
                  f"equality={q1.get('equality_gap', float('nan')):.3f} | "
                  f"q4_parity={pb:.3f}")

        print(f"\n[{dataset_name}] Saving results ({args.runs} runs) ...")
        agg, test_df = save_all(
            results_list, scalar_list, dataset_name, args.save_dir)

        print_key_findings(agg, test_df, dataset_name)

    print(f"\n{'='*65}")
    print(f"All analyses complete. Results saved to: {args.save_dir}/")
