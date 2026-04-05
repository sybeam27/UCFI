"""
SUMMIT Preliminary Analysis
===========================
논문의 Q1~Q4에 대응하는 분석 코드.
분류(classification)와 회귀(regression) 모두 지원.

추가: accuracy, AUC 집계 지원
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


# =========================================================
# 0. Utilities
# =========================================================

def _minmax_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _compute_fairness_metrics(df, task_type):
    m0 = df["sens"] == 0
    m1 = df["sens"] == 1
    nan_r = {"parity_gap": np.nan, "equality_gap": np.nan,
             "pred_gap": np.nan, "bias_gap": np.nan}
    if m0.sum() == 0 or m1.sum() == 0:
        return nan_r
    if task_type == "classification":
        parity = abs(df.loc[m0, "pred"].mean() - df.loc[m1, "pred"].mean())
        m0y1 = m0 & (df["label"] == 1)
        m1y1 = m1 & (df["label"] == 1)
        equality = (
            abs(df.loc[m0y1, "pred"].mean() - df.loc[m1y1, "pred"].mean())
            if m0y1.sum() > 0 and m1y1.sum() > 0 else np.nan
        )
        return {"parity_gap": float(parity),
                "equality_gap": float(equality) if not np.isnan(equality) else np.nan,
                "pred_gap": np.nan, "bias_gap": np.nan}
    else:
        pred0 = df.loc[m0, "prob"].astype(float)
        pred1 = df.loc[m1, "prob"].astype(float)
        y0 = df.loc[m0, "label"].astype(float)
        y1 = df.loc[m1, "label"].astype(float)
        return {"parity_gap": np.nan, "equality_gap": np.nan,
                "pred_gap":  float(abs(pred0.mean() - pred1.mean())),
                "bias_gap":  float(abs((pred0 - y0).mean() - (pred1 - y1).mean()))}


def _fair_cols(task_type):
    return ["parity_gap", "equality_gap"] if task_type == "classification" \
           else ["pred_gap", "bias_gap"]


# =========================================================
# 1. 구조 지표 계산
# =========================================================

def compute_structural_signals(edge_index, sens, num_nodes):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    s   = sens.cpu().numpy().astype(int)

    degree     = np.zeros(num_nodes, dtype=np.float32)
    same_count = np.zeros(num_nodes, dtype=np.float32)
    diff_count = np.zeros(num_nodes, dtype=np.float32)

    np.add.at(degree, src, 1.0)
    for u, v in zip(src, dst):
        if s[u] == s[v]:
            same_count[u] += 1.0
        else:
            diff_count[u] += 1.0

    safe_deg       = np.clip(degree, 1.0, None)
    w_degree       = _minmax_norm(np.log1p(degree))
    boundary_ratio = diff_count / safe_deg
    w_boundary     = _minmax_norm(boundary_ratio)
    h_local        = same_count / safe_deg
    w_lhd          = _minmax_norm(np.abs(h_local - h_local.mean()))

    return pd.DataFrame({
        "node_id":        np.arange(num_nodes),
        "sens":           s,
        "degree":         degree,
        "w_degree":       w_degree,
        "boundary_ratio": boundary_ratio,
        "w_boundary":     w_boundary,
        "h_local":        h_local,
        "w_lhd":          w_lhd,
    })


# =========================================================
# 2. 분석 테이블 구성 (accuracy, AUC 포함)
# =========================================================

@torch.no_grad()
def build_analysis_table(data, model, struct_df, task_type, split="test"):
    model.eval()
    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    raw = out.view(-1).cpu().numpy()

    num_nodes = data.x.size(0)
    label     = data.y.cpu().numpy().reshape(-1)
    sens      = data.sensitive_attr.cpu().numpy().astype(int).reshape(-1)

    if task_type == "classification":
        prob  = 1.0 / (1.0 + np.exp(-raw))
        pred  = (prob >= 0.5).astype(int)
        error = (pred != label.astype(int)).astype(int)
        eps   = 1e-6
        p     = np.clip(prob, eps, 1 - eps)
        unc   = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    else:
        prob  = raw
        pred  = raw
        error = np.abs(raw - label.astype(float))
        unc   = 1.0 / (np.abs(prob) + 1e-8)

    split_arr = np.array(["other"] * num_nodes, dtype=object)
    split_arr[data.idx_train.cpu().numpy()] = "train"
    split_arr[data.idx_val.cpu().numpy()]   = "val"
    split_arr[data.idx_test.cpu().numpy()]  = "test"

    node_df = pd.DataFrame({
        "node_id":       np.arange(num_nodes),
        "split":         split_arr,
        "label":         label,
        "pred":          pred,
        "prob":          prob,
        "sens":          sens,
        "error":         error,
        "uncertainty":   unc.astype(np.float32),
        "w_uncertainty": _minmax_norm(unc),
    })

    # ── Accuracy & AUC (test split 기준)
    test_mask = (split_arr == split)
    if task_type == "classification":
        acc = 1.0 - error[test_mask].mean()
        try:
            auc = roc_auc_score(label[test_mask], prob[test_mask])
        except Exception:
            auc = np.nan
    else:
        acc = np.nan
        auc = np.nan

    node_df["accuracy"] = acc   # 스칼라 → 전체 컬럼에 동일값
    node_df["auc"]      = auc

    return node_df.merge(
        struct_df[["node_id", "w_degree", "w_boundary", "w_lhd",
                   "degree", "boundary_ratio", "h_local"]],
        on="node_id", how="left"
    )


# =========================================================
# Q1
# =========================================================

def analyze_q1(analysis_df, task_type, split="test"):

    # ── 구조 지표 group_stats: 전체 데이터셋 기준
    df_all  = analysis_df.copy()
    overall_s1 = (df_all["sens"] == 1).mean()

    group_stats = df_all.groupby("sens").agg(
        n              =("node_id",        "count"),
        degree_mean    =("degree",         "mean"),
        boundary_mean  =("boundary_ratio", "mean"),
        w_degree_mean  =("w_degree",       "mean"),
        w_boundary_mean=("w_boundary",     "mean"),
        w_lhd_mean     =("w_lhd",          "mean"),
    ).reset_index()

    # MW 검정도 전체 기준
    rows = []
    for metric in ["w_degree", "w_boundary", "w_lhd"]:
        thr    = df_all[metric].quantile(0.90)
        top    = df_all[df_all[metric] >= thr].copy()
        top_s1 = (top["sens"] == 1).mean()
        x0 = df_all.loc[df_all["sens"] == 0, metric].values
        x1 = df_all.loc[df_all["sens"] == 1, metric].values
        mw_p = mannwhitneyu(x0, x1, alternative="two-sided").pvalue \
               if len(x0) > 0 and len(x1) > 0 else np.nan

        # ── 공정성 격차는 test set 기준
        df_test = analysis_df[analysis_df["split"] == split].copy()
        top_test = df_test[df_test[metric] >= thr].copy()
        fm = _compute_fairness_metrics(top_test, task_type)

        row = {
            "metric":           metric,
            "threshold_q90":    round(thr, 4),
            "top_n":            len(top),
            "overall_s1_ratio": round(overall_s1, 3),
            "top_s1_ratio":     round(top_s1, 3),
            "enrichment":       round(top_s1 / (overall_s1 + 1e-8), 3),
            "mannwhitney_p":    round(mw_p, 4),
        }
        fcols = _fair_cols(task_type)
        for k in fcols:
            row[k] = round(fm[k], 3) if not np.isnan(fm[k]) else np.nan
        rows.append(row)

    # ── overall fairness는 test set 기준
    df_test    = analysis_df[analysis_df["split"] == split].copy()
    overall_fm = _compute_fairness_metrics(df_test, task_type)

    return {
        "group_stats":      group_stats,
        "concentration_df": pd.DataFrame(rows),
        "overall_fairness": {k: round(v, 4) for k, v in overall_fm.items()
                             if not np.isnan(v)},
        "overall_s1_ratio": round(overall_s1, 3),
    }


# =========================================================
# Q2
# =========================================================

def analyze_q2(analysis_df, split="test"):
    df = analysis_df[analysis_df["split"] == split].copy()
    rows = []
    for metric in ["w_degree", "w_boundary", "w_lhd"]:
        thr      = df[metric].quantile(0.90)
        high     = df[df[metric] >= thr]
        low      = df[df[metric] <  thr]
        high_unc = high["uncertainty"].mean()
        low_unc  = low["uncertainty"].mean()
        mw_p = mannwhitneyu(
            high["uncertainty"].values, low["uncertainty"].values,
            alternative="two-sided"
        ).pvalue if len(high) > 0 and len(low) > 0 else np.nan
        rows.append({
            "metric":             metric,
            "high_risk_n":        len(high),
            "remaining_n":        len(low),
            "high_unc_mean":      round(high_unc, 5),
            "remaining_unc_mean": round(low_unc,  5),
            "diff":               round(high_unc - low_unc, 5),
            "mannwhitney_p":      round(mw_p, 4),
        })
    return {"highrisk_uncertainty_df": pd.DataFrame(rows)}


# =========================================================
# Q3
# =========================================================

def analyze_q3(analysis_df, task_type, split="test"):
    df = analysis_df[analysis_df["split"] == split].copy()

    df["boundary_x_unc"] = df["w_boundary"] * df["w_uncertainty"]
    df["lhd_x_unc"]      = df["w_lhd"]      * df["w_uncertainty"]
    df["degree_x_unc"]   = df["w_degree"]   * df["w_uncertainty"]
    df["boundary_p_unc"] = df["w_boundary"] + df["w_uncertainty"]
    df["lhd_p_unc"]      = df["w_lhd"]      + df["w_uncertainty"]

    score_cols = {
        "Uncertainty only":         "w_uncertainty",
        "w_boundary only":          "w_boundary",
        "w_lhd only":               "w_lhd",
        "w_degree only":            "w_degree",
        "w_boundary x Uncertainty": "boundary_x_unc",
        "w_lhd x Uncertainty":      "lhd_x_unc",
        "w_degree x Uncertainty":   "degree_x_unc",
        "w_boundary + Uncertainty": "boundary_p_unc",
        "w_lhd + Uncertainty":      "lhd_p_unc",
    }

    fcols = _fair_cols(task_type)
    rows  = []

    for label, col in score_cols.items():
        thr = df[col].quantile(0.90)
        top = df[df[col] >= thr].copy()
        fm  = _compute_fairness_metrics(top, task_type)
        row = {"score": label, "n": len(top),
               "error_rate": round(top["error"].mean(), 3),
               "s1_ratio":   round((top["sens"] == 1).mean(), 3)}
        for k in fcols:
            row[k] = round(fm[k], 3) if not np.isnan(fm[k]) else np.nan
        rows.append(row)

    overall_fm = _compute_fairness_metrics(df, task_type)
    row = {"score": "Overall", "n": len(df),
           "error_rate": round(df["error"].mean(), 3),
           "s1_ratio":   round((df["sens"] == 1).mean(), 3)}
    for k in fcols:
        row[k] = round(overall_fm[k], 3) if not np.isnan(overall_fm[k]) else np.nan
    rows.append(row)

    return {"selective_df": pd.DataFrame(rows),
            "overall_fairness": {k: round(v, 4) for k, v in overall_fm.items()
                                 if not np.isnan(v)}}


# =========================================================
# Q4
# =========================================================

def analyze_q4(analysis_df, task_type, split="test",
               structural_metric="w_boundary"):
    df = analysis_df[analysis_df["split"] == split].copy()

    struct_thr = df[structural_metric].quantile(0.90)
    unc_thr    = df["w_uncertainty"].quantile(0.90)
    hs = df[structural_metric] >= struct_thr
    hu = df["w_uncertainty"]   >= unc_thr

    df["group"] = "Other"
    df.loc[hs & hu,  "group"] = f"High {structural_metric} + High Uncertainty"
    df.loc[hs & ~hu, "group"] = f"High {structural_metric} only"
    df.loc[~hs & hu, "group"] = "High Uncertainty only"

    fcols = _fair_cols(task_type)
    group_order = [
        f"High {structural_metric} + High Uncertainty",
        f"High {structural_metric} only",
        "High Uncertainty only",
        "Other",
    ]

    rows = []
    for grp in group_order:
        sub = df[df["group"] == grp].copy()
        if len(sub) == 0:
            continue
        fm = _compute_fairness_metrics(sub, task_type)
        row = {"group": grp, "n": len(sub),
               "error_rate": round(sub["error"].mean(), 3),
               "s1_ratio":   round((sub["sens"] == 1).mean(), 3)}
        for k in fcols:
            row[k] = round(fm[k], 3) if not np.isnan(fm[k]) else np.nan
        rows.append(row)

    return {"intersection_df":   pd.DataFrame(rows),
            "structural_metric": structural_metric,
            "struct_threshold":  round(struct_thr, 4),
            "unc_threshold":     round(unc_thr, 4)}


# =========================================================
# 전체 파이프라인
# =========================================================

def run_full_analysis(model, data, task_type, split="test", verbose=True):
    num_nodes   = data.x.size(0)
    struct_df   = compute_structural_signals(
        data.edge_index, data.sensitive_attr, num_nodes)
    analysis_df = build_analysis_table(
        data, model, struct_df, task_type, split=split)

    # ── accuracy, AUC 추출 (test split 기준 단일값)
    test_row    = analysis_df[analysis_df["split"] == split]
    acc_val     = test_row["accuracy"].iloc[0] if len(test_row) > 0 else np.nan
    auc_val     = test_row["auc"].iloc[0]      if len(test_row) > 0 else np.nan

    q1 = analyze_q1(analysis_df, task_type, split=split)
    q2 = analyze_q2(analysis_df, split=split)
    q3 = analyze_q3(analysis_df, task_type, split=split)
    q4_boundary = analyze_q4(analysis_df, task_type, split=split,
                              structural_metric="w_boundary")
    q4_lhd      = analyze_q4(analysis_df, task_type, split=split,
                              structural_metric="w_lhd")

    if verbose:
        _print_results(q1, q2, q3, q4_boundary, task_type)

    return {
        "struct_df":    struct_df,
        "analysis_df":  analysis_df,
        "accuracy":     float(acc_val),
        "auc":          float(auc_val),
        "q1": q1, "q2": q2, "q3": q3,
        "q4_boundary":  q4_boundary,
        "q4_lhd":       q4_lhd,
    }


def _print_results(q1, q2, q3, q4, task_type):
    sep = "=" * 70
    print(f"\n{sep}\nQ1: 구조적 편향 위험은 특정 위치에 집중되는가?\n{sep}")
    print(f"  s=1 비율: {q1['overall_s1_ratio']}  |  공정성: {q1['overall_fairness']}")
    print(q1["group_stats"].to_string(index=False))
    print(q1["concentration_df"].to_string(index=False))

    print(f"\n{sep}\nQ2: 구조적 위험 노드에서 불확실성도 높아지는가?\n{sep}")
    print(q2["highrisk_uncertainty_df"].to_string(index=False))

    print(f"\n{sep}\nQ3: 불확실성 단독으로 공정성 취약 노드를 식별할 수 있는가?\n{sep}")
    print(q3["selective_df"].to_string(index=False))

    print(f"\n{sep}\nQ4: 구조 x 불확실성 교차 분석 ({q4['structural_metric']})\n{sep}")
    print(f"  struct q90={q4['struct_threshold']}  unc q90={q4['unc_threshold']}")
    print(q4["intersection_df"].to_string(index=False))
