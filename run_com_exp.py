"""
run_com_exp.py
==============
비교 모델 실험 실행 스크립트.
FairGNN, EDITS, FMP, GMMD를 run_exp.py와 동일한 설정으로 실행.

실행 예시:
    python run_com_exp.py \
        --task_type classification \
        --dataset_name pokec_z --sens_attr region \
        --device cuda:1 --runs 10 \
        --models all

    python run_com_exp.py \
        --dataset_name pokec_n \
        --sens_attr region \
        --models fairgnn fmp
"""

import os
import argparse
import random
import copy

import numpy as np
import pandas as pd
import torch

from utils.loader import load_dataset_from_args, build_pyg_data_from_loader_dict
from utils.compare_models import FairGNN, EDITS, FMP, GMMD, NIFTY, FairVGNN


# =========================================================
# 설정
# =========================================================

ALL_MODELS = ["fairgnn", "edits", "fmp", 
            #   "gmmd", # 너무 오래 걸림..
              "nifty", "fairvgnn"]

DATASET_SETTINGS = {
    "pokec_z": {"task_type": "classification", "sens_attr": "region"},
    "pokec_n": {"task_type": "classification", "sens_attr": "region"},
    "nba":     {"task_type": "classification", "sens_attr": "country"},
    "german":  {"task_type": "classification", "sens_attr": "Gender"},
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
# 데이터 전처리 (run_exp.py와 동일)
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

    print(f"[prepare] train={len(idx_train)} | "
          f"val={len(rest[:mid])} | test={len(rest[mid:])}")
    return dataset_dict


# =========================================================
# 모델 생성
# =========================================================

def build_model(model_name, in_feats, args):
    h = args.hidden_dim
    d = args.device

    if model_name == "fairgnn":
        return FairGNN(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            alpha=args.fairgnn_alpha,
            beta=args.fairgnn_beta,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    elif model_name == "edits":
        return EDITS(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            lambda_debias=args.edits_lambda,
            debias_epochs=args.edits_debias_epochs,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    elif model_name == "fmp":
        return FMP(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    elif model_name == "gmmd":
        return GMMD(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            lambda_f=args.gmmd_lambda_f,
            lambda_s=args.gmmd_lambda_s,
            gamma=args.gmmd_gamma,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    elif model_name == "nifty":
        return NIFTY(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            sim_coeff=args.nifty_sim_coeff,
            drop_edge_rate=args.nifty_drop_edge_rate,
            drop_feature_rate=args.nifty_drop_feature_rate,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    elif model_name == "fairvgnn":
        return FairVGNN(
            in_feats=in_feats, h_feats=h, device=d,
            dropout=args.dropout,
            eps=args.fairvgnn_eps,
            alpha_adv=args.fairvgnn_alpha_adv,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# 결과 저장 (run_exp.py와 동일 형식)
# =========================================================

def save_summary(summary: pd.DataFrame, args, save_dir: str = "outputs\compare"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{args.task_type}_{args.dataset_name}.csv")

    summary["dataset"]   = args.dataset_name
    summary["sens_attr"] = args.sens_attr
    summary["runs"]      = args.runs
    summary["epochs"]    = args.epochs
    summary["lr"]        = args.lr
    summary["hidden_dim"] = args.hidden_dim

    key_cols = ["dataset", "sens_attr", "task", "model", "runs",
                "epochs", "lr", "hidden_dim"]

    if os.path.exists(save_path):
        existing   = pd.read_csv(save_path)
        merge_keys = [c for c in key_cols
                      if c in existing.columns and c in summary.columns]
        merged     = existing.merge(
            summary[merge_keys], on=merge_keys, how="left", indicator=True)
        existing   = existing[
            merged["_merge"] == "left_only"
        ].drop(columns=["_merge"], errors="ignore")
        final = pd.concat([existing, summary], ignore_index=True)
    else:
        final = summary

    numeric_cols = final.select_dtypes(include="number").columns
    final[numeric_cols] = final[numeric_cols].round(4)
    final.to_csv(save_path, index=False)
    print(f"[Save] {save_path} ({len(final)} rows)")


# =========================================================
# 단일 모델 × 단일 run 실험
# =========================================================

def run_one(model_name, dataset_dict, args):
    data = build_pyg_data_from_loader_dict(
        dataset_dict, device=args.device, task_type=args.task_type
    )

    model = build_model(model_name, in_feats=data.x.size(1), args=args)

    model.fit(
        data,
        epochs       = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        verbose      = args.verbose,
        print_interval = args.print_interval,
    )

    result = model.evaluate(data, split="test")
    return result


# =========================================================
# Main
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()

    # ── 필수
    p.add_argument("--task_type",    type=str, default="classification",
                   choices=["classification"])
    p.add_argument("--dataset_name", type=str, required=True,
                   choices=list(DATASET_SETTINGS.keys()))
    p.add_argument("--sens_attr",    type=str, default=None,
                   help="생략 시 DATASET_SETTINGS에서 자동 설정")
    p.add_argument("--models",       nargs="+", default=["all"],
                   help="all 또는 fairgnn edits fmp gmmd 중 선택")

    # ── 실험
    p.add_argument("--device",    type=str,   default="cpu")
    p.add_argument("--seed",      type=int,   default=27)
    p.add_argument("--runs",      type=int,   default=10)
    p.add_argument("--epochs",    type=int,   default=1000)
    p.add_argument("--patience",  type=int,   default=100)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim",   type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--save_dir",     type=str,   default="outputs/compare")
    p.add_argument("--verbose",      action="store_true", default=False)
    p.add_argument("--print_interval", type=int, default=100)

    # ── val score
    p.add_argument("--val_tradeoff_dp", type=float, default=0.5)
    p.add_argument("--val_tradeoff_eo", type=float, default=0.5)

    # ── FairGNN 전용
    p.add_argument("--fairgnn_alpha", type=float, default=4.0)
    p.add_argument("--fairgnn_beta",  type=float, default=0.01)

    # ── EDITS 전용
    p.add_argument("--edits_lambda",        type=float, default=1.0)
    p.add_argument("--edits_debias_epochs", type=int,   default=200)

    # ── GMMD 전용
    p.add_argument("--gmmd_lambda_f", type=float, default=1.0)
    p.add_argument("--gmmd_lambda_s", type=float, default=0.1)
    p.add_argument("--gmmd_gamma",    type=float, default=1.0)

    # ── NIFTY 전용
    p.add_argument("--nifty_sim_coeff",        type=float, default=0.6,
                   help="similarity loss 강도 (논문 기본값 0.6)")
    p.add_argument("--nifty_drop_edge_rate",   type=float, default=0.1,
                   help="perturbation view edge drop rate")
    p.add_argument("--nifty_drop_feature_rate",type=float, default=0.1,
                   help="perturbation view feature drop rate")

    # ── FairVGNN 전용
    p.add_argument("--fairvgnn_eps",       type=float, default=0.1,
                   help="weight clamping threshold")
    p.add_argument("--fairvgnn_alpha_adv", type=float, default=1.0,
                   help="adversarial loss 강도")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # models 설정
    if "all" in args.models:
        models_to_run = ALL_MODELS
    else:
        models_to_run = [m.lower() for m in args.models]
        invalid = [m for m in models_to_run if m not in ALL_MODELS]
        if invalid:
            raise ValueError(f"Unknown models: {invalid}. "
                             f"Choose from {ALL_MODELS}")

    # sens_attr 자동 설정
    if args.sens_attr is None:
        args.sens_attr = DATASET_SETTINGS[args.dataset_name]["sens_attr"]

    print(f"\n{'='*70}")
    print(f"Dataset : {args.dataset_name}")
    print(f"Models  : {models_to_run}")
    print(f"Runs    : {args.runs}  |  Seed: {args.seed}")
    print(f"{'='*70}")

    all_results = []

    for model_name in models_to_run:
        print(f"\n{'─'*70}")
        print(f"Model: {model_name.upper()}")
        print(f"{'─'*70}")

        run_results = []

        for run in range(args.runs):
            seed = args.seed + run
            set_seed(seed)

            print(f"\n  [Run {run+1}/{args.runs}] seed={seed}")

            # 데이터 로드
            dataset_dict = load_dataset_from_args(
                dataset_name   = args.dataset_name,
                task_type      = args.task_type,
                sens_attr      = args.sens_attr,
                remove_leakage = True,
            )
            dataset_dict = prepare_classification_dataset(
                dataset_dict, train_per_class=500, seed=seed
            )

            # 실험
            result = run_one(model_name, dataset_dict, args)
            result["run"] = run + 1
            run_results.append(result)

            # 간단 출력
            acc = result.get("acc", float("nan"))
            dp  = result.get("dp",  float("nan"))
            eo  = result.get("eo",  float("nan"))
            print(f"    acc={acc:.4f} | dp={dp:.4f} | eo={eo:.4f}")

        # runs 집계
        df = pd.DataFrame(run_results)
        df["task"]  = args.task_type
        df["model"] = model_name

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "run"]

        summary = df.groupby(["task", "model"])[numeric_cols].agg(
            ["mean", "std"]
        )
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.reset_index()

        print(f"\n  [{model_name.upper()}] Summary:")
        for col in ["acc_mean", "acc_std", "dp_mean", "dp_std",
                    "eo_mean", "eo_std"]:
            if col in summary.columns:
                print(f"    {col}: {summary[col].values[0]:.4f}")

        all_results.append(summary)

    # 전체 결과 저장
    final_df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*70}")
    print("Final Results (all models)")
    print("="*70)
    print(final_df[["model",
                    "acc_mean", "acc_std",
                    "dp_mean",  "dp_std",
                    "eo_mean",  "eo_std"]].to_string(index=False))

    save_summary(final_df, args, save_dir=args.save_dir)

    print(f"\n{'='*70}")
    print(f"Done. Results saved to {args.save_dir}/")
