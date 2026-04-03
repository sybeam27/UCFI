import os
import torch
import random
import argparse

import numpy as np
import pandas as pd

from utils.model import (
    BaselineGNN,
    FnCGNN, SUMMIT_C,
    FnRGNN, SUMMIT_R,
)

from utils.loader import load_dataset_from_args, build_pyg_data_from_loader_dict


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_summary(summary: pd.DataFrame, args: argparse.Namespace, save_dir: str = "outputs"):
    """
    summary를 CSV로 저장. 동일 파일이 있으면 행을 추가하되,
    완전히 동일한 조건(모든 컬럼 값이 같은 행)은 덮어씀.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.task_type}_{args.dataset_name}.csv")

    # 실험 조건 컬럼 추가
    summary["dataset"]    = args.dataset_name
    summary["sens_attr"]  = args.sens_attr
    summary["runs"]       = args.runs
    summary["epochs"]     = args.epochs
    summary["lr"]         = args.lr
    summary["lambda_fair"] = args.lambda_fair
    summary["warm_up"]    = args.warm_up

    if args.model == "NaFn":
        summary["sbrs_threshold"]    = args.sbrs_threshold
        summary["lam"]               = args.lam
        summary["ablate_sbrs"]       = args.ablate_sbrs
        summary["ablate_uncertainty"] = args.ablate_uncertainty

    # 조건 식별 키
    key_cols = ["dataset", "sens_attr", "task", "model", "backbone",
                "runs", "epochs", "lr", "lambda_fair", "warm_up"]

    if os.path.exists(save_path):
        existing = pd.read_csv(save_path)

        # 동일 조건 행 제거 후 새 결과 추가
        merge_keys = [c for c in key_cols if c in existing.columns and c in summary.columns]
        merged = existing.merge(summary[merge_keys], on=merge_keys, how="left", indicator=True)
        existing = existing[merged["_merge"] == "left_only"].drop(columns=["_merge"], errors="ignore")

        final = pd.concat([existing, summary], ignore_index=True)
    else:
        final = summary

    # ── 저장 직전 수치 컬럼 소수점 4자리 반올림
    numeric_cols = final.select_dtypes(include="number").columns
    final[numeric_cols] = final[numeric_cols].round(4)

    final.to_csv(save_path, index=False)
    print(f"[Save] {save_path} ({len(final)} rows)")



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

def run_classification_experiment(dataset_dict, args):
    set_seed(args.seed)
    data = build_pyg_data_from_loader_dict(
        dataset_dict, device=args.device, task_type="classification"
    )

    print("=" * 80)
    print(f"[Classification] model={args.model} | backbone={args.backbone}")
    print("=" * 80)

    if args.model == "baseline":
        md = BaselineGNN(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, task_type="classification",
            name=args.backbone, dropout=args.dropout, sgc_k=args.sgc_k,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    elif args.model == "multi":
        md = FnCGNN(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, name=args.backbone,
            dropout=args.dropout, sgc_k=args.sgc_k,
            drop_edge_rate_struct=args.drop_edge_rate_struct,
            lambda_fair=args.lambda_fair, warm_up=args.warm_up,
            ablate_struct=args.ablate_struct,
            ablate_rep=args.ablate_rep,
            ablate_out=args.ablate_out,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    elif args.model == "summit":
        md = SUMMIT_C(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, name=args.backbone,
            dropout=args.dropout, sgc_k=args.sgc_k,
            drop_edge_rate_struct=args.drop_edge_rate_struct,
            lambda_fair=args.lambda_fair,
            ablate_struct=args.ablate_struct,
            ablate_rep=args.ablate_rep,
            ablate_out=args.ablate_out,
            val_tradeoff_dp=args.val_tradeoff_dp,
            val_tradeoff_eo=args.val_tradeoff_eo,
            sbrs_threshold=args.sbrs_threshold,
            lam=args.lam,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            warm_up=args.warm_up,
            ablate_sbrs=args.ablate_sbrs,
            ablate_uncertainty=args.ablate_uncertainty,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    test_result = md.evaluate(data, split="test")
    return pd.DataFrame([{
        "task": "classification",
        "model": args.model,
        "backbone": args.backbone,
        **test_result,
    }])

def run_regression_experiment(dataset_dict, args):
    set_seed(args.seed)
    data = build_pyg_data_from_loader_dict(
        dataset_dict, device=args.device, task_type="regression"
    )

    print("=" * 80)
    print(f"[Regression] model={args.model} | backbone={args.backbone}")
    print("=" * 80)

    if args.model == "baseline":
        md = BaselineGNN(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, task_type="regression",
            name=args.backbone, dropout=args.dropout, sgc_k=args.sgc_k,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    elif args.model == "multi":
        md = FnRGNN(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, name=args.backbone,
            dropout=args.dropout, sgc_k=args.sgc_k,
            drop_edge_rate_struct=args.drop_edge_rate_struct,
            lambda_fair=args.lambda_fair, warm_up=args.warm_up,
            ablate_struct=args.ablate_struct,
            ablate_rep=args.ablate_rep,
            ablate_out=args.ablate_out,
            val_tradeoff_mae=args.val_tradeoff_mae,
            val_tradeoff_bias=args.val_tradeoff_bias,
            val_tradeoff_mean_pred=args.val_tradeoff_mean_pred,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    elif args.model == "summit":
        md = SUMMIT_R(
            in_feats=data.x.size(1), h_feats=args.hidden_dim,
            device=args.device, name=args.backbone,
            dropout=args.dropout, sgc_k=args.sgc_k,
            drop_edge_rate_struct=args.drop_edge_rate_struct,
            lambda_fair=args.lambda_fair,
            ablate_struct=args.ablate_struct,
            ablate_rep=args.ablate_rep,
            ablate_out=args.ablate_out,
            val_tradeoff_mae=args.val_tradeoff_mae,
            val_tradeoff_bias=args.val_tradeoff_bias,
            val_tradeoff_mean_pred=args.val_tradeoff_mean_pred,
            sbrs_threshold=args.sbrs_threshold,
            lam=args.lam,
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            warm_up=args.warm_up,
            ablate_sbrs=args.ablate_sbrs,
            ablate_uncertainty=args.ablate_uncertainty,
        )
        md.fit(data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               patience=args.patience, verbose=True)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    test_result = md.evaluate(data, split="test")
    return pd.DataFrame([{
        "task": "regression",
        "model": args.model,
        "backbone": args.backbone,
        **test_result,
    }])


def parse_args():
    parser = argparse.ArgumentParser()

    # ── 필수
    parser.add_argument("--task_type", type=str, required=True,
                        choices=["classification", "regression"])
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["pokec_z", "pokec_n", "nba", "german"])
    parser.add_argument("--sens_attr", type=str, required=True,
                        help="e.g. region / gender / country / Gender")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["GCN", "GraphSAGE", "SGC"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["baseline", "multi", "summit"])

    # ── 데이터
    parser.add_argument("--remove_leakage", action="store_true")

    # ── 모델 구조
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--sgc_k",      type=int,   default=2)

    # ── 학습
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs",       type=int,   default=1000)
    parser.add_argument("--patience",     type=int,   default=100)
    parser.add_argument("--seed",         type=int,   default=27)
    parser.add_argument("--runs",         type=int,   default=5)

    # ── 분류 val score
    parser.add_argument("--val_tradeoff_dp",  type=float, default=0.5)
    parser.add_argument("--val_tradeoff_eo",  type=float, default=0.5)

    # ── 회귀 val score
    parser.add_argument("--val_tradeoff_mae",       type=float, default=1.0)
    parser.add_argument("--val_tradeoff_bias",      type=float, default=0.5)
    parser.add_argument("--val_tradeoff_mean_pred", type=float, default=0.5)

    # ── Fn / NaFn 공통
    parser.add_argument("--lambda_fair",           type=float, default=0.5)
    parser.add_argument("--warm_up",               type=int,   default=100)
    parser.add_argument("--drop_edge_rate_struct",  type=float, default=0.1)
    parser.add_argument("--ablate_struct", action="store_true")
    parser.add_argument("--ablate_rep",    action="store_true")
    parser.add_argument("--ablate_out",    action="store_true")

    # ── NaFn 전용: FIPS 게이팅
    parser.add_argument("--sbrs_threshold", type=float, default=0.5)
    parser.add_argument("--lam",            type=float, default=1.0)
    parser.add_argument("--min_weight",     type=float, default=0.5)
    parser.add_argument("--max_weight",     type=float, default=2.0)

    # ── NaFn 전용: FIPS ablation
    parser.add_argument("--ablate_sbrs",        action="store_true", help="Uncertainty only")
    parser.add_argument("--ablate_uncertainty", action="store_true", help="SBRS only")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results = []
    for run in range(args.runs):
        print(f"\n{'='*80}")
        print(f"Run {run+1}/{args.runs}")
        print(f"{'='*80}")

        run_args      = argparse.Namespace(**vars(args))
        run_args.seed = args.seed + run

        # ── load_dataset_from_args는 키워드 인자로 직접 전달
        dataset_dict = load_dataset_from_args(
            dataset_name   = args.dataset_name,
            task_type      = args.task_type,
            sens_attr      = args.sens_attr,
            remove_leakage = args.remove_leakage,
        )

        if args.task_type == "classification":
            if run == 0:
                print("task_type:", args.task_type)
                print("label dtype:", dataset_dict["labels"].dtype)
                unique, counts = torch.unique(dataset_dict["labels"], return_counts=True)
                print(dict(zip(unique.tolist(), counts.tolist())))

            dataset_dict = prepare_classification_dataset(
                dataset_dict,
                train_per_class=500,
                seed=run_args.seed,
            )

            if run == 0:
                print("[After filtering] train/val/test sizes:",
                      len(dataset_dict["idx_train"]),
                      len(dataset_dict["idx_val"]),
                      len(dataset_dict["idx_test"]))

            df = run_classification_experiment(dataset_dict, run_args)

        else:
            df = run_regression_experiment(dataset_dict, run_args)

        df["run"] = run + 1
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    numeric_cols = final_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "run"]

    summary = final_df.groupby(["task", "model", "backbone"])[numeric_cols].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(summary.to_string(index=False))

    # ── 저장
    save_summary(summary, args)