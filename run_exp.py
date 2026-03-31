import argparse
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import (
    build_pyg_data_from_loader_dict,
    build_backbone,
    FnCGNN, NAFnCGNN,
    FnRGNN, NAFnRGNN,
)

from utils.metrics import evaluate_pyg_model
from utils.loader import load_dataset_from_args


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# unlabeled 확인!
def prepare_classification_dataset(dataset_dict, train_per_class=500, seed=42):
    """
    FairGNN/NIFTY 등 주요 fairness GNN 논문의 pokec 처리 방식을 따름.

    1. label binarize: label > 1  →  1, label == 0 → 0, label == -1 → 제거
    2. split 재구성:
       - train: 각 클래스별 min(train_per_class, 50%) 개
       - val:   labeled 나머지의 50%
       - test:  labeled 나머지의 50%
    """
    labels = dataset_dict["labels"].clone()

    # Step 1: binarize  (>1 → 1, -1 → 제거)
    labels[labels > 1] = 1
    dataset_dict["labels"] = labels

    valid_mask = labels >= 0  # -1 제거용 마스크
    labeled_idx = torch.where(valid_mask)[0]

    # Step 2: class별로 분리
    idx_0 = labeled_idx[labels[labeled_idx] == 0]
    idx_1 = labeled_idx[labels[labeled_idx] == 1]

    rng = torch.Generator()
    rng.manual_seed(seed)

    def shuffle(idx):
        return idx[torch.randperm(len(idx), generator=rng)]

    idx_0 = shuffle(idx_0)
    idx_1 = shuffle(idx_1)

    # Step 3: train — 각 클래스별 min(train_per_class, 50%)
    n_train_0 = min(train_per_class, int(len(idx_0) * 0.5))
    n_train_1 = min(train_per_class, int(len(idx_1) * 0.5))

    train_0, rest_0 = idx_0[:n_train_0], idx_0[n_train_0:]
    train_1, rest_1 = idx_1[:n_train_1], idx_1[n_train_1:]

    idx_train = torch.cat([train_0, train_1])

    # Step 4: val/test — 나머지를 50/50
    rest = shuffle(torch.cat([rest_0, rest_1]))
    mid  = len(rest) // 2
    idx_val  = rest[:mid]
    idx_test = rest[mid:]

    dataset_dict["idx_train"]      = idx_train
    dataset_dict["idx_val"]        = idx_val
    dataset_dict["idx_test"]       = idx_test
    dataset_dict["idx_sens_train"] = idx_train  # sens_train도 train과 동일하게

    print(f"[prepare] binarize 완료 | "
          f"class 0: {len(idx_0)}개, class 1: {len(idx_1)}개")
    print(f"[prepare] train={len(idx_train)} "
          f"(0:{n_train_0}, 1:{n_train_1}) | "
          f"val={len(idx_val)} | test={len(idx_test)}")

    return dataset_dict

# =========================================================
# Baseline Wrapper
# =========================================================
class BaselineGNN:
    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        task_type="classification",
        name="GCN",
        dropout=0.1,
        sgc_k=2,
    ):
        assert task_type in ["classification", "regression"]
        assert name in ["GCN", "GraphSAGE", "SGC"]

        self.name = f"{name}/Baseline"
        self.backbone_name = name
        self.task_type = task_type
        self.device = device

        self.model = build_backbone(
            name=name,
            in_feats=in_feats,
            h_feats=h_feats,
            dropout=dropout,
            sgc_k=sgc_k,
        ).to(device)

    def _build_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _build_criterion(self):
        if self.task_type == "classification":
            return nn.BCEWithLogitsLoss()
        return nn.MSELoss()

    def _compute_val_score(self, val_result):
        if self.task_type == "classification":
            return float(val_result.get("acc", 0.0))
        else:
            return -float(val_result.get("mae", float("inf")))

    def train_step(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()

        out = self.model(data).view(-1)
        labels = data.y.float()
        idx_train = data.idx_train

        loss = criterion(out[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        return {
            "total_loss": float(loss.item()),
            "task_loss": float(loss.item()),
        }

    def fit(
        self,
        data,
        epochs=300,
        lr=1e-3,
        weight_decay=0.0,
        patience=50,
        verbose=True,
        print_interval=50,
    ):
        optimizer = self._build_optimizer(lr=lr, weight_decay=weight_decay)
        criterion = self._build_criterion()

        best_val_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        counter = 0

        for epoch in range(epochs):
            train_info = self.train_step(data, optimizer, criterion)

            val_result = evaluate_pyg_model(
                self.model,
                data,
                split="val",
                task_type=self.task_type,
            )
            val_score = self._compute_val_score(val_result)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                train_result = evaluate_pyg_model(
                    self.model,
                    data,
                    split="train",
                    task_type=self.task_type,
                )
                print(
                    f"[{self.name}] "
                    f"Epoch {epoch+1:04d} | "
                    f"Loss {train_info['total_loss']:.4f} | "
                    f"Train {train_result} | "
                    f"Val {val_result} | "
                    f"ValScore {val_score:.4f}"
                )

            if counter >= patience:
                break

        self.model.load_state_dict(best_state)

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model,
            data,
            split=split,
            task_type=self.task_type,
        )


# =========================================================
# Experiment runners
# =========================================================
def run_classification_experiment(
    dataset_dict,
    device="cpu",
    backbone="GCN",
    baseline = False,
    hidden_dim=64,
    dropout=0.1,
    sgc_k=2,
    lr=1e-3,
    weight_decay=0.0,
    epochs=300,
    patience=50,
    seed=42,
):
    set_seed(seed)

    data = build_pyg_data_from_loader_dict(
        dataset_dict,
        device=device,
        task_type="classification",
    )

    results = []

    print("=" * 80)
    print(f"[Classification] Backbone = {backbone}")
    print("=" * 80)

    if baseline:
        baseline_md = BaselineGNN(
            in_feats=data.x.size(1),
            h_feats=hidden_dim,
            device=device,
            task_type="classification",
            name=backbone,
            dropout=dropout,
            sgc_k=sgc_k,
        )
        baseline_md.fit(
            data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            verbose=True,
        )
        baseline_test = baseline_md.evaluate(data, split="test")
        results.append({
            "task": "classification",
            "model": "Baseline",
            "backbone": backbone,
            **baseline_test,
        })

    fnc = FnCGNN(
        in_feats=data.x.size(1),
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,

        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,

        drop_edge_rate_struct=0.1,

        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,

        val_tradeoff_dp=0.3,
        val_tradeoff_eo=0.3,
    )
    fnc.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=True,
    )
    fnc_test = fnc.evaluate(data, split="test")
    results.append({
        "task": "classification",
        "model": "FnCGNN",
        "backbone": backbone,
        **fnc_test,
    })

    # ── NAFnRGNN 추가
    na_fnc = NAFnCGNN(
        in_feats=data.x.size(1),
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,
        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,
        drop_edge_rate_struct=0.1,
        val_tradeoff_dp=0.3,
        val_tradeoff_eo=0.3,
        hub_percentile=95,      # 80 → 95 (상위 5%만 허브)
        isolate_percentile=10,  # 20 → 10 (하위 10%만 고립)
        boundary_threshold=0.3, # 유지
        hub_weight=2.0,
        boundary_weight=1.5,
        isolate_weight=0.5,
    )
    na_fnc.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=True,
    )
    na_fnc_test = na_fnc.evaluate(data, split="test")
    results.append({
        "task": "classification",
        "model": "NAFnCGNN",
        "backbone": backbone,
        **na_fnc_test,
    })

    return pd.DataFrame(results)


def run_regression_experiment(
    dataset_dict,
    device="cpu",
    backbone="GCN",
    baseline = False,
    hidden_dim=64,
    dropout=0.1,
    sgc_k=2,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=300,
    patience=50,
    seed=42,
):
    set_seed(seed)

    data = build_pyg_data_from_loader_dict(
        dataset_dict,
        device=device,
        task_type="regression",
    )

    results = []

    print("=" * 80)
    print(f"[Regression] Backbone = {backbone}")
    print("=" * 80)

    if baseline:
        baseline_md = BaselineGNN(
            in_feats=data.x.size(1),
            h_feats=hidden_dim,
            device=device,
            task_type="regression",
            name=backbone,
            dropout=dropout,
            sgc_k=sgc_k,
        )
        baseline_md.fit(
            data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            verbose=True,
        )
        baseline_test = baseline_md.evaluate(data, split="test")
        results.append({
            "task": "regression",
            "model": "Baseline",
            "backbone": backbone,
            **baseline_test,
        })

    fnr = FnRGNN(
        in_feats=data.x.size(1),
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,

        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,

        drop_edge_rate_struct=0.1,

        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,

        val_tradeoff_mae=1.0,
        val_tradeoff_bias=1.0,
        val_tradeoff_mean_pred=0.5,
    )
    fnr.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=True,
    )
    fnr_test = fnr.evaluate(data, split="test")
    results.append({
        "task": "regression",
        "model": "FnRGNN",
        "backbone": backbone,
        **fnr_test,
    })

    # ── NAFnRGNN 추가
    na_fnr = NAFnRGNN(
        in_feats=data.x.size(1),
        h_feats=hidden_dim,
        device=device,
        name=backbone,
        dropout=dropout,
        sgc_k=sgc_k,
        lambda_struct=0.001,
        lambda_rep=0.01,
        lambda_out=0.1,
        drop_edge_rate_struct=0.1,
        ablate_struct=False,
        ablate_rep=False,
        ablate_out=False,
        val_tradeoff_mae=1.0,
        val_tradeoff_bias=1.0,
        val_tradeoff_mean_pred=0.5,
    )
    na_fnr.fit(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=True,
    )
    na_fnr_test = na_fnr.evaluate(data, split="test")
    results.append({
        "task": "regression",
        "model": "NAFnRGNN",
        "backbone": backbone,
        **na_fnr_test,
    })

    return pd.DataFrame(results)


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"], )
    parser.add_argument("--dataset_name", type=str, required=True, choices=["pokec_z", "pokec_n", "nba", "german"],)
    parser.add_argument("--sens_attr", type=str, required=True, help="e.g. region / gender / country / Gender",)
    parser.add_argument("--remove_leakage", action="store_true")\
    
    parser.add_argument("--backbone", type=str, required=True, choices=["GCN", "GraphSAGE", "SGC"], )
    parser.add_argument("--baseline", action="store_true")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sgc_k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--runs", type=int, default=5, help="runs")


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    if args.weight_decay is None:
        weight_decay = 0.0 if args.task_type == "classification" else 1e-5
    else:
        weight_decay = args.weight_decay

    # seed 목록 생성: args.seed를 시작점으로 runs개 생성
    seeds = [args.seed + i for i in range(args.runs)]

    all_results = []  # 각 run의 DataFrame을 쌓을 리스트

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"[Run {run_idx+1}/{args.runs}] seed={seed}")
        print(f"{'='*80}")

        dataset_dict = load_dataset_from_args(
            dataset_name=args.dataset_name,
            task_type=args.task_type,
            sens_attr=args.sens_attr,
            remove_leakage=args.remove_leakage,
        )

        if args.task_type == "classification":
            if run_idx == 0:
                # 첫 run에서만 레이블 분포 출력
                print("task_type:", args.task_type)
                print("label dtype:", dataset_dict["labels"].dtype)
                unique, counts = torch.unique(dataset_dict["labels"], return_counts=True)
                print(dict(zip(unique.tolist(), counts.tolist())))

            dataset_dict = prepare_classification_dataset(
                dataset_dict,
                train_per_class=500,
                seed=seed,  # run마다 다른 seed → 다른 split
            )

            if run_idx == 0:
                print("[After filtering] train/val/test sizes:",
                      len(dataset_dict["idx_train"]),
                      len(dataset_dict["idx_val"]),
                      len(dataset_dict["idx_test"]))

            result_df = run_classification_experiment(
                dataset_dict=dataset_dict,
                device=args.device,
                backbone=args.backbone,
                baseline=args.baseline,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                sgc_k=args.sgc_k,
                lr=args.lr,
                weight_decay=weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                seed=seed,
            )
        else:
            result_df = run_regression_experiment(
                dataset_dict=dataset_dict,
                device=args.device,
                backbone=args.backbone,
                baseline=args.baseline,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                sgc_k=args.sgc_k,
                lr=args.lr,
                weight_decay=weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                seed=seed,
            )

        result_df["run"] = run_idx + 1
        result_df["seed"] = seed
        all_results.append(result_df)

    # 전체 raw 결과
    raw_df = pd.concat(all_results, ignore_index=True)

    # 평균 ± 표준편차 계산
    metric_cols = [c for c in raw_df.columns if c not in ["task", "model", "backbone", "run", "seed"]]
    
    summary_rows = []
    for (task, model, backbone), group in raw_df.groupby(["task", "model", "backbone"]):
        row = {"task": task, "model": model, "backbone": backbone}
        for col in metric_cols:
            mean = group[col].mean()
            std  = group[col].std()
            row[f"{col}_mean"] = round(mean, 4)
            row[f"{col}_std"]  = round(std, 4)
            row[col]           = f"{mean:.4f}±{std:.4f}"  # 보기 편한 형식
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n[Final Results — Mean ± Std]")
    # 보기 편한 컬럼만 출력
    display_cols = ["task", "model", "backbone"] + metric_cols
    print(summary_df[display_cols].to_string(index=False))

    # 저장: raw + summary 둘 다
    base_name = f"outputs/{args.dataset_name}_{args.task_type}_{args.sens_attr}_runs{args.runs}"
    raw_df.to_csv(f"{base_name}_raw.csv", index=False)
    summary_df.to_csv(f"{base_name}_summary.csv", index=False)
    print(f"\nSaved: {base_name}_raw.csv")
    print(f"Saved: {base_name}_summary.csv")