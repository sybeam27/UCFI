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
    FnCGNN,
    FnRGNN,
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
def filter_classification_splits_for_labeled_nodes(dataset_dict):
    """
    classification에서 labels == -1 을 unlabeled로 보고
    train/val/test split에서 제거한다.
    """
    labels = dataset_dict["labels"]
    valid_mask = labels >= 0   # only 0/1 are valid

    for split_key in ["idx_train", "idx_val", "idx_test"]:
        if split_key not in dataset_dict:
            continue
        idx = dataset_dict[split_key]
        dataset_dict[split_key] = idx[valid_mask[idx]]

    if "idx_sens_train" in dataset_dict and dataset_dict["idx_sens_train"] is not None:
        idx = dataset_dict["idx_sens_train"]
        dataset_dict["idx_sens_train"] = idx[valid_mask[idx]]

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
    backbones=("GCN", "GraphSAGE", "SGC"),
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

    for backbone in backbones:
        print("=" * 80)
        print(f"[Classification] Backbone = {backbone}")
        print("=" * 80)

        baseline = BaselineGNN(
            in_feats=data.x.size(1),
            h_feats=hidden_dim,
            device=device,
            task_type="classification",
            name=backbone,
            dropout=dropout,
            sgc_k=sgc_k,
        )
        baseline.fit(
            data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            verbose=True,
        )
        baseline_test = baseline.evaluate(data, split="test")
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
            lambda_struct=0.01,
            lambda_rep=0.05,
            lambda_out=0.05,
            drop_edge_rate_struct=0.15,
            ablate_struct=False,
            ablate_rep=False,
            ablate_out=False,
            val_tradeoff_dp=0.5,
            val_tradeoff_eo=0.5,
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

    return pd.DataFrame(results)


def run_regression_experiment(
    dataset_dict,
    device="cpu",
    backbones=("GCN", "GraphSAGE", "SGC"),
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

    for backbone in backbones:
        print("=" * 80)
        print(f"[Regression] Backbone = {backbone}")
        print("=" * 80)

        baseline = BaselineGNN(
            in_feats=data.x.size(1),
            h_feats=hidden_dim,
            device=device,
            task_type="regression",
            name=backbone,
            dropout=dropout,
            sgc_k=sgc_k,
        )
        baseline.fit(
            data,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            verbose=True,
        )
        baseline_test = baseline.evaluate(data, split="test")
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
            lambda_struct=0.01,
            lambda_rep=0.05,
            lambda_out=0.05,
            drop_edge_rate_struct=0.15,
            ablate_struct=False,
            ablate_rep=False,
            ablate_out=False,
            val_tradeoff_mae=1.0,
            val_tradeoff_bias=0.5,
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

    return pd.DataFrame(results)


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"], )
    parser.add_argument("--dataset_name", type=str, required=True, choices=["pokec_z", "pokec_n", "nba", "german"],)
    parser.add_argument("--sens_attr", type=str, required=True, help="e.g. region / gender / country / Gender",)
    parser.add_argument("--remove_leakage", action="store_true")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sgc_k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    # task에 따라 weight_decay 기본값 분리
    if args.weight_decay is None:
        if args.task_type == "classification":
            weight_decay = 0.0
        else:
            weight_decay = 1e-5
    else:
        weight_decay = args.weight_decay

    dataset_dict = load_dataset_from_args(
        dataset_name=args.dataset_name,
        task_type=args.task_type,
        sens_attr=args.sens_attr,
        remove_leakage=args.remove_leakage,
    )

    print("task_type:", args.task_type)
    print("label dtype:", dataset_dict["labels"].dtype)
    print("unique labels:", torch.unique(dataset_dict["labels"]))

    if args.task_type == "classification":
        unique, counts = torch.unique(dataset_dict["labels"], return_counts=True)
        print(dict(zip(unique.tolist(), counts.tolist())))

        dataset_dict = filter_classification_splits_for_labeled_nodes(dataset_dict)

        print("[After filtering] train/val/test sizes:",
              len(dataset_dict["idx_train"]), len(dataset_dict["idx_val"]), len(dataset_dict["idx_test"]),)

        # sanity check: split 안에는 0/1만 남아야 함
        for split_key in ["idx_train", "idx_val", "idx_test"]:
            idx = dataset_dict[split_key]
            split_unique = torch.unique(dataset_dict["labels"][idx])
            print(f"[{split_key}] unique labels:", split_unique)

        result_df = run_classification_experiment(
            dataset_dict=dataset_dict,
            device=args.device,
            backbones=("GCN", "GraphSAGE", "SGC"),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            sgc_k=args.sgc_k,
            lr=args.lr,
            weight_decay=weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
    else:

        result_df = run_regression_experiment(
            dataset_dict=dataset_dict,
            device=args.device,
            backbones=("GCN", "GraphSAGE", "SGC"),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            sgc_k=args.sgc_k,
            lr=args.lr,
            weight_decay=weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )

    print("\n[Final Results]")
    print(result_df)

    save_name = f"outputs/{args.dataset_name}_{args.task_type}_{args.sens_attr}_{args.seed}.csv"
    result_df.to_csv(save_name, index=False)
    print(f"\nSaved to: {save_name}")