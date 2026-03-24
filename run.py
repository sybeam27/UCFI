import os
import json
import argparse
import pandas as pd
import torch

from data.loader import load_fair_graph_dataset
from utils.graph_utils import set_seed, prepare_pyg_data

from trainers.baseline_gcn_train import train_baseline_gcn_model
from trainers.gcn_groupnorm_train import train_gcn_groupnorm_model
from trainers.gcn_mmd_train import train_gcn_mmd_model
from trainers.gcn_mmd_groupnorm_train import train_gcn_mmd_groupnorm_model
from trainers.fnrgnn_train import train_fnrgnn_model
from trainers.gcn_groupnorm_selective_train import train_gcn_groupnorm_selective_model
from trainers.gcn_groupnorm_softweighted_train import train_gcn_groupnorm_softweighted_model

def parse_args():
    parser = argparse.ArgumentParser(description="Run GCN fairness experiments")

    # basic
    parser.add_argument("--model", type=str, default="baseline_gcn",
                        choices=[
                            "baseline_gcn",
                            "gcn_groupnorm",
                            "gcn_groupnorm_selective",
                            "gcn_groupnorm_softweighted",
                            "gcn_mmd",
                            "gcn_mmd_groupnorm",
                            "fnrgnn",
                        ])
    parser.add_argument("--dataset", type=str, default="pokec_z",
                        choices=["pokec_z", "pokec_n", "nba", "german"])
    parser.add_argument("--root", type=str, default="./data/pokec/")
    parser.add_argument("--device", type=str, default=None,
                        help="e.g. cpu, cuda, cuda:0, cuda:1")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--gpu", type=int, default=1)

    # dataset split / loading
    parser.add_argument("--sens_attr", type=str, default=None)
    parser.add_argument("--predict_attr", type=str, default=None)
    parser.add_argument("--label_number", type=int, default=500)
    parser.add_argument("--sens_number", type=int, default=500)
    parser.add_argument("--sens_ratio", type=float, default=0.2)
    parser.add_argument("--use_all_sensitive", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--test_idx_as_val", action="store_true")
    parser.add_argument("--sens_binary", action="store_true",
                        help="binarize sensitive attribute for binary fairness metrics")

    # model/training
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--selection", type=str, default="tradeoff",
                        choices=["f1", "tradeoff", "fair"])
    parser.add_argument("--use_pos_weight", action="store_true")

    # fairness-specific
    parser.add_argument("--lambda_dist", type=float, default=0.05)
    parser.add_argument("--lambda_mmd", type=float, default=0.05)
    parser.add_argument("--mmd_sample", type=int, default=256)
    parser.add_argument("--edge_penalty", type=float, default=0.5)

    # output
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--save_history", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    # selective
    parser.add_argument("--lambda_unc", type=float, default=1.0)
    parser.add_argument("--num_perturbations", type=int, default=4)
    parser.add_argument("--drop_edge_rate", type=float, default=0.1)

    parser.add_argument("--risk_boundary", type=float, default=1.0)
    parser.add_argument("--risk_exposure", type=float, default=1.0)
    parser.add_argument("--risk_influence", type=float, default=1.0)

    parser.add_argument("--priority_alpha", type=float, default=1.0)
    parser.add_argument("--priority_beta", type=float, default=1.0)

    parser.add_argument("--priority_mode", type=str, default="topk", choices=["topk", "threshold"])
    parser.add_argument("--priority_k_frac", type=float, default=0.2)
    parser.add_argument("--priority_threshold", type=float, default=None)

    # soft
    parser.add_argument("--weight_transform", type=str, default="linear",
                    choices=["linear", "power", "sigmoid", "softmax"])
    parser.add_argument("--weight_power", type=float, default=1.0)
    parser.add_argument("--weight_temperature", type=float, default=1.0)
    parser.add_argument("--min_fair_weight", type=float, default=0.05)
    parser.add_argument("--normalize_fair_weights", type=str, default="mean1",
                        choices=["none", "mean1", "sum1"])
    parser.add_argument("--detach_fair_weights", action="store_true")

    return parser.parse_args()


def infer_default_attrs(dataset_name):
    if dataset_name in ["pokec_z", "pokec_n"]:
        return "region", "I_am_working_in_field"
    if dataset_name == "nba":
        return "country", "SALARY"
    if dataset_name == "german":
        return "Gender", "GoodCustomer"
    raise ValueError(f"Unknown dataset: {dataset_name}")


def infer_default_root(dataset_name, user_root):
    if user_root is not None and user_root != "":
        return user_root
    if dataset_name in ["pokec_z", "pokec_n"]:
        return "./data/pokec/"
    if dataset_name == "nba":
        return "./data/NBA/"
    if dataset_name == "german":
        return "./data/NIFTY/"
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_run_name(args):
    parts = [
        args.model,
        args.dataset,
        f"seed{args.seed}",
        f"hd{args.hidden_dim}",
        f"do{args.dropout}",
        f"lr{args.lr}",
        f"wd{args.weight_decay}",
        f"ep{args.epochs}",
    ]

    if args.model in ["gcn_groupnorm", "gcn_mmd_groupnorm", "fnrgnn"]:
        parts.append(f"ld{args.lambda_dist}")
    if args.model in ["gcn_mmd", "gcn_mmd_groupnorm", "fnrgnn"]:
        parts.append(f"lm{args.lambda_mmd}")
        parts.append(f"mmd{args.mmd_sample}")
    if args.model == "fnrgnn":
        parts.append(f"edge{args.edge_penalty}")
    if args.model == "gcn_groupnorm_selective":
        parts.append(f"ld{args.lambda_dist}")
        parts.append(f"lu{args.lambda_unc}")
        parts.append(f"pert{args.num_perturbations}")
        parts.append(f"dropedge{args.drop_edge_rate}")
        parts.append(f"pk{args.priority_k_frac}")

    return "_".join(map(str, parts))


def print_result(title, result_dict):
    print(f"\n=== {title} ===")
    for k, v in result_dict.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.device is None:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    sens_attr_default, predict_attr_default = infer_default_attrs(args.dataset)
    sens_attr = args.sens_attr or sens_attr_default
    predict_attr = args.predict_attr or predict_attr_default
    root = infer_default_root(args.dataset, args.root)

    print("device:", device)
    print("model:", args.model)
    print("dataset:", args.dataset)
    print("root:", root)

    data_dict = load_fair_graph_dataset(
        dataset_name=args.dataset,
        root=root,
        sens_attr=sens_attr,
        predict_attr=predict_attr,
        label_number=args.label_number,
        sens_number=args.sens_number,
        use_all_sensitive=args.use_all_sensitive,
        sens_ratio=args.sens_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_idx_as_val=args.test_idx_as_val,
    )

    pyg_data = prepare_pyg_data(
        data_dict,
        device=device,
        sens_binary=args.sens_binary
    )
    nfeat = pyg_data.x.shape[1]
    print("nfeat:", nfeat)

    if args.model == "baseline_gcn":
        model, hist_df, val_result, test_result = train_baseline_gcn_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "gcn_groupnorm":
        model, hist_df, val_result, test_result = train_gcn_groupnorm_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_dist=args.lambda_dist,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "gcn_groupnorm_selective":
        model, hist_df, val_result, test_result = train_gcn_groupnorm_selective_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_dist=args.lambda_dist,
            lambda_unc=args.lambda_unc,
            num_perturbations=args.num_perturbations,
            drop_edge_rate=args.drop_edge_rate,
            risk_weights=(args.risk_boundary, args.risk_exposure, args.risk_influence),
            priority_exponents=(args.priority_alpha, args.priority_beta),
            priority_mode=args.priority_mode,
            priority_k_frac=args.priority_k_frac,
            priority_threshold=args.priority_threshold,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "gcn_groupnorm_softweighted":
        model, hist_df, val_result, test_result = train_gcn_groupnorm_softweighted_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_dist=args.lambda_dist,
            lambda_unc=args.lambda_unc,
            num_perturbations=args.num_perturbations,
            drop_edge_rate=args.drop_edge_rate,
            risk_weights=(args.risk_boundary, args.risk_exposure, args.risk_influence),
            priority_exponents=(args.priority_alpha, args.priority_beta),
            weight_transform=args.weight_transform,
            weight_power=args.weight_power,
            weight_temperature=args.weight_temperature,
            min_fair_weight=args.min_fair_weight,
            normalize_fair_weights=args.normalize_fair_weights,
            detach_fair_weights=args.detach_fair_weights,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "gcn_mmd":
        model, hist_df, val_result, test_result = train_gcn_mmd_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_mmd=args.lambda_mmd,
            mmd_sample=args.mmd_sample,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "gcn_mmd_groupnorm":
        model, hist_df, val_result, test_result = train_gcn_mmd_groupnorm_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_mmd=args.lambda_mmd,
            lambda_dist=args.lambda_dist,
            mmd_sample=args.mmd_sample,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    elif args.model == "fnrgnn":
        model, hist_df, val_result, test_result = train_fnrgnn_model(
            data=pyg_data,
            nfeat=nfeat,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lambda_mmd=args.lambda_mmd,
            lambda_dist=args.lambda_dist,
            mmd_sample=args.mmd_sample,
            edge_penalty=args.edge_penalty,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            verbose=args.verbose,
            selection=args.selection,
            use_pos_weight=args.use_pos_weight,
            device=device,
        )

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print_result("Validation", val_result)
    print_result("Test", test_result)

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = build_run_name(args)

    summary = {
        "run_name": run_name,
        "args": vars(args),
        "val_result": val_result,
        "test_result": test_result,
    }

    summary_path = os.path.join(args.save_dir, f"{run_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_history:
        hist_path = os.path.join(args.save_dir, f"{run_name}_history.csv")
        hist_df.to_csv(hist_path, index=False)
        print("saved history:", hist_path)

    if args.save_model:
        model_path = os.path.join(args.save_dir, f"{run_name}_model.pt")
        torch.save(model.state_dict(), model_path)
        print("saved model:", model_path)

    print("saved summary:", summary_path)


if __name__ == "__main__":
    main()