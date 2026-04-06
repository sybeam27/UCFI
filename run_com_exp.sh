#!/bin/bash
# run_com_exp.sh
# 비교 모델(FairGNN, EDITS, FMP, GMMD, NIFTY, FairVGNN) 전체 실험 스크립트
#
# 실행:
#   bash run_com_exp.sh           # 전체 모델
#   bash run_com_exp.sh all       # 전체 모델
#   bash run_com_exp.sh gmmd      # 특정 모델만

set -e

# ──────────────────────────────────────────────
# 실행할 모델 (인자로 지정, 기본: all)
# ──────────────────────────────────────────────

MODELS=${1:-all}

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────

GPU=0
RUNS=10
SEED=27
EPOCHS=1000
HIDDEN=128
DROPOUT=0.1
LR=1e-3
PATIENCE=100
SAVE_DIR="outputs/compare"

# ──────────────────────────────────────────────
# 데이터셋 조합
# ──────────────────────────────────────────────

CLS_DATASETS=(
    "pokec_z:region"
    "pokec_n:region"
    "pokec_z:gender"
    "pokec_n:gender"
    "nba:country"
    "credit:Age"
    "bail:WHITE"
)

# ──────────────────────────────────────────────
# 실행 함수
# ──────────────────────────────────────────────

run_com_exp() {
    local dataset=$1
    local sens=$2
    local models=$3

    echo "========================================================"
    echo "Dataset: $dataset | Sens: $sens | Models: $models"
    echo "========================================================"

    python run_com_exp.py \
        --task_type    classification \
        --dataset_name "$dataset"     \
        --sens_attr    "$sens"        \
        --device       "cuda:$GPU"    \
        --runs         "$RUNS"        \
        --seed         "$SEED"        \
        --epochs       "$EPOCHS"      \
        --hidden_dim   "$HIDDEN"      \
        --dropout      "$DROPOUT"     \
        --lr           "$LR"          \
        --patience     "$PATIENCE"    \
        --save_dir     "$SAVE_DIR"    \
        --models       $models
}

# ──────────────────────────────────────────────
# 전체 데이터셋 조합 실행
# ──────────────────────────────────────────────

echo ""
echo "######## COMPARE MODELS: $MODELS ########"
echo ""

for ds_sens in "${CLS_DATASETS[@]}"; do
    dataset="${ds_sens%%:*}"
    sens="${ds_sens##*:}"
    run_com_exp "$dataset" "$sens" "$MODELS"
done

echo ""
echo "All compare experiments done. Results saved to: $SAVE_DIR/"