#!/bin/bash
# run_exp.sh
# FairGate-GNN 전체 실험 스크립트
# backbone × model × dataset/sens_attr × ablation 조합을 순회
#
# 실행:
#   bash run_exp.sh               # 전체 (classification + regression)
#   bash run_exp.sh classification # 분류만
#   bash run_exp.sh regression     # 회귀만

set -e

# ──────────────────────────────────────────────
# 실행 모드 (인자로 지정, 기본: all)
# ──────────────────────────────────────────────

MODE=${1:-all}

if [[ "$MODE" != "all" && "$MODE" != "classification" && "$MODE" != "regression" ]]; then
    echo "[ERROR] Unknown mode: $MODE"
    echo "Usage: bash run_exp.sh [all|classification|regression]"
    exit 1
fi

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────

GPU=1
RUNS=10
SEED=27
EPOCHS=1000
HIDDEN=128
DROPOUT=0.1
LR=1e-3
PATIENCE=100
SAVE_DIR="outputs"

# ──────────────────────────────────────────────
# 실험 조합 정의
# ──────────────────────────────────────────────

BACKBONES=("GCN")
# BACKBONES=("GCN" "GraphSAGE" "SGC")

CLS_DATASETS=(
    "pokec_z:region"
    "pokec_n:region"
    "pokec_z:gender"
    "pokec_n:gender"
    "nba:country"
    "credit:Age"
    "bail:WHITE"
)

REG_DATASETS=(
    "pokec_z:region"
    "pokec_n:region"
    "pokec_z:gender"
    "pokec_n:gender"
    "nba:country"
    "credit:Age"
    "bail:WHITE"
)

# ──────────────────────────────────────────────
# multi/gate level ablation 조합
# 각 원소: "ablate_struct_flag|ablate_rep_flag|ablate_out_flag"
# 빈 문자열 = 해당 ablation 없음
# ──────────────────────────────────────────────

LEVEL_ABLATIONS=(
    "||"                          # full (ablation 없음)
    "--ablate_struct||"           # struct 제거
    "|--ablate_rep|"              # rep 제거
    "||--ablate_out"              # out 제거
)

# ──────────────────────────────────────────────
# gate 전용 ablation 조합
# 각 원소: "ablate_sbrs_flag|ablate_uncertainty_flag"
# ──────────────────────────────────────────────

GATE_ABLATIONS=(
    "|"                           # full (ablation 없음)
    "--ablate_sbrs|"              # uncertainty only
    "|--ablate_uncertainty"       # SBRS only
)

# ──────────────────────────────────────────────
# 공통 실행 함수
# ──────────────────────────────────────────────

run_exp() {
    local task=$1
    local dataset=$2
    local sens=$3
    local backbone=$4
    local model=$5
    local a_struct=$6
    local a_rep=$7
    local a_out=$8
    local a_sbrs=$9
    local a_unc=${10}

    echo "------------------------------------------------------------"
    echo "Task=$task | DS=$dataset | Sens=$sens | BB=$backbone | Model=$model"
    echo "  struct=$a_struct rep=$a_rep out=$a_out sbrs=$a_sbrs unc=$a_unc"
    echo "------------------------------------------------------------"

    python run_exp.py \
        --task_type     "$task"     \
        --dataset_name  "$dataset"  \
        --sens_attr     "$sens"     \
        --backbone      "$backbone" \
        --model         "$model"    \
        --device        "cuda:$GPU" \
        --runs          "$RUNS"     \
        --seed          "$SEED"     \
        --epochs        "$EPOCHS"   \
        --hidden_dim    "$HIDDEN"   \
        --dropout       "$DROPOUT"  \
        --lr            "$LR"       \
        --patience      "$PATIENCE" \
        --remove_leakage            \
        $a_struct $a_rep $a_out     \
        $a_sbrs $a_unc
}

# ──────────────────────────────────────────────
# 모델별 실행 (ablation 포함)
# ──────────────────────────────────────────────

run_all_combos() {
    local task=$1
    local dataset=$2
    local sens=$3
    local backbone=$4

    # ── baseline: ablation 없음
    run_exp "$task" "$dataset" "$sens" "$backbone" "baseline" "" "" "" "" ""

    # ── multi: level ablation 4가지
    for level_combo in "${LEVEL_ABLATIONS[@]}"; do
        IFS='|' read -r a_struct a_rep a_out <<< "$level_combo"
        run_exp "$task" "$dataset" "$sens" "$backbone" "multi" \
                "$a_struct" "$a_rep" "$a_out" "" ""
    done

    # ── gate: level 4 × gate 3 = 12가지
    for level_combo in "${LEVEL_ABLATIONS[@]}"; do
        IFS='|' read -r a_struct a_rep a_out <<< "$level_combo"
        for gate_combo in "${GATE_ABLATIONS[@]}"; do
            IFS='|' read -r a_sbrs a_unc <<< "$gate_combo"
            run_exp "$task" "$dataset" "$sens" "$backbone" "gate" \
                    "$a_struct" "$a_rep" "$a_out" "$a_sbrs" "$a_unc"
        done
    done
}

# ──────────────────────────────────────────────
# Classification 실험
# ──────────────────────────────────────────────

if [[ "$MODE" == "all" || "$MODE" == "classification" ]]; then
    echo ""
    echo "######## CLASSIFICATION ########"
    echo ""

    for ds_sens in "${CLS_DATASETS[@]}"; do
        dataset="${ds_sens%%:*}"
        sens="${ds_sens##*:}"
        for backbone in "${BACKBONES[@]}"; do
            run_all_combos "classification" "$dataset" "$sens" "$backbone"
        done
    done
fi

# ──────────────────────────────────────────────
# Regression 실험
# ──────────────────────────────────────────────

if [[ "$MODE" == "all" || "$MODE" == "regression" ]]; then
    echo ""
    echo "######## REGRESSION ########"
    echo ""

    for ds_sens in "${REG_DATASETS[@]}"; do
        dataset="${ds_sens%%:*}"
        sens="${ds_sens##*:}"
        for backbone in "${BACKBONES[@]}"; do
            run_all_combos "regression" "$dataset" "$sens" "$backbone"
        done
    done
fi

echo ""
echo "All experiments done. Results saved to: $SAVE_DIR/"
