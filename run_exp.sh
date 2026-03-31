# classification
python run_exp.py \
    --task_type classification \
    --dataset_name pokec_z \
    --sens_attr region \
    --remove_leakage \
    --backbone GraphSAGE \
    --hidden_dim 256 \
    --dropout 0.5 \
    --device cuda:1 \
    --runs 3

python run_exp.py --task_type classification --dataset_name pokec_z --sens_attr gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr region --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type classification --dataset_name german --sens_attr Gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type classification --dataset_name nba --sens_attr country --remove_leakage --backbone GraphSAGE --device cuda:1

# regression
python run_exp.py \
    --task_type regression \
    --dataset_name pokec_z \
    --sens_attr region \
    --remove_leakage --baseline \
    --backbone GraphSAGE \
    --hidden_dim 256 \
    --dropout 0.5 \
    --device cuda:1 \
    --runs 5

python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr region --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type regression --dataset_name german --sens_attr Gender --remove_leakage --backbone GraphSAGE --device cuda:1
python run_exp.py --task_type regression --dataset_name nba --sens_attr country --remove_leakage --backbone GraphSAGE--device cuda:1
