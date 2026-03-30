# classification
python run_experiments.py --task_type classification --dataset_name pokec_z --sens_attr region --remove_leakage --device cuda:1
python run_experiments.py --task_type classification --dataset_name pokec_z --sens_attr gender --remove_leakage --device cuda:1
python run_experiments.py --task_type classification --dataset_name pokec_n --sens_attr region --remove_leakage --device cuda:1
python run_experiments.py --task_type classification --dataset_name pokec_n --sens_attr gender --remove_leakage --device cuda:1
python run_experiments.py --task_type classification --dataset_name german --sens_attr Gender --remove_leakage --device cuda:1
python run_experiments.py --task_type classification --dataset_name nba --sens_attr country --remove_leakage --device cuda:1

# regression
python run_experiments.py --task_type regression --dataset_name pokec_z --sens_attr region --remove_leakage --device cuda:1
python run_experiments.py --task_type regression --dataset_name pokec_z --sens_attr gender --remove_leakage --device cuda:1
python run_experiments.py --task_type regression --dataset_name pokec_n --sens_attr region --remove_leakage --device cuda:1
python run_experiments.py --task_type regression --dataset_name pokec_n --sens_attr gender --remove_leakage --device cuda:1
python run_experiments.py --task_type regression --dataset_name german --sens_attr Gender --remove_leakage --device cuda:1
python run_experiments.py --task_type regression --dataset_name nba --sens_attr country --remove_leakage --device cuda:1
