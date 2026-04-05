# classification
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr region \
    --remove_leakage \
    --backbone GCN \
    --model baseline \
    --device cuda:1 --runs 10 --lambda_fair 0.1


python run_exp.py --task_type classification --dataset_name pokec_z --sens_attr region
python run_exp.py --task_type classification --dataset_name pokec_z --sens_attr gender 
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr region 
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr gender 
python run_exp.py --task_type classification --dataset_name german --sens_attr Gender 
python run_exp.py --task_type classification --dataset_name nba --sens_attr country 


# regression
python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr region \
    --remove_leakage \
    --backbone GCN \
    --model baseline \
    --device cuda:1 --runs 10

python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr region
python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr gender 
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr region 
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr gender
python run_exp.py --task_type regression --dataset_name german --sens_attr Gender 
python run_exp.py --task_type regression --dataset_name nba --sens_attr country 