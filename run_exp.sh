# classification
# --backbone GCN / SGC / GraphSAGE 
# --model baseline / multi / gate

python run_exp.py --task_type classification \
        --remove_leakage  --device cuda:0 \
        --backbone SGC --model multi \
        --dataset_name pokec_n --sens_attr region
        

python run_exp.py --task_type classification --dataset_name pokec_z --sens_attr region
python run_exp.py --task_type classification --dataset_name pokec_z --sens_attr gender 
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr region 
python run_exp.py --task_type classification --dataset_name pokec_n --sens_attr gender 
python run_exp.py --task_type classification --dataset_name german --sens_attr Gender 
python run_exp.py --task_type classification --dataset_name nba --sens_attr country 


# regression
python run_exp.py --task_type classification \
        --remove_leakage  --device cuda:0 \
        --backbone GraphSAGE --model baseline \
        --dataset_name pokec_n --sens_attr region


python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr region
python run_exp.py --task_type regression --dataset_name pokec_z --sens_attr gender 
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr region 
python run_exp.py --task_type regression --dataset_name pokec_n --sens_attr gender
python run_exp.py --task_type regression --dataset_name german --sens_attr Gender 
python run_exp.py --task_type regression --dataset_name nba --sens_attr country 