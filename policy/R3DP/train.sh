# 
task_name=${1}
head_camera_type=${2}
front_camera_type=L515
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

alg_name=robot_dp
# task choices: See TASK.md
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

if [ ! -d "./data/${task_name}_${head_camera_type}_${expert_data_num}.zarr" ]; then
    echo "zarr does not exist, run pkl2zarr_r3dp.py"
    cd ../..
    expert_data_num_minus_one=$((expert_data_num - 1))
    if [ ! -d "./data/${task_name}_${head_camera_type}_pkl/episode${expert_data_num_minus_one}" ]; then
        echo "error: expert data does not exist"
        exit 1
    else
        python script/pkl2zarr_r3dp.py ${task_name} ${head_camera_type} ${expert_data_num}
        cd policy/R3DP
    fi
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
num_gpus=$(echo "$gpu_id" | tr ',' '\n' | wc -l)
accelerate launch --main_process_port 29505 --num_processes ${num_gpus} --gpu_ids ${gpu_id} --mixed_precision bf16 train.py --config-name=${config_name}.yaml \
        task.name=${task_name} \
        task.dataset.zarr_path="./data/${task_name}_${head_camera_type}_${expert_data_num}.zarr" \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        head_camera_type=${head_camera_type} \
        front_camera_type="L515"\
        expert_data_num=${expert_data_num}

# bash train.sh block_hammer_beat L515 100 0 0,1,2,3