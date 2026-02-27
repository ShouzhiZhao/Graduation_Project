export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd)/simulation_agent
export TARGET_MOVIE_ID=5

cd simulation_agent

gpu_count=$(nvidia-smi --list-gpus | wc -l)
time=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="output/run_simulation/$time"

mkdir -p "$log_dir"

for ((i=0; i<$gpu_count; i++)); do
    nohup env CUDA_VISIBLE_DEVICES=$i python run_simulation.py \
        --config_file config/config.yaml \
        --output_file "${time}_run_simulation_messages_$((i)).json" \
        --log_file "-" \
        > "$log_dir/nohup_run_simulation_$((i)).log" 2>&1 &
done

for ((i=0; i<$gpu_count; i++)); do
    nohup env CUDA_VISIBLE_DEVICES=$i python run_simulation.py \
        --config_file config/config_1.yaml \
        --output_file "${time}_run_simulation_messages_$((i + $gpu_count)).json" \
        --log_file "-" \
        > "$log_dir/nohup_run_simulation_$((i + $gpu_count)).log" 2>&1 &
done

for ((i=0; i<$gpu_count; i++)); do
    nohup env CUDA_VISIBLE_DEVICES=$i python run_simulation.py \
        --config_file config/config_2.yaml \
        --output_file "${time}_run_simulation_messages_$((i + $gpu_count * 2)).json" \
        --log_file "-" \
        > "$log_dir/nohup_run_simulation_$((i + $gpu_count * 2)).log" 2>&1 &
done

for ((i=0; i<$gpu_count; i++)); do
    nohup env CUDA_VISIBLE_DEVICES=$i python run_simulation.py \
        --config_file config/config.yaml \
        --output_file "${time}_run_simulation_messages_$((i + $gpu_count * 3)).json" \
        --log_file "-" \
        > "$log_dir/nohup_run_simulation_$((i + $gpu_count * 3)).log" 2>&1 &
done
