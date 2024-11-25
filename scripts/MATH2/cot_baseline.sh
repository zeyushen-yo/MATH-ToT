python3 run.py \
    --backend Qwen2.5-1.5B-Instruct \
    --task MATH2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 20 \
    ${@}