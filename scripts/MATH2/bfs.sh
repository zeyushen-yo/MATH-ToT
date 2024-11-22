python3.11 -B run.py \
    --backend Llama3.1-8B-Instruct \
    --task MATH^2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --method_select greedy \
    --n_generate_sample 4 \
    --n_evaluate_sample 2 \
    --n_select_sample 2 \
    ${@}